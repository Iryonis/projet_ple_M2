package com.ple;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

/**
 * MapReduce job to generate nodes and edges from cleaned game data.
 *
 * Nodes: All unique decks AND archetypes (sub-decks of 1-7 cards) with their occurrence count and win count.
 * Edges: All deck matchups with match count and win count (directional).
 *
 * This job runs two sequential MapReduce jobs:
 * 1. NodesJob: Generates deck and archetype statistics (archetype;count;wins)
 * 2. EdgesJob: Generates matchup statistics (source;target;count;wins)
 */
public class NodesEdgesGenerator extends Configured implements Tool {

  // ==================== NODES JOB ====================

  /**
   * Mapper for nodes generation with IN-MAPPER COMBINING optimization.
   * OPTIMIZATION: Local aggregation dramatically reduces network traffic from ~51M to ~few M emissions.
   *
   * Reads SequenceFile input (NullWritable, Text) from DataCleaner output.
   * Emits:
   * - Full deck (8 cards)
   * - All archetypes (sub-decks of configurable size range)
   * Format: (archetype, "count,wins") aggregated locally per split
   */
  public static class NodesMapper
    extends Mapper<NullWritable, Text, Text, Text> {

    private final Gson gson = new Gson();
    private final Text outputKey = new Text();
    private final Text outputValue = new Text();

    // OPTIMIZATION: In-mapper combining - local aggregation before emission
    // Reduces emissions from 51M to ~few million (10-20x reduction)
    private Map<String, long[]> localAggregation;

    // CONFIGURATION: Archetype size range (configurable via CLI)
    private int minArchetypeSize;
    private int maxArchetypeSize;
    private boolean generateFullDeck;

    /**
     * Setup method to initialize local aggregation and read configuration.
     * OPTIMIZATION: Pre-allocate HashMap with estimated capacity to reduce resizing.
     */
    @Override
    protected void setup(Context context)
      throws IOException, InterruptedException {
      super.setup(context);

      // OPTIMIZATION: Pre-allocate with estimated capacity (avg ~1000 unique archetypes per split)
      localAggregation = new HashMap<>(2000, 0.75f);

      // READ CONFIGURATION: Archetype size range from job configuration
      Configuration conf = context.getConfiguration();
      minArchetypeSize = conf.getInt("archetype.min.size", 1);
      maxArchetypeSize = conf.getInt("archetype.max.size", 7);
      generateFullDeck = conf.getBoolean("archetype.include.full.deck", true);

      System.out.println(
        "NodesMapper configured: archetypes size [" +
        minArchetypeSize +
        "-" +
        maxArchetypeSize +
        "], full deck: " +
        generateFullDeck
      );
    }

    /**
     * Map function that processes each game line.
     * OPTIMIZATION: Aggregates locally instead of emitting for each archetype.
     *
     * @param key     NullWritable from SequenceFile
     * @param value   JSON game data line
     * @param context MapReduce context
     * @throws IOException          on I/O errors
     * @throws InterruptedException on thread interruption
     */
    @Override
    protected void map(NullWritable key, Text value, Context context)
      throws IOException, InterruptedException {
      String line = value.toString().trim();

      // Parse JSON
      JsonObject game;
      try {
        game = gson.fromJson(line, JsonObject.class);
        if (game == null) {
          return;
        }
      } catch (Exception e) {
        return; // Skip invalid JSON
      }

      // Extract players array and winner
      JsonArray players;
      int winner;
      try {
        players = game.getAsJsonArray("players");
        winner = game.get("winner").getAsInt();
        if (players == null || players.size() != 2) {
          return;
        }
      } catch (Exception e) {
        return; // Skip if players array is invalid
      }

      // Process both players
      for (int i = 0; i < 2; i++) {
        try {
          JsonObject player = players.get(i).getAsJsonObject();
          String deck = player.get("deck").getAsString();
          boolean win = (i == winner);

          // Generate all archetypes (sub-decks) for this deck
          List<String> archetypes = generateArchetypes(deck);

          // OPTIMIZATION: Aggregate locally instead of immediate emission
          for (String archetype : archetypes) {
            long[] stats = localAggregation.get(archetype);
            if (stats == null) {
              stats = new long[2]; // [count, wins]
              localAggregation.put(archetype, stats);
            }
            stats[0]++; // Increment count
            if (win) {
              stats[1]++; // Increment wins if player won
            }
          }
        } catch (Exception e) {
          continue; // Skip this player if parsing fails
        }
      }
    }

    /**
     * Cleanup method to emit aggregated results.
     * OPTIMIZATION: Emit once per unique archetype instead of once per occurrence.
     * This reduces emissions from ~51M to ~few million (massive network traffic reduction).
     */
    @Override
    protected void cleanup(Context context)
      throws IOException, InterruptedException {
      // Emit all locally aggregated statistics
      for (Map.Entry<String, long[]> entry : localAggregation.entrySet()) {
        outputKey.set(entry.getKey());
        long[] stats = entry.getValue();
        outputValue.set(stats[0] + "," + stats[1]); // count,wins
        context.write(outputKey, outputValue);
      }

      // Clear map to free memory
      localAggregation.clear();

      super.cleanup(context);
    }

    /**
     * Generates archetypes (sub-decks) from a full deck with configurable size range.
     * CONFIGURATION: Size range controlled by minArchetypeSize and maxArchetypeSize.
     * OPTIMIZATION: Allows reducing archetype generation (e.g., only 3-6 cards for performance).
     * Cards are kept in sorted order to normalize archetypes.
     *
     * @param deck The full deck as a 16-character hex string (8 cards × 2 chars)
     * @return List of all archetype strings within configured size range
     */
    private List<String> generateArchetypes(String deck) {
      List<String> archetypes = new ArrayList<>();

      // Parse deck into individual cards (2 hex chars each)
      List<String> cards = new ArrayList<>();
      for (int i = 0; i < deck.length(); i += 2) {
        cards.add(deck.substring(i, i + 2));
      }

      // Sort cards to normalize archetypes
      Collections.sort(cards);

      // CONFIGURATION: Generate combinations within specified size range
      for (
        int size = minArchetypeSize;
        size <= Math.min(maxArchetypeSize, cards.size() - 1);
        size++
      ) {
        generateCombinations(cards, size, 0, new ArrayList<>(), archetypes);
      }

      // CONFIGURATION: Optionally add the full deck (8 cards)
      if (generateFullDeck) {
        archetypes.add(String.join("", cards));
      }

      return archetypes;
    }

    /**
     * Recursive helper to generate all combinations of a given size.
     *
     * @param cards       List of all cards
     * @param size        Target combination size
     * @param start       Current starting index
     * @param current     Current combination being built
     * @param archetypes  Output list to collect archetypes
     */
    private void generateCombinations(
      List<String> cards,
      int size,
      int start,
      List<String> current,
      List<String> archetypes
    ) {
      if (current.size() == size) {
        archetypes.add(String.join("", current));
        return;
      }

      for (int i = start; i < cards.size(); i++) {
        current.add(cards.get(i));
        generateCombinations(cards, size, i + 1, current, archetypes);
        current.remove(current.size() - 1);
      }
    }
  }

  /**
   * Combiner for nodes generation (local aggregation).
   * Sums count and wins for each deck locally before sending to reducer.
   */
  public static class NodesCombiner extends Reducer<Text, Text, Text, Text> {

    private final Text outputValue = new Text();

    /**
     * Reduce function that aggregates local statistics.
     *
     * @param key     Deck identifier
     * @param values  Iterator of "count,wins" strings
     * @param context MapReduce context
     * @throws IOException          on I/O errors
     * @throws InterruptedException on thread interruption
     */
    @Override
    protected void reduce(Text key, Iterable<Text> values, Context context)
      throws IOException, InterruptedException {
      long totalCount = 0;
      long totalWins = 0;

      // Sum all count and wins values
      for (Text value : values) {
        String[] parts = value.toString().split(",");
        if (parts.length == 2) {
          totalCount += Long.parseLong(parts[0]);
          totalWins += Long.parseLong(parts[1]);
        }
      }

      outputValue.set(totalCount + "," + totalWins);
      context.write(key, outputValue);
    }
  }

  /**
   * Reducer for nodes generation (global aggregation).
   * Outputs final deck statistics in CSV format: deck;count;wins
   */
  public static class NodesReducer extends Reducer<Text, Text, Text, Text> {

    private final Text outputValue = new Text();

    /**
     * Reduce function that outputs final statistics.
     *
     * @param key     Deck identifier
     * @param values  Iterator of "count,wins" strings
     * @param context MapReduce context
     * @throws IOException          on I/O errors
     * @throws InterruptedException on thread interruption
     */
    @Override
    protected void reduce(Text key, Iterable<Text> values, Context context)
      throws IOException, InterruptedException {
      long totalCount = 0;
      long totalWins = 0;

      // Sum all count and wins values
      for (Text value : values) {
        String[] parts = value.toString().split(",");
        if (parts.length == 2) {
          totalCount += Long.parseLong(parts[0]);
          totalWins += Long.parseLong(parts[1]);
        }
      }

      // Output format: deck;count;wins
      outputValue.set(key.toString() + ";" + totalCount + ";" + totalWins);
      context.write(null, outputValue);
    }
  }

  // ==================== EDGES JOB ====================

  /**
   * Mapper for edges generation with IN-MAPPER COMBINING optimization.
   * OPTIMIZATION: Local aggregation reduces emissions and network traffic.
   *
   * Reads SequenceFile input (NullWritable, Text) from DataCleaner output.
   * Emits (source|target, "count,wins") aggregated locally per split.
   * Each match generates two edges (one from each player's perspective).
   */
  public static class EdgesMapper
    extends Mapper<NullWritable, Text, Text, Text> {

    private final Gson gson = new Gson();
    private final Text outputKey = new Text();
    private final Text outputValue = new Text();

    // OPTIMIZATION: In-mapper combining - local aggregation for edges
    private Map<String, long[]> localAggregation;

    /**
     * Setup method to initialize local aggregation.
     * OPTIMIZATION: Pre-allocate HashMap to reduce resizing overhead.
     */
    @Override
    protected void setup(Context context)
      throws IOException, InterruptedException {
      super.setup(context);
      // OPTIMIZATION: Pre-allocate with estimated capacity
      localAggregation = new HashMap<>(5000, 0.75f);
    }

    /**
     * Map function that processes each game line.
     * OPTIMIZATION: Aggregates locally instead of immediate emission.
     *
     * @param key     NullWritable from SequenceFile
     * @param value   JSON game data line
     * @param context MapReduce context
     * @throws IOException          on I/O errors
     * @throws InterruptedException on thread interruption
     */
    @Override
    protected void map(NullWritable key, Text value, Context context)
      throws IOException, InterruptedException {
      String line = value.toString().trim();

      // Parse JSON
      JsonObject game;
      try {
        game = gson.fromJson(line, JsonObject.class);
        if (game == null) {
          return;
        }
      } catch (Exception e) {
        return; // Skip invalid JSON
      }

      // Extract players array and winner
      JsonArray players;
      int winner;
      try {
        players = game.getAsJsonArray("players");
        winner = game.get("winner").getAsInt();
        if (players == null || players.size() != 2) {
          return;
        }
      } catch (Exception e) {
        return; // Skip if players array is invalid
      }

      // Extract both players' data
      try {
        JsonObject player0 = players.get(0).getAsJsonObject();
        JsonObject player1 = players.get(1).getAsJsonObject();

        String deck0 = player0.get("deck").getAsString();
        String deck1 = player1.get("deck").getAsString();
        boolean win0 = (winner == 0);
        boolean win1 = (winner == 1);

        // OPTIMIZATION: Aggregate edge from player0's perspective locally
        String edge0 = deck0 + "|" + deck1;
        long[] stats0 = localAggregation.get(edge0);
        if (stats0 == null) {
          stats0 = new long[2]; // [count, wins]
          localAggregation.put(edge0, stats0);
        }
        stats0[0]++; // Increment count
        if (win0) {
          stats0[1]++; // Increment wins
        }

        // OPTIMIZATION: Aggregate edge from player1's perspective locally
        String edge1 = deck1 + "|" + deck0;
        long[] stats1 = localAggregation.get(edge1);
        if (stats1 == null) {
          stats1 = new long[2]; // [count, wins]
          localAggregation.put(edge1, stats1);
        }
        stats1[0]++; // Increment count
        if (win1) {
          stats1[1]++; // Increment wins
        }
      } catch (Exception e) {
        return; // Skip if player data is invalid
      }
    }

    /**
     * Cleanup method to emit aggregated edge results.
     * OPTIMIZATION: Emit once per unique edge instead of once per match.
     */
    @Override
    protected void cleanup(Context context)
      throws IOException, InterruptedException {
      // Emit all locally aggregated edge statistics
      for (Map.Entry<String, long[]> entry : localAggregation.entrySet()) {
        outputKey.set(entry.getKey());
        long[] stats = entry.getValue();
        outputValue.set(stats[0] + "," + stats[1]); // count,wins
        context.write(outputKey, outputValue);
      }

      // Clear map to free memory
      localAggregation.clear();

      super.cleanup(context);
    }
  }

  /**
   * Combiner for edges generation (local aggregation).
   * Sums count and wins for each matchup locally before sending to reducer.
   */
  public static class EdgesCombiner extends Reducer<Text, Text, Text, Text> {

    private final Text outputValue = new Text();

    /**
     * Reduce function that aggregates local statistics.
     *
     * @param key     Edge identifier (source|target)
     * @param values  Iterator of "count,wins" strings
     * @param context MapReduce context
     * @throws IOException          on I/O errors
     * @throws InterruptedException on thread interruption
     */
    @Override
    protected void reduce(Text key, Iterable<Text> values, Context context)
      throws IOException, InterruptedException {
      long totalCount = 0;
      long totalWins = 0;

      // Sum all count and wins values
      for (Text value : values) {
        String[] parts = value.toString().split(",");
        if (parts.length == 2) {
          totalCount += Long.parseLong(parts[0]);
          totalWins += Long.parseLong(parts[1]);
        }
      }

      outputValue.set(totalCount + "," + totalWins);
      context.write(key, outputValue);
    }
  }

  /**
   * Reducer for edges generation (global aggregation).
   * Outputs final matchup statistics in CSV format: source;target;count;wins
   */
  public static class EdgesReducer extends Reducer<Text, Text, Text, Text> {

    private final Text outputValue = new Text();

    /**
     * Reduce function that outputs final statistics.
     *
     * @param key     Edge identifier (source|target)
     * @param values  Iterator of "count,wins" strings
     * @param context MapReduce context
     * @throws IOException          on I/O errors
     * @throws InterruptedException on thread interruption
     */
    @Override
    protected void reduce(Text key, Iterable<Text> values, Context context)
      throws IOException, InterruptedException {
      long totalCount = 0;
      long totalWins = 0;

      // Sum all count and wins values
      for (Text value : values) {
        String[] parts = value.toString().split(",");
        if (parts.length == 2) {
          totalCount += Long.parseLong(parts[0]);
          totalWins += Long.parseLong(parts[1]);
        }
      }

      // Parse key to extract source and target
      String[] decks = key.toString().split("\\|");
      if (decks.length != 2) {
        return; // Skip invalid keys
      }

      // Output format: source;target;count;wins
      outputValue.set(
        decks[0] + ";" + decks[1] + ";" + totalCount + ";" + totalWins
      );
      context.write(null, outputValue);
    }
  }

  // ==================== MAIN DRIVER ====================

  /**
   * Run method that executes both jobs sequentially.
   * CONFIGURATION: Accepts archetype size range parameters for optimization.
   *
   * @param args Command-line arguments: [input] [nodesOutput] [edgesOutput] [numReducers] [minArchetype] [maxArchetype]
   * @return 0 on success, 1 on failure
   * @throws Exception on job execution errors
   */
  @Override
  public int run(String[] args) throws Exception {
    if (args.length < 3 || args.length > 6) {
      System.err.println(
        "Usage: NodesEdgesGenerator <input> <nodesOutput> <edgesOutput> [numReducers] [minArchetype] [maxArchetype]"
      );
      System.err.println(
        "  input:         Path to cleaned game data (SequenceFile)"
      );
      System.err.println("  nodesOutput:   Path for nodes output");
      System.err.println("  edgesOutput:   Path for edges output");
      System.err.println(
        "  numReducers:   Optional, number of reducers (default: 1, recommended: 10-20 for 100k+ records)"
      );
      System.err.println(
        "  minArchetype:  Optional, minimum archetype size (default: 1, recommended: 3-4 for performance)"
      );
      System.err.println(
        "  maxArchetype:  Optional, maximum archetype size (default: 7, max: 7)"
      );
      System.err.println("");
      System.err.println("PERFORMANCE TIPS:");
      System.err.println(
        "  - For 100k records: use numReducers=10-20, minArchetype=3, maxArchetype=6"
      );
      System.err.println(
        "  - For 1M records: use numReducers=50, minArchetype=4, maxArchetype=6"
      );
      System.err.println(
        "  - Limiting archetype range reduces processing time by 50-80%"
      );
      return 1;
    }

    Configuration conf = getConf();

    // HDFS OPTIMIZATION: Configure for maximum performance
    // Increase sort buffer to reduce disk spills during shuffle
    conf.setInt("io.sort.mb", 512); // Default is 100MB, increase to 512MB
    conf.setFloat("io.sort.spill.percent", 0.9f); // Default is 0.8, delay spilling

    // HDFS OPTIMIZATION: Enable map output compression to reduce network traffic
    conf.setBoolean("mapreduce.map.output.compress", true);
    conf.set(
      "mapreduce.map.output.compress.codec",
      "org.apache.hadoop.io.compress.SnappyCodec"
    );

    // HDFS OPTIMIZATION: Increase buffer sizes for better I/O performance
    conf.setInt("io.file.buffer.size", 131072); // 128KB, default is 4KB

    // HDFS OPTIMIZATION: Configure reduce task memory
    conf.set("mapreduce.reduce.memory.mb", "2048"); // 2GB per reducer
    conf.set("mapreduce.reduce.java.opts", "-Xmx1638m"); // 80% of reducer memory

    int numReducers = 1;
    int minArchetypeSize = 1;
    int maxArchetypeSize = 7;

    // Parse optional numReducers parameter
    if (args.length >= 4) {
      try {
        numReducers = Integer.parseInt(args[3]);
        if (numReducers < 1) {
          System.err.println("Error: numReducers must be >= 1");
          return 1;
        }
      } catch (NumberFormatException e) {
        System.err.println("Error: numReducers must be an integer");
        return 1;
      }
    }

    // CONFIGURATION: Parse optional minArchetype parameter
    if (args.length >= 5) {
      try {
        minArchetypeSize = Integer.parseInt(args[4]);
        if (minArchetypeSize < 1 || minArchetypeSize > 8) {
          System.err.println("Error: minArchetype must be between 1 and 8");
          return 1;
        }
      } catch (NumberFormatException e) {
        System.err.println("Error: minArchetype must be an integer");
        return 1;
      }
    }

    // CONFIGURATION: Parse optional maxArchetype parameter
    if (args.length >= 6) {
      try {
        maxArchetypeSize = Integer.parseInt(args[5]);
        if (maxArchetypeSize < 1 || maxArchetypeSize > 7) {
          System.err.println("Error: maxArchetype must be between 1 and 7");
          return 1;
        }
        if (maxArchetypeSize < minArchetypeSize) {
          System.err.println("Error: maxArchetype must be >= minArchetype");
          return 1;
        }
      } catch (NumberFormatException e) {
        System.err.println("Error: maxArchetype must be an integer");
        return 1;
      }
    }

    // CONFIGURATION: Set archetype parameters in configuration for mappers
    conf.setInt("archetype.min.size", minArchetypeSize);
    conf.setInt("archetype.max.size", maxArchetypeSize);
    conf.setBoolean("archetype.include.full.deck", true);

    long globalStartTime = System.currentTimeMillis();
    System.out.println("=== Starting Nodes & Edges Generation ===");
    System.out.println("CONFIGURATION:");
    System.out.println("  Number of Reducers: " + numReducers);
    System.out.println(
      "  Archetype Size Range: [" +
      minArchetypeSize +
      "-" +
      maxArchetypeSize +
      "]"
    );
    System.out.println("  Include Full Deck: true");
    System.out.println("HDFS OPTIMIZATIONS:");
    System.out.println("  Map Output Compression: Snappy");
    System.out.println("  Sort Buffer: 512 MB");
    System.out.println("  File Buffer: 128 KB");
    System.out.println("  In-Mapper Combining: Enabled");

    // ========== JOB 1: NODES GENERATION ==========
    System.out.println("\n--- Job 1/2: Generating Nodes ---");
    long nodesStartTime = System.currentTimeMillis();

    Job nodesJob = Job.getInstance(conf, "Nodes Generator");
    nodesJob.setJarByClass(NodesEdgesGenerator.class);

    // Input format: SequenceFile from DataCleaner
    nodesJob.setInputFormatClass(SequenceFileInputFormat.class);

    // Mapper configuration
    nodesJob.setMapperClass(NodesMapper.class);
    nodesJob.setMapOutputKeyClass(Text.class);
    nodesJob.setMapOutputValueClass(Text.class);

    // Combiner configuration
    nodesJob.setCombinerClass(NodesCombiner.class);

    // Reducer configuration
    nodesJob.setReducerClass(NodesReducer.class);
    nodesJob.setOutputKeyClass(Text.class);
    nodesJob.setOutputValueClass(Text.class);
    nodesJob.setNumReduceTasks(numReducers);

    // Input/Output paths
    Path nodesOutputPath = new Path(args[1]);
    FileInputFormat.addInputPath(nodesJob, new Path(args[0]));
    FileOutputFormat.setOutputPath(nodesJob, nodesOutputPath);

    // Delete output directory if exists
    org.apache.hadoop.fs.FileSystem fs = org.apache.hadoop.fs.FileSystem.get(
      conf
    );
    if (fs.exists(nodesOutputPath)) {
      fs.delete(nodesOutputPath, true);
      System.out.println("Deleted existing nodes output: " + args[1]);
    }

    // Run nodes job
    boolean nodesSuccess = nodesJob.waitForCompletion(true);
    long nodesEndTime = System.currentTimeMillis();
    long nodesDuration = nodesEndTime - nodesStartTime;

    if (!nodesSuccess) {
      System.err.println("Nodes job failed!");
      return 1;
    }

    System.out.printf(
      "Nodes job completed in %.2f seconds%n",
      nodesDuration / 1000.0
    );

    // ========== JOB 2: EDGES GENERATION ==========
    System.out.println("\n--- Job 2/2: Generating Edges ---");
    long edgesStartTime = System.currentTimeMillis();

    Job edgesJob = Job.getInstance(conf, "Edges Generator");
    edgesJob.setJarByClass(NodesEdgesGenerator.class);

    // Input format: SequenceFile from DataCleaner
    edgesJob.setInputFormatClass(SequenceFileInputFormat.class);

    // Mapper configuration
    edgesJob.setMapperClass(EdgesMapper.class);
    edgesJob.setMapOutputKeyClass(Text.class);
    edgesJob.setMapOutputValueClass(Text.class);

    // Combiner configuration
    edgesJob.setCombinerClass(EdgesCombiner.class);

    // Reducer configuration
    edgesJob.setReducerClass(EdgesReducer.class);
    edgesJob.setOutputKeyClass(Text.class);
    edgesJob.setOutputValueClass(Text.class);
    edgesJob.setNumReduceTasks(numReducers);

    // Input/Output paths
    Path edgesOutputPath = new Path(args[2]);
    FileInputFormat.addInputPath(edgesJob, new Path(args[0]));
    FileOutputFormat.setOutputPath(edgesJob, edgesOutputPath);

    // Delete output directory if exists
    if (fs.exists(edgesOutputPath)) {
      fs.delete(edgesOutputPath, true);
      System.out.println("Deleted existing edges output: " + args[2]);
    }

    // Run edges job
    boolean edgesSuccess = edgesJob.waitForCompletion(true);
    long edgesEndTime = System.currentTimeMillis();
    long edgesDuration = edgesEndTime - edgesStartTime;

    if (!edgesSuccess) {
      System.err.println("Edges job failed!");
      return 1;
    }

    System.out.printf(
      "Edges job completed in %.2f seconds%n",
      edgesDuration / 1000.0
    );

    // ========== SUMMARY ==========
    long globalEndTime = System.currentTimeMillis();
    long totalDuration = globalEndTime - globalStartTime;

    System.out.println("\n=== All Jobs Completed Successfully ===");
    System.out.printf("Nodes job: %.2f sec%n", nodesDuration / 1000.0);
    System.out.printf("Edges job: %.2f sec%n", edgesDuration / 1000.0);
    System.out.printf("Total time: %.2f sec%n", totalDuration / 1000.0);

    return 0;
  }

  /**
   * Main entry point for the application.
   *
   * @param args Command-line arguments
   * @throws Exception on execution errors
   */
  public static void main(String[] args) throws Exception {
    System.exit(ToolRunner.run(new NodesEdgesGenerator(), args));
  }
}
