package com.ple;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
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
   * Mapper for nodes generation.
   * Reads SequenceFile input (NullWritable, Text) from DataCleaner output.
   * Emits:
   * - Full deck (8 cards)
   * - All archetypes (sub-decks of 1-7 cards)
   * Format: (archetype, "1,1") for wins, (archetype, "1,0") for losses
   */
  public static class NodesMapper
    extends Mapper<NullWritable, Text, Text, Text> {

    private final Gson gson = new Gson();
    private final Text outputKey = new Text();
    private final Text outputValue = new Text();

    /**
     * Map function that processes each game line.
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
          
          String valueStr = win ? "1,1" : "1,0";
          outputValue.set(valueStr);
          
          // Emit all archetypes
          for (String archetype : archetypes) {
            outputKey.set(archetype);
            context.write(outputKey, outputValue);
          }
        } catch (Exception e) {
          continue; // Skip this player if parsing fails
        }
      }
    }
    
    /**
     * Generates all archetypes (sub-decks) from a full deck.
     * An archetype is any combination of 1 to 7 cards from the 8-card deck.
     * Cards are kept in sorted order to normalize archetypes.
     * 
     * @param deck The full deck as a 16-character hex string (8 cards × 2 chars)
     * @return List of all archetype strings
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
      
      // Generate all combinations of size 1 to 7
      for (int size = 1; size <= 7; size++) {
        generateCombinations(cards, size, 0, new ArrayList<>(), archetypes);
      }
      
      // Also add the full deck (8 cards)
      archetypes.add(String.join("", cards));
      
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
   * Mapper for edges generation.
   * Reads SequenceFile input (NullWritable, Text) from DataCleaner output.
   * Emits (source|target, "1,1") for wins and (source|target, "1,0") for losses.
   * Each match generates two edges (one from each player's perspective).
   */
  public static class EdgesMapper
    extends Mapper<NullWritable, Text, Text, Text> {

    private final Gson gson = new Gson();
    private final Text outputKey = new Text();
    private final Text outputValue = new Text();

    /**
     * Map function that processes each game line.
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

        // Emit edge from player0's perspective: deck0 vs deck1
        outputKey.set(deck0 + "|" + deck1);
        outputValue.set(win0 ? "1,1" : "1,0");
        context.write(outputKey, outputValue);

        // Emit edge from player1's perspective: deck1 vs deck0
        outputKey.set(deck1 + "|" + deck0);
        outputValue.set(win1 ? "1,1" : "1,0");
        context.write(outputKey, outputValue);
      } catch (Exception e) {
        return; // Skip if player data is invalid
      }
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
   *
   * @param args Command-line arguments: [input] [nodesOutput] [edgesOutput] [numReducers]
   * @return 0 on success, 1 on failure
   * @throws Exception on job execution errors
   */
  @Override
  public int run(String[] args) throws Exception {
    if (args.length < 3 || args.length > 4) {
      System.err.println(
        "Usage: NodesEdgesGenerator <input> <nodesOutput> <edgesOutput> [numReducers]"
      );
      System.err.println(
        "  input:        Path to cleaned game data (SequenceFile)"
      );
      System.err.println("  nodesOutput:  Path for nodes output");
      System.err.println("  edgesOutput:  Path for edges output");
      System.err.println("  numReducers:  Optional, default is 1");
      return 1;
    }

    Configuration conf = getConf();
    int numReducers = 1;

    // Parse optional numReducers parameter
    if (args.length == 4) {
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

    long globalStartTime = System.currentTimeMillis();
    System.out.println("=== Starting Nodes & Edges Generation ===");
    System.out.println("Number of Reducers: " + numReducers);

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
