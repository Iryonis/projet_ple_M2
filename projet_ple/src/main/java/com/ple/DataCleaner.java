package com.ple;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.io.IOException;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.compress.SnappyCodec;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

/**
 * MapReduce job to clean raw game data and remove duplicates.
 * Outputs cleaned data in SequenceFile format for optimal performance in subsequent jobs.
 */
public class DataCleaner extends Configured implements Tool {

  /**
   * Custom counters for detailed performance metrics.
   */
  public enum Counters {
    MAPPER_INPUT_BYTES,
    MAPPER_OUTPUT_BYTES,
    MAPPER_TIME_MS,
    COMBINER_INPUT_BYTES,
    COMBINER_OUTPUT_BYTES,
    COMBINER_TIME_MS,
    REDUCER_INPUT_BYTES,
    REDUCER_OUTPUT_BYTES,
    REDUCER_TIME_MS,
  }

  /**
   * Mapper that validates and extracts a unique key for each game entry.
   */
  public static class CleanMapper
    extends Mapper<LongWritable, Text, Text, Text> {

    private final Gson gson = new Gson();
    private final Text outputKey = new Text();
    private final Text outputValue = new Text();

    @Override
    protected void map(LongWritable key, Text value, Context context)
      throws IOException, InterruptedException {
      long startTime = System.nanoTime();
      String line = value.toString().trim();

      // Count input bytes
      context
        .getCounter(Counters.MAPPER_INPUT_BYTES)
        .increment(value.getLength());

      // Parse JSON
      JsonObject game;
      try {
        game = gson.fromJson(line, JsonObject.class);
        if (game == null) {
          return;
        }
      } catch (Exception e) {
        return;
      }

      // Validate and extract unique key
      String uniqueKey = validateAndExtract(game);
      if (uniqueKey == null) {
        return;
      }

      // Emit sorted player utags, date (YYYY-MM-DD), and round as key, full line as value
      outputKey.set(uniqueKey);
      outputValue.set(line);

      context.write(outputKey, outputValue);
      context.getCounter("DataCleaner", "Mapper Output Lines").increment(1);

      // Count output bytes (key + value)
      context
        .getCounter(Counters.MAPPER_OUTPUT_BYTES)
        .increment(outputKey.getLength() + outputValue.getLength());

      // Measure execution time
      long endTime = System.nanoTime();
      context
        .getCounter(Counters.MAPPER_TIME_MS)
        .increment((endTime - startTime) / 1_000_000);
    }

    /**
     * Validates the game JSON object and extracts a unique key.
     * The key is composed of sorted player utags, date (YYYY-MM-DD), and round.
     *
     * @param game JsonObject representing the game
     * @return unique key string or null if invalid
     */
    private String validateAndExtract(JsonObject game) {
      try {
        // Extract players array and check if it has exactly 2 players
        JsonArray players = game.getAsJsonArray("players");
        if (players == null || players.size() != 2) {
          return null;
        }

        // Extract the two player objects
        JsonObject player0 = players.get(0).getAsJsonObject();
        JsonObject player1 = players.get(1).getAsJsonObject();

        // Extract required fields (utags, decks, date, round)
        String utag0 = player0.get("utag").getAsString();
        String utag1 = player1.get("utag").getAsString();
        String deck0 = player0.get("deck").getAsString();
        String deck1 = player1.get("deck").getAsString();
        String dateStr = game.get("date").getAsString();
        int round = game.get("round").getAsInt();

        // Validate that each deck has exactly 16 characters (8 cards × 2 hex chars)
        if (deck0.length() != 16 || deck1.length() != 16) {
          return null;
        }

        // Extract date part without time (YYYY-MM-DD)
        String dateOnly = dateStr.substring(0, 10);

        // Sort utags to normalize the key
        String sortedUtag0, sortedUtag1;
        if (utag0.compareTo(utag1) < 0) {
          sortedUtag0 = utag0;
          sortedUtag1 = utag1;
        } else {
          sortedUtag0 = utag1;
          sortedUtag1 = utag0;
        }

        // Return the composite key
        return String.format(
          "%s|%s|%s|%d",
          sortedUtag0,
          sortedUtag1,
          dateOnly,
          round
        );
      } catch (Exception e) {
        return null;
      }
    }
  }

  /**
   * Combiner that removes exact duplicates locally to reduce data sent to Reducer.
   */
  public static class CleanCombiner extends Reducer<Text, Text, Text, Text> {

    private final Set<String> seenLines = new HashSet<>();

    @Override
    protected void reduce(Text key, Iterable<Text> values, Context context)
      throws IOException, InterruptedException {
      long startTime = System.nanoTime();
      // Clear the set for each key
      seenLines.clear();

      // For each value, emit only unique lines (not already seen)
      for (Text value : values) {
        String line = value.toString();

        // Count input bytes
        context
          .getCounter(Counters.COMBINER_INPUT_BYTES)
          .increment(key.getLength() + value.getLength());

        if (!seenLines.contains(line)) {
          seenLines.add(line);
          context.write(key, value);
          context
            .getCounter("DataCleaner", "Combiner Output Lines")
            .increment(1);

          // Count output bytes
          context
            .getCounter(Counters.COMBINER_OUTPUT_BYTES)
            .increment(key.getLength() + value.getLength());
        } else {
          context
            .getCounter("DataCleaner", "Combiner Exact Duplicates Removed")
            .increment(1);
        }
      }

      // Measure execution time
      long endTime = System.nanoTime();
      context
        .getCounter(Counters.COMBINER_TIME_MS)
        .increment((endTime - startTime) / 1_000_000);
    }
  }

  /**
   * Reducer that eliminates exact duplicates and games with timestamps within ±10 seconds.
   */
  public static class CleanReducer
    extends Reducer<Text, Text, NullWritable, Text> {

    private final Gson gson = new Gson();
    private static final long TIME_THRESHOLD_SECONDS = 10;
    private final Set<String> seenLines = new HashSet<>();

    @Override
    protected void reduce(Text key, Iterable<Text> values, Context context)
      throws IOException, InterruptedException {
      long startTime = System.nanoTime();
      // Clear the set for each key
      seenLines.clear();
      List<GameEntry> games = new ArrayList<>();

      // Remove exact duplicates and parse timestamps
      for (Text value : values) {
        String line = value.toString();

        // Count input bytes
        context
          .getCounter(Counters.REDUCER_INPUT_BYTES)
          .increment(key.getLength() + value.getLength());

        if (!seenLines.contains(line)) {
          seenLines.add(line);
          // Parse JSON and extract timestamp
          // If parsing fails, emit the original line as-is
          try {
            JsonObject game = gson.fromJson(line, JsonObject.class);
            String dateStr = game.get("date").getAsString();
            Instant timestamp = Instant.parse(dateStr);
            games.add(new GameEntry(line, timestamp));
          } catch (Exception e) {
            context
              .getCounter("DataCleaner", "Reducer Invalid Lines Skipped")
              .increment(1);
          }
        } else {
          context
            .getCounter("DataCleaner", "Reducer Exact Duplicates Removed")
            .increment(1);
        }
      }

      // Sort by timestamp (ascending order) with tie-breaker for deterministic results
      games.sort((a, b) -> {
        int timestampCompare = a.timestamp.compareTo(b.timestamp);
        if (timestampCompare != 0) return timestampCompare;
        return a.line.compareTo(b.line); // Tie-breaker: lexicographic order
      });

      // Remove games with close timestamps (±10 seconds window)
      for (int i = 0; i < games.size(); i++) {
        GameEntry current = games.get(i);
        boolean isDuplicate = false;

        // Check only previous games within the time threshold (backward search)
        for (int j = i - 1; j >= 0; j--) {
          GameEntry previous = games.get(j);
          long secondsDiff = ChronoUnit.SECONDS.between(
            previous.timestamp,
            current.timestamp
          );

          // Break early if outside the time window (optimization)
          if (secondsDiff > TIME_THRESHOLD_SECONDS) {
            break;
          }

          isDuplicate = true;
          context
            .getCounter("DataCleaner", "Reducer Temporal Duplicates Removed")
            .increment(1);
        }

        if (!isDuplicate) {
          Text outputValue = new Text(current.line);
          context.write(NullWritable.get(), outputValue);
          context
            .getCounter("DataCleaner", "Reducer Output Lines")
            .increment(1);

          // Count output bytes
          context
            .getCounter(Counters.REDUCER_OUTPUT_BYTES)
            .increment(outputValue.getLength());
        }
      }

      // Measure execution time
      long endTime = System.nanoTime();
      context
        .getCounter(Counters.REDUCER_TIME_MS)
        .increment((endTime - startTime) / 1_000_000);
    }

    /**
     * Helper class to store game line and its timestamp.
     */
    private static class GameEntry {

      final String line;
      final Instant timestamp;

      GameEntry(String line, Instant timestamp) {
        this.line = line;
        this.timestamp = timestamp;
      }
    }
  }

  @Override
  public int run(String[] args) throws Exception {
    if (args.length < 2 || args.length > 3) {
      System.err.println("Usage: DataCleaner <input> <output> [numReducers]");
      System.err.println("  numReducers: optional, default is 1");
      return 1;
    }

    long startTime = System.currentTimeMillis();
    System.out.println("=== Starting Data Cleaner Job ===");

    Configuration conf = getConf();
    Job job = Job.getInstance(conf, "Data Cleaner");
    job.setJarByClass(DataCleaner.class);

    // Mapper configuration
    job.setMapperClass(CleanMapper.class);
    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(Text.class);

    // Combiner configuration
    job.setCombinerClass(CleanCombiner.class);

    // Reducer configuration
    job.setReducerClass(CleanReducer.class);
    job.setOutputKeyClass(NullWritable.class);
    job.setOutputValueClass(Text.class);

    // Input/Output format configuration
    job.setInputFormatClass(TextInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);

    // Enable Snappy compression for SequenceFile
    SequenceFileOutputFormat.setCompressOutput(job, true);
    SequenceFileOutputFormat.setOutputCompressorClass(job, SnappyCodec.class);
    SequenceFileOutputFormat.setOutputCompressionType(
      job,
      SequenceFile.CompressionType.BLOCK
    );

    // Determine number of reducers
    int numReducers = 1; // Default value
    if (args.length == 3) {
      try {
        numReducers = Integer.parseInt(args[2]);
        if (numReducers < 1) {
          System.err.println("Error: numReducers must be >= 1");
          return 1;
        }
      } catch (NumberFormatException e) {
        System.err.println("Error: numReducers must be an integer");
        return 1;
      }
    }
    job.setNumReduceTasks(numReducers);
    System.out.println("Number of Reducers: " + numReducers);

    // Input/Output paths
    Path outputPath = new Path(args[1]);
    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, outputPath);

    // Delete output directory if it exists
    org.apache.hadoop.fs.FileSystem fs = org.apache.hadoop.fs.FileSystem.get(
      conf
    );
    if (fs.exists(outputPath)) {
      fs.delete(outputPath, true);
      System.out.println("Output directory deleted: " + args[1]);
    }

    boolean success = job.waitForCompletion(true);

    long endTime = System.currentTimeMillis();
    long durationMs = endTime - startTime;
    long seconds = durationMs / 1000;
    long minutes = seconds / 60;
    long remainingSeconds = seconds % 60;

    System.out.println("\n=== Job Completed ===");
    System.out.printf(
      "Total execution time: %d min %d sec (%.2f sec)%n",
      minutes,
      remainingSeconds,
      durationMs / 1000.0
    );

    // Display detailed performance metrics
    if (success) {
      org.apache.hadoop.mapreduce.Counters counters = job.getCounters();

      System.out.println("\n=== Performance Metrics ===");

      // Mapper metrics
      long mapperInputBytes = counters
        .findCounter(Counters.MAPPER_INPUT_BYTES)
        .getValue();
      long mapperOutputBytes = counters
        .findCounter(Counters.MAPPER_OUTPUT_BYTES)
        .getValue();
      long mapperTimeMs = counters
        .findCounter(Counters.MAPPER_TIME_MS)
        .getValue();
      System.out.println("\nMapper:");
      System.out.printf(
        "  Input:  %.2f MB (%d bytes)%n",
        mapperInputBytes / 1_000_000.0,
        mapperInputBytes
      );
      System.out.printf(
        "  Output: %.2f MB (%d bytes)%n",
        mapperOutputBytes / 1_000_000.0,
        mapperOutputBytes
      );
      System.out.printf(
        "  Time:   %.2f seconds (%d ms)%n",
        mapperTimeMs / 1000.0,
        mapperTimeMs
      );

      // Combiner metrics
      long combinerInputBytes = counters
        .findCounter(Counters.COMBINER_INPUT_BYTES)
        .getValue();
      long combinerOutputBytes = counters
        .findCounter(Counters.COMBINER_OUTPUT_BYTES)
        .getValue();
      long combinerTimeMs = counters
        .findCounter(Counters.COMBINER_TIME_MS)
        .getValue();
      if (combinerInputBytes > 0) {
        System.out.println("\nCombiner:");
        System.out.printf(
          "  Input:  %.2f MB (%d bytes)%n",
          combinerInputBytes / 1_000_000.0,
          combinerInputBytes
        );
        System.out.printf(
          "  Output: %.2f MB (%d bytes)%n",
          combinerOutputBytes / 1_000_000.0,
          combinerOutputBytes
        );
        System.out.printf(
          "  Time:   %.2f seconds (%d ms)%n",
          combinerTimeMs / 1000.0,
          combinerTimeMs
        );
        System.out.printf(
          "  Reduction: %.2f%%%n",
          (1 - (double) combinerOutputBytes / combinerInputBytes) * 100
        );
      }

      // Reducer metrics
      long reducerInputBytes = counters
        .findCounter(Counters.REDUCER_INPUT_BYTES)
        .getValue();
      long reducerOutputBytes = counters
        .findCounter(Counters.REDUCER_OUTPUT_BYTES)
        .getValue();
      long reducerTimeMs = counters
        .findCounter(Counters.REDUCER_TIME_MS)
        .getValue();
      System.out.println("\nReducer:");
      System.out.printf(
        "  Input:  %.2f MB (%d bytes)%n",
        reducerInputBytes / 1_000_000.0,
        reducerInputBytes
      );
      System.out.printf(
        "  Output: %.2f MB (%d bytes)%n",
        reducerOutputBytes / 1_000_000.0,
        reducerOutputBytes
      );
      System.out.printf(
        "  Time:   %.2f seconds (%d ms)%n",
        reducerTimeMs / 1000.0,
        reducerTimeMs
      );
      System.out.printf(
        "  Reduction: %.2f%%%n",
        (1 - (double) reducerOutputBytes / reducerInputBytes) * 100
      );

      // Overall statistics
      System.out.println("\n=== Overall Statistics ===");
      System.out.printf(
        "  Total input:  %.2f MB%n",
        mapperInputBytes / 1_000_000.0
      );
      System.out.printf(
        "  Total output: %.2f MB%n",
        reducerOutputBytes / 1_000_000.0
      );
      System.out.printf(
        "  Total reduction: %.2f%%%n",
        (1 - (double) reducerOutputBytes / mapperInputBytes) * 100
      );
      System.out.printf(
        "  Processing time: %.2f seconds (Mapper: %.2f, Combiner: %.2f, Reducer: %.2f)%n",
        (mapperTimeMs + combinerTimeMs + reducerTimeMs) / 1000.0,
        mapperTimeMs / 1000.0,
        combinerTimeMs / 1000.0,
        reducerTimeMs / 1000.0
      );
    }

    return success ? 0 : 1;
  }

  public static void main(String[] args) throws Exception {
    System.exit(ToolRunner.run(new DataCleaner(), args));
  }
}
