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
      String line = value.toString().trim();

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
      // Clear the set for each key
      seenLines.clear();

      // For each value, emit only unique lines (not already seen)
      for (Text value : values) {
        String line = value.toString();

        if (!seenLines.contains(line)) {
          seenLines.add(line);
          context.write(key, value);
          context
            .getCounter("DataCleaner", "Combiner Output Lines")
            .increment(1);
        } else {
          context
            .getCounter("DataCleaner", "Combiner Exact Duplicates Removed")
            .increment(1);
        }
      }
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
      // Clear the set for each key
      seenLines.clear();
      List<GameEntry> games = new ArrayList<>();

      // Remove exact duplicates and parse timestamps
      for (Text value : values) {
        String line = value.toString();

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
            context.write(NullWritable.get(), value);
            context
              .getCounter("DataCleaner", "Reducer Output Lines")
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
          context.write(NullWritable.get(), new Text(current.line));
          context
            .getCounter("DataCleaner", "Reducer Output Lines")
            .increment(1);
        }
      }
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
      "Execution time: %d min %d sec (%.2f sec)%n",
      minutes,
      remainingSeconds,
      durationMs / 1000.0
    );

    return success ? 0 : 1;
  }

  public static void main(String[] args) throws Exception {
    System.exit(ToolRunner.run(new DataCleaner(), args));
  }
}