package com.ple;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

/**
 * Statistical Analysis Job implementing a Map-Side Join architecture.
 * * <p><strong>Purpose:</strong> Enriches the raw Edge list with global statistics derived from the Nodes dataset
 * to compute the "Null Model" prediction (expected frequency assuming independence).</p>
 * * <p><strong>Architecture (Map-Side Join):</strong>
 * <ul>
 * <li><strong>Setup Phase:</strong> Each Mapper loads the entire 'Nodes' dataset (small) into an in-memory HashMap.</li>
 * <li><strong>Map Phase:</strong> Streams the 'Edges' dataset (large). For each edge, looks up source/target statistics in memory.</li>
 * <li><strong>Reduce Phase:</strong> Disabled (0 reducers). This eliminates the Shuffle/Sort phase, 
 * providing a significant performance boost compared to a standard Reduce-Side Join.</li>
 * </ul>
 */
public class Stats extends Configured implements Tool {

  /**
   * Core Mapper class. Handles both data loading (side-data) and stream processing.
   */
  public static class MapSideJoinMapper
    extends Mapper<LongWritable, Text, NullWritable, Text> {

    /** In-memory lookup table: Archetype ID -> Total Occurrences */
    private final Map<String, Long> nodesCache = new HashMap<>();
    
    /** Total number of games played (sum of all archetype occurrences), used for probability calculation. */
    private long totalGamesPlayed = 0;
    
    private final Text outValue = new Text();

    /**
     * Initializes the Mapper.
     * Loads the 'Nodes' dataset from HDFS into the local RAM cache.
     * This method is executed once per Mapper instance before processing records.
     */
    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
      System.out.println("SETUP: Loading Nodes into in-memory cache...");
      
      Configuration conf = context.getConfiguration();
      String nodesPathStr = conf.get("stats.nodes.path");
      Path nodesDir = new Path(nodesPathStr);
      FileSystem fs = FileSystem.get(conf);

      // Retrieve all part-files from the Nodes directory
      FileStatus[] files = fs.globStatus(new Path(nodesDir, "part-*"));
      
      if (files == null || files.length == 0) {
          System.err.println("ERROR: No node files found in " + nodesDir);
          return;
      }

      long countLoaded = 0;
      
      // Read each file and populate the HashMap
      for (FileStatus file : files) {
        try (FSDataInputStream in = fs.open(file.getPath());
             BufferedReader br = new BufferedReader(new InputStreamReader(in))) {
             
          String line;
          while ((line = br.readLine()) != null) {
            // Expected format: archetype;count;wins
            String[] parts = line.split(";");
            if (parts.length >= 2) {
              String archetype = parts[0];
              try {
                  long count = Long.parseLong(parts[1]);
                  nodesCache.put(archetype, count);
                  totalGamesPlayed += count;
                  countLoaded++;
              } catch (NumberFormatException e) {
                  // Ignore malformed lines to prevent job failure
              }
            }
          }
        }
      }
      
      // Report cache size for debugging/monitoring
      context.getCounter("Stats", "Nodes Loaded in RAM").increment(countLoaded);
      System.out.printf("SETUP COMPLETE: %d nodes loaded. Total Games Aggregated: %d%n", countLoaded, totalGamesPlayed);
    }

    /**
     * Processes a single Edge record.
     * Joins the edge data with the in-memory Node statistics to compute the prediction.
     */
    @Override
    protected void map(LongWritable key, Text value, Context context)
      throws IOException, InterruptedException {
      
      // Input Format: source;target;count;wins
      String line = value.toString().trim();
      String[] parts = line.split(";");
      
      // Basic validation
      if (parts.length < 3) return;

      String source = parts[0];
      String target = parts[1];
      String edgeCountStr = parts[2];
      String winsStr = (parts.length > 3) ? parts[3] : "0";

      // --- JOIN LOGIC (In-Memory Lookup) ---
      Long countSourceObj = nodesCache.get(source);
      Long countTargetObj = nodesCache.get(target);

      // Handle potentially missing keys (safety check)
      long countSource = (countSourceObj != null) ? countSourceObj : 0;
      long countTarget = (countTargetObj != null) ? countTargetObj : 0;

      // Only emit valid data points where both archetypes exist in the dictionary
      if (countSource > 0 && countTarget > 0 && totalGamesPlayed > 0) {
          
          // --- PREDICTION CALCULATION ---
          // Theoretical frequency based on independence assumption:
          // P(A n B) = P(A) * P(B) => Expected Count = (CountA * CountB) / Total_Games
          double prediction = (double) (countSource * countTarget) / totalGamesPlayed;

          // --- OUTPUT GENERATION ---
          // Format: source;target;observed_count;wins;total_source;total_target;prediction
          String output = String.format(
            "%s;%s;%s;%s;%d;%d;%.4f",
            source,
            target,
            edgeCountStr,
            winsStr,
            countSource,
            countTarget,
            prediction
          );

          outValue.set(output);
          context.write(NullWritable.get(), outValue);
          context.getCounter("Stats", "Predictions Generated").increment(1);
      }
    }
  }

  @Override
  public int run(String[] args) throws Exception {
    if (args.length < 2) {
      System.err.println("Usage: Stats <input_base> <output>");
      System.err.println("  input_base: Parent directory containing /nodes and /edges");
      System.err.println("  output: Destination directory for the CSV result");
      return 1;
    }

    String inputBase = args[0];
    String output = args[1];

    Configuration conf = getConf();
    FileSystem fs = FileSystem.get(conf);
    
    Path nodesDir = new Path(inputBase + "/nodes");
    Path edgesDir = new Path(inputBase + "/edges");
    Path outputPath = new Path(output);

    // Validate input existence
    if (!fs.exists(nodesDir)) {
        System.err.println("CRITICAL ERROR: Nodes directory not found at " + nodesDir);
        return 1;
    }
    if (!fs.exists(edgesDir)) {
        System.err.println("CRITICAL ERROR: Edges directory not found at " + edgesDir);
        return 1;
    }

    // Cleanup previous output
    if (fs.exists(outputPath)) {
      fs.delete(outputPath, true);
    }

    // Pass the Nodes directory path to Mappers via Configuration
    conf.set("stats.nodes.path", nodesDir.toString());
    
    // Enable Output Compression (Snappy) to optimize disk I/O
    //conf.setBoolean("mapreduce.output.fileoutputformat.compress", true);
    //conf.set("mapreduce.output.fileoutputformat.compress.codec", "org.apache.hadoop.io.compress.SnappyCodec");

    Job job = Job.getInstance(conf, "Stats Generation - Map Side Join");
    job.setJarByClass(Stats.class);

    // INPUT: Only 'Edges' are fed to the MapReduce engine as the main stream
    FileInputFormat.addInputPath(job, edgesDir);
    job.setInputFormatClass(TextInputFormat.class);

    // MAPPER CONFIGURATION
    job.setMapperClass(MapSideJoinMapper.class);
    job.setMapOutputKeyClass(NullWritable.class);
    job.setMapOutputValueClass(Text.class);
    
    // REDUCER CONFIGURATION
    // Set to 0 to enable Map-Only mode (disables Shuffle/Sort)
    job.setNumReduceTasks(0); 

    // OUTPUT CONFIGURATION
    job.setOutputKeyClass(NullWritable.class);
    job.setOutputValueClass(Text.class);
    job.setOutputFormatClass(TextOutputFormat.class);
    FileOutputFormat.setOutputPath(job, outputPath);

    System.out.println("╔════════════════════════════════════════╗");
    System.out.println("║  STATS JOB INITIALIZED (Map-Side Join) ║");
    System.out.println("╚════════════════════════════════════════╝");
    System.out.println("Side Data (RAM):  " + nodesDir);
    System.out.println("Stream Data:      " + edgesDir);
    System.out.println("Output:           " + outputPath);

    long t0 = System.currentTimeMillis();
    boolean success = job.waitForCompletion(true);
    long t1 = System.currentTimeMillis();

    if (success) {
        System.out.println("\n✅ Job Completed Successfully in " + (t1 - t0)/1000.0 + " s");
        System.out.println("  - Nodes Loaded: " + job.getCounters().findCounter("Stats", "Nodes Loaded in RAM").getValue());
        System.out.println("  - Records Generated: " + job.getCounters().findCounter("Stats", "Predictions Generated").getValue());
    }

    return success ? 0 : 1;
  }

  public static void main(String[] args) throws Exception {
    System.exit(ToolRunner.run(new Stats(), args));
  }
}