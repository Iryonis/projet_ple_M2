package com.ple;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

/**
 * OPTIMIZED MapReduce for statistical prediction generation.
 *
 * ARCHITECTURE: Single Job with Distributed Cache
 * - Reads nodes data (archetype counts) into distributed cache
 * - Processes edges data and enriches each edge with:
 *   - source archetype count
 *   - target archetype count
 *   - prediction = (count_source × count_target) / total_archetypes
 */
public class Stats extends Configured implements Tool {

  /**
   * Mapper that reads edges and enriches them with archetype counts and predictions.
   * Uses archetype counts loaded from distributed cache.
   */
  public static class StatsMapper
    extends Mapper<LongWritable, Text, NullWritable, Text> {

    // Map to store archetype counts: archetype -> total_count
    private final Map<String, Long> archetypeCounts = new HashMap<>();
    private long totalArchetypes = 0;
    private final Text outValue = new Text();

    /**
     * Setup phase: Load archetype counts from distributed cache.
     * Calculates total number of archetypes for normalization.
     */
    @Override
    protected void setup(Context context)
      throws IOException, InterruptedException {
      Configuration conf = context.getConfiguration();
      String nodesPath = conf.get("stats.nodes.path");

      if (nodesPath == null) {
        throw new IOException("stats.nodes.path configuration is missing");
      }

      // Load archetype counts from nodes files
      FileSystem fs = FileSystem.get(conf);
      Path nodesDir = new Path(nodesPath);

      if (!fs.exists(nodesDir)) {
        throw new IOException("Nodes directory does not exist: " + nodesPath);
      }

      // Read all part files in nodes directory
      FileStatus[] statuses = fs.listStatus(nodesDir);
      for (FileStatus status : statuses) {
        if (status.getPath().getName().startsWith("part-r-")) {
          loadNodeCounts(fs, status.getPath());
        }
      }

      context
        .getCounter("Stats", "Unique Archetypes Loaded")
        .increment(archetypeCounts.size());
      context
        .getCounter("Stats", "Total Archetype Count")
        .increment(totalArchetypes);
    }

    /**
     * Load archetype counts from a single nodes file.
     * Format: archetype;count;wins
     */
    private void loadNodeCounts(FileSystem fs, Path filePath)
      throws IOException {
      try (
        BufferedReader reader = new BufferedReader(
          new InputStreamReader(fs.open(filePath))
        )
      ) {
        String line;
        while ((line = reader.readLine()) != null) {
          String[] parts = line.split(";");
          if (parts.length >= 2) {
            String archetype = parts[0].trim();
            long count = Long.parseLong(parts[1].trim());

            // Accumulate count for this archetype
            archetypeCounts.merge(archetype, count, Long::sum);
            totalArchetypes += count;
          }
        }
      }
    }

    /**
     * Map phase: Process each edge and enrich with counts and prediction.
     * Input format: source;target;count;wins
     * Output format: source;target;count;wins;count_source;count_target;prediction
     */
    @Override
    protected void map(LongWritable key, Text value, Context context)
      throws IOException, InterruptedException {
      String line = value.toString().trim();

      // Parse edge data: source;target;count;wins
      String[] parts = line.split(";");
      String source = parts[0];
      String target = parts[1];
      String count = parts[2];
      String wins = parts[3];

      // Lookup archetype counts
      Long countSource = archetypeCounts.get(source);
      Long countTarget = archetypeCounts.get(target);

      // Calculate prediction: (count_source × count_target) / total_archetypes
      double prediction = (double) (countSource * countTarget) /
      totalArchetypes;

      // Format output: source;target;count;wins;count_source;count_target;prediction
      String output = String.format(
        "%s;%s;%s;%s;%d;%d;%.1f",
        source,
        target,
        count,
        wins,
        countSource,
        countTarget,
        prediction
      );

      outValue.set(output);
      context.getCounter("Stats", "Edges Processed").increment(1);
      context.write(NullWritable.get(), outValue);
    }
  }

  /**
   * Main execution method for the Stats job.
   */
  @Override
  public int run(String[] args) throws Exception {
    if (args.length < 2) {
      System.err.println("Usage: Stats <input_base> <output>");
      System.err.println(
        "  input_base:  Base directory containing /nodes and /edges subdirectories"
      );
      System.err.println("  output:      Output directory for statistics");
      System.err.println();
      System.err.println("Example: Stats /output/k4 /output/stats");
      return 1;
    }

    String inputBase = args[0];
    String output = args[1];

    Configuration conf = getConf();

    // Set nodes path in configuration for mapper to access
    String nodesPath = inputBase + "/nodes";
    conf.set("stats.nodes.path", nodesPath);

    // Validate input directories exist
    FileSystem fs = FileSystem.get(conf);
    Path nodesDir = new Path(nodesPath);
    Path edgesDir = new Path(inputBase + "/edges");

    if (!fs.exists(nodesDir)) {
      System.err.println("ERROR: Nodes directory does not exist: " + nodesPath);
      return 1;
    }

    if (!fs.exists(edgesDir)) {
      System.err.println(
        "ERROR: Edges directory does not exist: " + inputBase + "/edges"
      );
      return 1;
    }

    System.out.println(
      "╔════════════════════════════════════════════════════╗"
    );
    System.out.println(
      "║         STATS GENERATION - PART III                ║"
    );
    System.out.println(
      "╚════════════════════════════════════════════════════╝"
    );
    System.out.println("Input nodes:  " + nodesPath);
    System.out.println("Input edges:  " + inputBase + "/edges");
    System.out.println("Output:       " + output);
    System.out.println();

    long t0 = System.currentTimeMillis();

    // Delete output directory if exists
    Path outputPath = new Path(output);
    if (fs.exists(outputPath)) {
      fs.delete(outputPath, true);
    }

    // Configure job
    Job job = Job.getInstance(conf, "Stats Generation");
    job.setJarByClass(Stats.class);

    // Input: edges directory
    FileInputFormat.addInputPath(job, edgesDir);

    // Mapper configuration
    job.setMapperClass(StatsMapper.class);
    job.setMapOutputKeyClass(NullWritable.class);
    job.setMapOutputValueClass(Text.class);

    // No reducer needed - direct output from mapper
    job.setNumReduceTasks(0);

    // Output configuration
    job.setOutputKeyClass(NullWritable.class);
    job.setOutputValueClass(Text.class);
    FileOutputFormat.setOutputPath(job, outputPath);

    // Execute job
    boolean success = job.waitForCompletion(true);

    if (!success) {
      System.err.println("ERROR: Stats job failed!");
      return 1;
    }

    long t1 = System.currentTimeMillis();

    // Print statistics
    System.out.println(
      "\n╔════════════════════════════════════════════════════╗"
    );
    System.out.println(
      "║                STATS COMPLETED                     ║"
    );
    System.out.println(
      "╚════════════════════════════════════════════════════╝"
    );
    System.out.printf(
      "Total time:          %.1f seconds%n",
      (t1 - t0) / 1000.0
    );
    System.out.printf(
      "Edges processed:     %d%n",
      job.getCounters().findCounter("Stats", "Edges Processed").getValue()
    );
    System.out.printf(
      "Unique archetypes:   %d%n",
      job
        .getCounters()
        .findCounter("Stats", "Unique Archetypes Loaded")
        .getValue()
    );
    System.out.printf(
      "Total arch. count:   %d%n",
      job.getCounters().findCounter("Stats", "Total Archetype Count").getValue()
    );
    System.out.println("\nOutput: " + output);

    return 0;
  }

  /**
   * Main entry point.
   */
  public static void main(String[] args) throws Exception {
    System.exit(ToolRunner.run(new Stats(), args));
  }
}
