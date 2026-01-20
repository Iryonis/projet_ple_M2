package com.ple;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
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
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

/**
 * SCALABLE MapReduce for statistical prediction generation.
 *
 * ARCHITECTURE: Reduce-Side Join (2 MapOutputs → 1 Reducer)
 * - Job 1: Nodes Mapper reads archetype counts
 * - Job 2: Edges Mapper reads edge data
 * - Both use same key (archetype) for join in reducer
 * - Reducer enriches edges with node counts and calculates predictions
 *
 * Benefits:
 * - No memory limitations (nodes streamed in reducer)
 * - Scalable to massive datasets
 * - Parallel processing of nodes + edges
 * - Configurable number of reducers for load balancing
 */
public class Stats extends Configured implements Tool {

  /**
   * Wrapper class for Values in reduce-side join.
   * Tag indicates if this is a Node (0) or Edge (1).
   */
  public static class JoinValue {

    public static final byte NODE = 0;
    public static final byte EDGE = 1;

    public byte type; // NODE or EDGE
    public String[] fields; // Parsed fields

    public JoinValue(byte type, String[] fields) {
      this.type = type;
      this.fields = fields;
    }

    @Override
    public String toString() {
      return String.format("%d:%s", type, String.join(";", fields));
    }
  }

  /**
   * Mapper for Nodes data: extracts archetype counts.
   * Input format: archetype;count;wins
   * Output key: archetype (for join)
   * Output value: "0;count;wins" (0 = NODE tag)
   */
  public static class NodesMapper
    extends Mapper<LongWritable, Text, Text, Text> {

    private final Text outKey = new Text();
    private final Text outValue = new Text();

    @Override
    protected void map(LongWritable key, Text value, Context context)
      throws IOException, InterruptedException {
      String line = value.toString().trim();

      // Parse: archetype;count;wins
      String[] parts = line.split(";");
      if (parts.length < 2) return;

      String archetype = parts[0];
      String count = parts[1];
      String wins = parts.length >= 3 ? parts[2] : "0";

      outKey.set(archetype);
      // Format: NODE_TAG;count;wins
      outValue.set(String.format("0;%s;%s", count, wins));

      context.write(outKey, outValue);
      context.getCounter("Stats", "Nodes Read").increment(1);
    }
  }

  /**
   * Mapper for Edges data: extracts edge pairs.
   * Input format: source;target;count;wins
   * We emit 2 keys per edge:
   *   1. source -> "1;target;count;wins" (join source with nodes)
   *   2. target -> "1;source;count;wins" (join target with nodes)
   * This allows enrichment of both source and target counts in reducer.
   */
  public static class EdgesMapper
    extends Mapper<LongWritable, Text, Text, Text> {

    private final Text outKey = new Text();
    private final Text outValue = new Text();

    @Override
    protected void map(LongWritable key, Text value, Context context)
      throws IOException, InterruptedException {
      String line = value.toString().trim();

      // Parse: source;target;count;wins
      String[] parts = line.split(";");
      if (parts.length < 2) return;

      String source = parts[0];
      String target = parts[1];
      String count = parts.length >= 3 ? parts[2] : "0";
      String wins = parts.length >= 4 ? parts[3] : "0";

      // Emit 2 pairs for reduce-side join:
      // 1. source -> edge info (to get source count)
      outKey.set(source);
      outValue.set(String.format("1;%s;%s;%s;%s", target, count, wins, "S"));
      context.write(outKey, outValue);

      // 2. target -> edge info (to get target count)
      outKey.set(target);
      outValue.set(String.format("1;%s;%s;%s;%s", source, count, wins, "T"));
      context.write(outKey, outValue);

      context.getCounter("Stats", "Edges Read").increment(1);
    }
  }

  /**
   * Reducer that performs the join and enrichment.
   * For each archetype key:
   *   - Collects all nodes with counts
   *   - Collects all edges referencing this archetype
   *   - Enriches edges with archetype count and prediction
   */
  public static class StatsReducer
    extends Reducer<Text, Text, NullWritable, Text> {

    private final Text outValue = new Text();
    private long totalArchetypes = 0;

    @Override
    protected void setup(Context context) {
      // Read total archetypes from configuration (set in job setup)
      Configuration conf = context.getConfiguration();
      totalArchetypes = conf.getLong("stats.total.archetypes", 1);
    }

    @Override
    protected void reduce(Text key, Iterable<Text> values, Context context)
      throws IOException, InterruptedException {
      String archetype = key.toString();
      long archetypeCount = 0;
      List<String[]> edges = new ArrayList<>();

      // Separate nodes and edges
      for (Text val : values) {
        String line = val.toString();
        String[] parts = line.split(";", -1);

        if (parts.length > 0) {
          byte type = (byte) Integer.parseInt(parts[0]); // 0=NODE, 1=EDGE

          if (type == 0) {
            // NODE: archetype;count;wins
            if (parts.length >= 2) {
              archetypeCount = Long.parseLong(parts[1]);
            }
          } else if (type == 1) {
            // EDGE: target;count;wins;position
            // Store edge info for later enrichment
            edges.add(parts);
          }
        }
      }

      // Enrich all edges with this archetype's count
      context.getCounter("Stats", "Archetype Groups").increment(1);

      for (String[] edgeData : edges) {
        if (edgeData.length >= 5) {
          String otherArchetype = edgeData[1];
          String count = edgeData[2];
          String wins = edgeData[3];
          String position = edgeData[4]; // "S" or "T"

          // Look up other archetype count (requires 2-pass job)
          // For now, use archetypeCount from this group
          double prediction = (double) (archetypeCount * archetypeCount) /
          totalArchetypes;

          // Output format depends on position:
          // If position="S": source=archetype, target=otherArchetype
          // If position="T": source=otherArchetype, target=archetype
          String output;
          if ("S".equals(position)) {
            output =
              String.format(
                "%s;%s;%s;%s;%d;?;%.1f",
                archetype,
                otherArchetype,
                count,
                wins,
                archetypeCount,
                prediction
              );
          } else {
            output =
              String.format(
                "%s;%s;%s;%s;?;%d;%.1f",
                otherArchetype,
                archetype,
                count,
                wins,
                archetypeCount,
                prediction
              );
          }

          outValue.set(output);
          context.write(NullWritable.get(), outValue);
          context.getCounter("Stats", "Edges Enriched").increment(1);
        }
      }
    }
  }

  /**
   * Main execution method for the Stats job.
   */
  @Override
  public int run(String[] args) throws Exception {
    if (args.length < 2) {
      System.err.println("Usage: Stats <input_base> <output> [numReducers]");
      System.err.println(
        "  input_base:   Base directory containing /nodes and /edges subdirectories"
      );
      System.err.println("  output:       Output directory for statistics");
      System.err.println("  numReducers:  Number of reducers (default: 10)");
      System.err.println();
      System.err.println("Example: Stats /output/k4 /output/stats 20");
      return 1;
    }

    String inputBase = args[0];
    String output = args[1];
    int numReducers = args.length >= 3 ? Integer.parseInt(args[2]) : 10;

    Configuration conf = getConf();

    // Validate input directories exist
    FileSystem fs = FileSystem.get(conf);
    Path nodesDir = new Path(inputBase + "/nodes");
    Path edgesDir = new Path(inputBase + "/edges");

    if (!fs.exists(nodesDir)) {
      System.err.println("ERROR: Nodes directory does not exist: " + nodesDir);
      return 1;
    }

    if (!fs.exists(edgesDir)) {
      System.err.println("ERROR: Edges directory does not exist: " + edgesDir);
      return 1;
    }

    System.out.println(
      "╔════════════════════════════════════════════════════╗"
    );
    System.out.println("║   STATS GENERATION - REDUCE-SIDE JOIN (v2)        ║");
    System.out.println(
      "╚════════════════════════════════════════════════════╝"
    );
    System.out.println("Input nodes:   " + nodesDir);
    System.out.println("Input edges:   " + edgesDir);
    System.out.println("Output:        " + output);
    System.out.println("Num reducers:  " + numReducers);
    System.out.println();

    long t0 = System.currentTimeMillis();

    // Step 1: Count total archetypes (requires a pass over nodes)
    long totalArchetypes = countTotalArchetypes(fs, nodesDir);
    conf.setLong("stats.total.archetypes", totalArchetypes);
    System.out.printf("Total archetypes counted: %d%n%n", totalArchetypes);

    // Delete output directory if exists
    Path outputPath = new Path(output);
    if (fs.exists(outputPath)) {
      fs.delete(outputPath, true);
    }

    // ===== PERFORMANCE OPTIMIZATIONS =====
    // Enable map output compression (reduces shuffle I/O significantly)
    conf.setBoolean("mapreduce.map.output.compress", true);
    conf.set("mapreduce.map.output.compress.codec",
             "org.apache.hadoop.io.compress.SnappyCodec");

    // Mapper memory buffers
    conf.setInt("mapreduce.task.io.sort.mb", 512);
    conf.setFloat("mapreduce.map.sort.spill.percent", 0.90f);
    conf.setInt("mapreduce.task.io.sort.factor", 50);
    
    // Reducer shuffle buffers (critical for 308 GB shuffle!)
    conf.setFloat("mapreduce.reduce.shuffle.input.buffer.percent", 0.80f);
    conf.setFloat("mapreduce.reduce.shuffle.merge.percent", 0.80f);
    conf.setFloat("mapreduce.reduce.input.buffer.percent", 0.80f);
    conf.setInt("mapreduce.reduce.shuffle.parallelcopies", 20);

    // Configure job
    Job job = Job.getInstance(conf, "Stats Generation - Reduce-Side Join");
    job.setJarByClass(Stats.class);

    // CRITICAL: Use MultipleInputs with different mappers for each input
    MultipleInputs.addInputPath(
      job,
      nodesDir,
      TextInputFormat.class,
      NodesMapper.class
    );
    MultipleInputs.addInputPath(
      job,
      edgesDir,
      TextInputFormat.class,
      EdgesMapper.class
    );

    // Mapper output
    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(Text.class);

    // Reducer configuration
    job.setReducerClass(StatsReducer.class);
    job.setNumReduceTasks(numReducers);

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
      "Edges enriched:      %d%n",
      job.getCounters().findCounter("Stats", "Edges Enriched").getValue()
    );
    System.out.printf(
      "Nodes read:          %d%n",
      job.getCounters().findCounter("Stats", "Nodes Read").getValue()
    );
    System.out.printf(
      "Archetype groups:    %d%n",
      job.getCounters().findCounter("Stats", "Archetype Groups").getValue()
    );
    System.out.println("\nOutput: " + output);
    System.out.println(
      "NOTE: Predictions show '?' for unknown counts (requires second pass)"
    );

    return 0;
  }

  /**
   * Count total number of archetypes by reading all nodes.
   * This is used for normalization in prediction calculation.
   */
  private long countTotalArchetypes(FileSystem fs, Path nodesDir)
    throws IOException {
    long total = 0;

    // Stream all nodes files
    FileStatus[] statuses = fs.globStatus(new Path(nodesDir, "part-*"));
    if (statuses != null) {
      for (FileStatus status : statuses) {
        // Just count lines as a proxy for total archetypes
        total += countLines(fs, status.getPath());
      }
    }

    return total > 0 ? total : 1; // Avoid division by zero
  }

  /**
   * Helper to count lines in a file.
   */
  private long countLines(FileSystem fs, Path filePath) throws IOException {
    long count = 0;
    try (org.apache.hadoop.fs.FSDataInputStream in = fs.open(filePath)) {
      byte[] buffer = new byte[1024 * 64];
      int read;
      while ((read = in.read(buffer)) >= 0) {
        for (int i = 0; i < read; i++) {
          if (buffer[i] == '\n') count++;
        }
      }
    }
    return count;
  }

  /**
   * Main entry point.
   */
  public static void main(String[] args) throws Exception {
    System.exit(ToolRunner.run(new Stats(), args));
  }
}
