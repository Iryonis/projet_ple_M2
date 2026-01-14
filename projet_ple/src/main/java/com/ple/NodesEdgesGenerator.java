package com.ple;

import static com.ple.ArchetypeUtils.binomial;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

/**
 * MapReduce driver for nodes and edges generation.
 *
 * ARCHITECTURE: 2 SEPARATE JOBS
 * - Job 1 (Nodes): Generates C(8,k) archetypes per deck with statistics
 * - Job 2 (Edges): Generates C(8,k)² matchups between archetypes
 *
 * KEY OPTIMIZATIONS:
 * - In-mapper combining (local HashMap aggregation)
 * - Binary archetype encoding (long instead of String)
 * - Raw comparator for EdgeKey (avoids deserialization)
 * - Snappy compression on map output
 * - Manual JSON parsing (GameWritable)
 *
 * OUTPUT:
 * - <output>/nodes/part-r-* : archetype;count;wins
 * - <output>/edges/part-r-* : source;target;count;wins
 *
 * @see NodesMapper
 * @see EdgesMapper
 * @see ArchetypeUtils
 */
public class NodesEdgesGenerator extends Configured implements Tool {

  @Override
  public int run(String[] args) throws Exception {
    if (args.length < 3 || args.length > 4) {
      printUsage();
      return 1;
    }

    Configuration conf = getConf();
    int k = Integer.parseInt(args[2]);
    int numReducers = args.length == 4 ? Integer.parseInt(args[3]) : 10;

    if (k < 1 || k > 8) {
      System.err.println("Error: k must be 1-8");
      return 1;
    }

    configureJob(conf, k);
    printBanner(k, numReducers);

    Path input = new Path(args[0]);
    Path outBase = new Path(args[1]);
    Path nodesOut = new Path(outBase, "nodes");
    Path edgesOut = new Path(outBase, "edges");
    FileSystem fs = FileSystem.get(conf);

    long t0 = System.currentTimeMillis();

    // ===== JOB 1: NODES =====
    Job nodesJob = runNodesJob(conf, k, numReducers, input, nodesOut, fs);
    if (nodesJob == null) {
      return 1;
    }

    // ===== JOB 2: EDGES =====
    Job edgesJob = runEdgesJob(conf, k, numReducers, input, edgesOut, fs);
    if (edgesJob == null) {
      return 1;
    }

    long t1 = System.currentTimeMillis();
    
    // Count final output records and games processed
    long numNodes = nodesJob.getCounters()
        .findCounter("org.apache.hadoop.mapreduce.TaskCounter", "REDUCE_OUTPUT_RECORDS")
        .getValue();
    long numEdges = edgesJob.getCounters()
        .findCounter("org.apache.hadoop.mapreduce.TaskCounter", "REDUCE_OUTPUT_RECORDS")
        .getValue();
    long numGames = nodesJob.getCounters()
        .findCounter(NodesEdgesMetrics.NodesMetrics.GAMES_PROCESSED)
        .getValue();
    
    printSummary(t1 - t0, numGames, numNodes, numEdges, outBase);
    return 0;
  }

  private void configureJob(Configuration conf, int k) {
    conf.setInt("archetype.size", k);
    conf.setBoolean("mapreduce.map.output.compress", true);
    conf.set("mapreduce.map.output.compress.codec",
             "org.apache.hadoop.io.compress.SnappyCodec");

    // ===== MAPPER MEMORY BUFFERS =====
    // Increase sort buffer to reduce spills (default 100MB)
    // With 4.6B edges for 37M games, we need large buffers
    conf.setInt("mapreduce.task.io.sort.mb", 512);
    // Spill threshold: higher = fewer spills but more memory (default 0.80)
    conf.setFloat("mapreduce.map.sort.spill.percent", 0.90f);
    // Sort factor: merge streams during spill merge (default 10)
    conf.setInt("mapreduce.task.io.sort.factor", 50);
    
    // ===== REDUCER SHUFFLE BUFFERS =====
    // Buffer for shuffle data in memory (default 0.70 of heap)
    conf.setFloat("mapreduce.reduce.shuffle.input.buffer.percent", 0.80f);
    // When to start merging shuffle data (default 0.66)
    conf.setFloat("mapreduce.reduce.shuffle.merge.percent", 0.80f);
    // Memory-to-memory merge threshold (default 0)
    conf.setFloat("mapreduce.reduce.input.buffer.percent", 0.80f);
    // Parallel copies during shuffle (default 5)
    conf.setInt("mapreduce.reduce.shuffle.parallelcopies", 20);
    
    // ===== FORCE MORE MAPPERS =====
    // Default is 128MB, we set to 16MB to get ~8x more mappers
    // This is the KEY optimization for cluster utilization
    conf.setLong("mapreduce.input.fileinputformat.split.maxsize", 16 * 1024 * 1024);
    conf.setLong("mapreduce.input.fileinputformat.split.minsize", 8 * 1024 * 1024);
  }

  private Job runNodesJob(Configuration conf, int k, int numReducers,
                          Path input, Path output, FileSystem fs) throws Exception {
    System.out.println("\n━━━ JOB 1: NODES ━━━");
    if (fs.exists(output)) {
      fs.delete(output, true);
    }

    Job job = Job.getInstance(conf, "NodesEdges_NODES (k=" + k + ")");
    job.setJarByClass(NodesEdgesGenerator.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    FileInputFormat.addInputPath(job, input);

    job.setMapperClass(NodesMapper.class);
    job.setMapOutputKeyClass(LongWritable.class);
    job.setMapOutputValueClass(LongWritable.class);

    job.setCombinerClass(NodesCombiner.class);
    job.setReducerClass(NodesReducer.class);
    job.setOutputKeyClass(NullWritable.class);
    job.setOutputValueClass(Text.class);
    job.setNumReduceTasks(numReducers);

    FileOutputFormat.setOutputPath(job, output);

    long t0 = System.currentTimeMillis();
    boolean success = job.waitForCompletion(true);
    long t1 = System.currentTimeMillis();

    if (!success) {
      System.err.println("NODES job failed!");
      return null;
    }

    System.out.printf("NODES: %.1fs, %d games, %d nodes emitted%n",
        (t1 - t0) / 1000.0,
        job.getCounters().findCounter(NodesEdgesMetrics.NodesMetrics.GAMES_PROCESSED).getValue(),
        job.getCounters().findCounter(NodesEdgesMetrics.NodesMetrics.NODES_EMITTED).getValue());

    return job;
  }

  private Job runEdgesJob(Configuration conf, int k, int numReducers,
                          Path input, Path output, FileSystem fs) throws Exception {
    System.out.println("\n━━━ JOB 2: EDGES ━━━");
    if (fs.exists(output)) {
      fs.delete(output, true);
    }

    Job job = Job.getInstance(conf, "NodesEdges_EDGES (k=" + k + ")");
    job.setJarByClass(NodesEdgesGenerator.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    FileInputFormat.addInputPath(job, input);

    job.setMapperClass(EdgesMapper.class);
    job.setMapOutputKeyClass(EdgeKey.class);
    job.setMapOutputValueClass(LongWritable.class);

    job.setCombinerClass(EdgesCombiner.class);
    job.setSortComparatorClass(EdgeKey.Comparator.class);
    job.setReducerClass(EdgesReducer.class);
    job.setOutputKeyClass(NullWritable.class);
    job.setOutputValueClass(Text.class);
    job.setNumReduceTasks(numReducers);

    FileOutputFormat.setOutputPath(job, output);

    long t0 = System.currentTimeMillis();
    boolean success = job.waitForCompletion(true);
    long t1 = System.currentTimeMillis();

    if (!success) {
      System.err.println("EDGES job failed!");
      return null;
    }

    System.out.printf("EDGES: %.1fs, %d games, %d edges emitted%n",
        (t1 - t0) / 1000.0,
        job.getCounters().findCounter(NodesEdgesMetrics.EdgesMetrics.GAMES_PROCESSED).getValue(),
        job.getCounters().findCounter(NodesEdgesMetrics.EdgesMetrics.EDGES_EMITTED).getValue());

    return job;
  }

  private void printUsage() {
    System.err.println("Usage: nodesedges <input> <output> <k> [numReducers]");
    System.err.println("  input:        SequenceFile from DataCleaner");
    System.err.println("  output:       Output directory (creates /nodes and /edges)");
    System.err.println("  k:            Archetype size (1-8, recommended: 6-7)");
    System.err.println("  numReducers:  Optional (default: 10)");
    System.err.println();
    System.err.println("OPTIMIZATIONS:");
    System.err.println("  ✓ IN-MAPPER COMBINING (local aggregation)");
    System.err.println("  ✓ COMBINER for nodes AND edges");
    System.err.println("  ✓ Binary archetype encoding (LongWritable)");
    System.err.println("  ✓ SNAPPY compression on map output");
    System.err.println("  ✓ Raw comparator for EdgeKey");
  }

  private void printBanner(int k, int numReducers) {
    int comb = binomial(8, k);
    int edges = comb * comb * 2;

    System.out.println("╔════════════════════════════════════════════════════╗");
    System.out.println("║     NodesEdgesGenerator - 2 JOBS OPTIMIZED         ║");
    System.out.println("╠════════════════════════════════════════════════════╣");
    System.out.printf("║  k=%d, C(8,%d)=%d archetypes/deck                   ║%n", k, k, comb);
    System.out.printf("║  Nodes/game: %d, Edges/game: %,d                    ║%n", comb * 2, edges);
    System.out.printf("║  Reducers: %d                                       ║%n", numReducers);

    if (k <= 5) {
      System.out.println("╠════════════════════════════════════════════════════╣");
      System.out.println("║  ⚠️  WARNING: k≤5 generates MASSIVE edge volume!   ║");
      System.out.printf("║  For 37M games: ~%,d BILLION edges to shuffle!     ║%n",
          37L * edges / 1_000_000_000);
      System.out.println("║  Consider k=6 or k=7 for better performance.       ║");
    }

    System.out.println("║  ✓ IN-MAPPER COMBINING (local aggregation)         ║");
    System.out.println("╚════════════════════════════════════════════════════╝");
  }

  private void printSummary(long totalMs, long numGames, long numNodes, long numEdges, Path outBase) {
    System.out.println("\n╔════════════════════════════════════════════════════╗");
    System.out.println("║              GENERATION COMPLETE                   ║");
    System.out.println("╠════════════════════════════════════════════════════╣");
    System.out.printf("║  TOTAL TIME:       %.1f seconds                    ║%n", totalMs / 1000.0);
    System.out.printf("║  GAMES PROCESSED:  %,d                             ║%n", numGames);
    System.out.printf("║  NODES CREATED:    %,d                             ║%n", numNodes);
    System.out.printf("║  EDGES CREATED:    %,d                             ║%n", numEdges);
    System.out.println("╠════════════════════════════════════════════════════╣");
    System.out.println("║  Output: " + outBase + "/nodes, " + outBase + "/edges");
    System.out.println("╚════════════════════════════════════════════════════╝");
  }

  public static void main(String[] args) throws Exception {
    System.exit(ToolRunner.run(new NodesEdgesGenerator(), args));
  }
}
