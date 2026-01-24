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
    if (args.length < 3 || args.length > 5) {
      printUsage();
      return 1;
    }

    Configuration conf = getConf();
    int k = Integer.parseInt(args[2]);
    
    // Parse reducer counts:
    // 3 args: default NODES=1, EDGES=10
    // 4 args: NODES=1, EDGES=args[3]
    // 5 args: NODES=args[3], EDGES=args[4]
    int numReducersNodes = 1;  // Default: 1 reducer for NODES (optimal for small output)
    int numReducersEdges = 10; // Default: 10 reducers for EDGES
    
    if (args.length == 4) {
      numReducersEdges = Integer.parseInt(args[3]);
    } else if (args.length == 5) {
      numReducersNodes = Integer.parseInt(args[3]);
      numReducersEdges = Integer.parseInt(args[4]);
    }

    if (k < 1 || k > 8) {
      System.err.println("Error: k must be 1-8");
      return 1;
    }

    configureJob(conf, k);
    printBanner(k, numReducersNodes, numReducersEdges);

    Path input = new Path(args[0]);
    Path outBase = new Path(args[1]);
    Path nodesOut = new Path(outBase, "nodes");
    Path edgesOut = new Path(outBase, "edges");
    FileSystem fs = FileSystem.get(conf);

    long t0 = System.currentTimeMillis();

    // ===== JOB 1: NODES =====
    long tNodesStart = System.currentTimeMillis();
    Job nodesJob = runNodesJob(conf, k, numReducersNodes, input, nodesOut, fs);
    long tNodesEnd = System.currentTimeMillis();
    if (nodesJob == null) {
      return 1;
    }

    // ===== JOB 2: EDGES =====
    long tEdgesStart = System.currentTimeMillis();
    Job edgesJob = runEdgesJob(conf, k, numReducersEdges, input, edgesOut, fs);
    long tEdgesEnd = System.currentTimeMillis();
    if (edgesJob == null) {
      return 1;
    }

    long t1 = System.currentTimeMillis();

    // Pass measured wall times to the report
    printFinalReport(k, t1 - t0, nodesJob, edgesJob, outBase, 
                     tNodesEnd - tNodesStart, tEdgesEnd - tEdgesStart);
    
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
    System.err.println("Usage: nodesedges <input> <output> <k> [numReducersEdges] [numReducersNodes numReducersEdges]");
    System.err.println("  input:              SequenceFile from DataCleaner");
    System.err.println("  output:             Output directory (creates /nodes and /edges)");
    System.err.println("  k:                  Archetype size (1-8, recommended: 6-7)");
    System.err.println("  numReducersEdges:   Optional, reducers for EDGES job (default: 10, NODES: 1)");
    System.err.println("  numReducersNodes:   Optional, reducers for NODES job (requires numReducersEdges)");
    System.err.println();
    System.err.println("Examples:");
    System.err.println("  nodesedges input.seq output 7           # NODES=1, EDGES=10");
    System.err.println("  nodesedges input.seq output 7 150       # NODES=1, EDGES=150");
    System.err.println("  nodesedges input.seq output 7 2 150     # NODES=2, EDGES=150");
    System.err.println();
    System.err.println("OPTIMIZATIONS:");
    System.err.println("  ✓ IN-MAPPER COMBINING (local aggregation)");
    System.err.println("  ✓ COMBINER for nodes AND edges");
    System.err.println("  ✓ Binary archetype encoding (LongWritable)");
    System.err.println("  ✓ SNAPPY compression on map output");
    System.err.println("  ✓ Raw comparator for EdgeKey");
  }

  private void printBanner(int k, int numReducersNodes, int numReducersEdges) {
    int comb = binomial(8, k);
    int edges = comb * comb * 2;

    System.out.println("╔════════════════════════════════════════════════════╗");
    System.out.println("║     NodesEdgesGenerator - 2 JOBS OPTIMIZED         ║");
    System.out.println("╠════════════════════════════════════════════════════╣");
    System.out.printf("║  k=%d, C(8,%d)=%d archetypes/deck                   ║%n", k, k, comb);
    System.out.printf("║  Nodes/game: %d, Edges/game: %,d                    ║%n", comb * 2, edges);
    System.out.printf("║  Reducers: NODES=%d, EDGES=%d                       ║%n", numReducersNodes, numReducersEdges);

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

  private void printFinalReport(int k, long totalMs, Job nodesJob, Job edgesJob, Path outBase, long nWallTime, long eWallTime) throws Exception {
    // --- DATA COLLECTION ---
    // Counters Nodes
    long nGames = getCounter(nodesJob.getCounters(), NodesEdgesMetrics.NodesMetrics.class, "GAMES_PROCESSED");
    long nNodes = getCounter(nodesJob.getCounters(), NodesEdgesMetrics.NodesMetrics.class, "NODES_EMITTED");
    long nMapIn = getCounter(nodesJob.getCounters(), NodesEdgesMetrics.NodesMetrics.class, "MAPPER_INPUT_BYTES");
    long nRedOut = getCounter(nodesJob.getCounters(), NodesEdgesMetrics.NodesMetrics.class, "REDUCER_OUTPUT_BYTES");
    long nMapTime = getCounter(nodesJob.getCounters(), NodesEdgesMetrics.NodesMetrics.class, "MAPPER_TIME_MS");
    long nCombTime = getCounter(nodesJob.getCounters(), NodesEdgesMetrics.NodesMetrics.class, "COMBINER_TIME_MS");
    long nRedTime = getCounter(nodesJob.getCounters(), NodesEdgesMetrics.NodesMetrics.class, "REDUCER_TIME_MS");
    long nCombIn = getCounter(nodesJob.getCounters(), NodesEdgesMetrics.NodesMetrics.class, "COMBINER_INPUT_BYTES");
    long nCombOut = getCounter(nodesJob.getCounters(), NodesEdgesMetrics.NodesMetrics.class, "COMBINER_OUTPUT_BYTES");
    int nReducers = nodesJob.getConfiguration().getInt("mapreduce.job.reduces", 1);

    // Counters Edges
    long eEdges = getCounter(edgesJob.getCounters(), NodesEdgesMetrics.EdgesMetrics.class, "EDGES_EMITTED");
    long eRedOut = getCounter(edgesJob.getCounters(), NodesEdgesMetrics.EdgesMetrics.class, "REDUCER_OUTPUT_BYTES");
    long eMapTime = getCounter(edgesJob.getCounters(), NodesEdgesMetrics.EdgesMetrics.class, "MAPPER_TIME_MS");
    long eCombTime = getCounter(edgesJob.getCounters(), NodesEdgesMetrics.EdgesMetrics.class, "COMBINER_TIME_MS");
    long eRedTime = getCounter(edgesJob.getCounters(), NodesEdgesMetrics.EdgesMetrics.class, "REDUCER_TIME_MS");
    long eCombIn = getCounter(edgesJob.getCounters(), NodesEdgesMetrics.EdgesMetrics.class, "COMBINER_INPUT_BYTES");
    long eCombOut = getCounter(edgesJob.getCounters(), NodesEdgesMetrics.EdgesMetrics.class, "COMBINER_OUTPUT_BYTES");
    int eReducers = edgesJob.getConfiguration().getInt("mapreduce.job.reduces", 1);

    // Calculations
    double totalInputMB = nMapIn / 1_000_000.0;
    double totalOutputMB = (nRedOut + eRedOut) / 1_000_000.0;
    double expansion = totalInputMB > 0 ? totalOutputMB / totalInputMB : 0;
    
    double nCombReduc = nCombIn > 0 ? (1.0 - (double)nCombOut/nCombIn) * 100 : 0;
    double eCombReduc = eCombIn > 0 ? (1.0 - (double)eCombOut/eCombIn) * 100 : 0;

    double nAvgFileMB = nReducers > 0 ? (nRedOut / 1_000_000.0) / nReducers : 0;
    double eAvgFileMB = eReducers > 0 ? (eRedOut / 1_000_000.0) / eReducers : 0;

    // OPTIMIZATION SUGGESTIONS (Target: 128MB HDFS Block)
    long HDFS_BLOCK_SIZE = 128 * 1024 * 1024;
    int nOptimalReducers = (int) Math.max(1, (nRedOut + HDFS_BLOCK_SIZE - 1) / HDFS_BLOCK_SIZE);
    int eOptimalReducers = (int) Math.max(1, (eRedOut + HDFS_BLOCK_SIZE - 1) / HDFS_BLOCK_SIZE);

    // --- PRINTING ---
    System.out.println("\n╔════════════════════════════════════════════════════════════════╗");
    System.out.printf("║             NODES & EDGES GENERATION REPORT (k=%d)              ║%n", k);
    System.out.println("╠════════════════════════════════════════════════════════════════╣");
    
    // GLOBAL SECTION
    System.out.println("║  EXECUTION SUMMARY                                             ║");
    System.out.printf("║    Total Duration:   %-15s                           ║%n", String.format("%.1f s", totalMs / 1000.0));
    System.out.printf("║    Games Processed:  %-15s                           ║%n", String.format("%,d", nGames));
    System.out.printf("║    Total Input:      %-15s                           ║%n", String.format("%.2f MB", totalInputMB));
    System.out.printf("║    Total Output:     %-15s                           ║%n", String.format("%.2f MB", totalOutputMB));
    System.out.printf("║    Expansion Factor: %-15s                           ║%n", String.format("%.2fx", expansion));

    System.out.println("╠════════════════════════════════════════════════════════════════╣");
    
    // NODES SECTION
    System.out.println("║  JOB 1: NODES                                                  ║");
    System.out.printf("║    Job Duration:     %-15s                           ║%n", String.format("%.1f s", nWallTime / 1000.0));
    System.out.printf("║    Generated:        %-15s                           ║%n", String.format("%,d nodes", nNodes));
    System.out.printf("║    Mapper CPU:       %-15s                           ║%n", String.format("%.2f s", nMapTime/1000.0));
    System.out.printf("║    Combiner:         %-15s                           ║%n", String.format("%.1f%% red. (%.2fs)", nCombReduc, nCombTime/1000.0));
    System.out.printf("║    Reducer CPU:      %-15s                           ║%n", String.format("%.2f s", nRedTime/1000.0));
    System.out.printf("║    Output Files:     %-15s                           ║%n", String.format("%d (avg %.1f MB)", nReducers, nAvgFileMB));
    System.out.printf("║    Suggestion:       %-15s                           ║%n", String.format("Use ~%d reducers", nOptimalReducers));

    System.out.println("╠════════════════════════════════════════════════════════════════╣");
    
    // EDGES SECTION
    System.out.println("║  JOB 2: EDGES                                                  ║");
    System.out.printf("║    Job Duration:     %-15s                           ║%n", String.format("%.1f s", eWallTime / 1000.0));
    System.out.printf("║    Generated:        %-15s                           ║%n", String.format("%,d edges", eEdges));
    System.out.printf("║    Mapper CPU:       %-15s                           ║%n", String.format("%.2f s", eMapTime/1000.0));
    System.out.printf("║    Combiner:         %-15s                           ║%n", String.format("%.1f%% red. (%.2fs)", eCombReduc, eCombTime/1000.0));
    System.out.printf("║    Reducer CPU:      %-15s                           ║%n", String.format("%.2f s", eRedTime/1000.0));
    System.out.printf("║    Output Files:     %-15s                           ║%n", String.format("%d (avg %.1f MB)", eReducers, eAvgFileMB));
    System.out.printf("║    Suggestion:       %-15s                           ║%n", String.format("Use ~%d reducers", eOptimalReducers));
    
    System.out.println("╚════════════════════════════════════════════════════════════════╝");
    System.out.println("Output locations:");
    System.out.println("Nodes: " + new Path(outBase, "nodes"));
    System.out.println("Edges: " + new Path(outBase, "edges"));
  }

  private long getCounter(org.apache.hadoop.mapreduce.Counters counters, Class<?> metricsEnum, String counterName) {
    try {
      Object enumValue = null;
      for (Object e : metricsEnum.getEnumConstants()) {
        if (e.toString().equals(counterName)) {
          enumValue = e;
          break;
        }
      }
      if (enumValue == null) return 0;
      return counters.findCounter((Enum<?>) enumValue).getValue();
    } catch (Exception e) {
      return 0;
    }
  }

  public static void main(String[] args) throws Exception {
    System.exit(ToolRunner.run(new NodesEdgesGenerator(), args));
  }
}
