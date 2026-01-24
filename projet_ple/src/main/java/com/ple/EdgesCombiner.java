package com.ple;

import static com.ple.ArchetypeUtils.*;

import java.io.IOException;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Reducer;

/**
 * Combiner for edges - aggregates by (source,target) locally.
 * Critical for reducing shuffle volume in high-cardinality edge generation.
 */
public class EdgesCombiner extends Reducer<EdgeKey, LongWritable, EdgeKey, LongWritable> {

  private final LongWritable out = new LongWritable();
  private long totalTimeNs = 0;

  @Override
  protected void reduce(EdgeKey key, Iterable<LongWritable> vals, Context ctx)
      throws IOException, InterruptedException {
    long startTime = System.nanoTime();
    long count = 0, wins = 0;
    for (LongWritable v : vals) {
      long p = v.get();
      count += unpackCount(p);
      wins += unpackWins(p);
      // Count input bytes (EdgeKey = 16 bytes + value = 8 bytes = 24 bytes)
      ctx.getCounter(NodesEdgesMetrics.EdgesMetrics.COMBINER_INPUT_BYTES)
          .increment(24);
    }
    out.set(pack(count, wins));
    ctx.write(key, out);
    // Count output bytes (EdgeKey = 16 bytes + value = 8 bytes = 24 bytes)
    ctx.getCounter(NodesEdgesMetrics.EdgesMetrics.COMBINER_OUTPUT_BYTES)
        .increment(24);
    // Accumulate execution time
    totalTimeNs += (System.nanoTime() - startTime);
  }

  @Override
  protected void cleanup(Context ctx) throws IOException, InterruptedException {
    // Convert accumulated time to milliseconds
    ctx.getCounter(NodesEdgesMetrics.EdgesMetrics.COMBINER_TIME_MS)
        .increment(totalTimeNs / 1_000_000);
  }
}
