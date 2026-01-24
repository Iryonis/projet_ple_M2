package com.ple;

import static com.ple.ArchetypeUtils.*;

import java.io.IOException;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

/**
 * Combiner for nodes - aggregates locally before shuffle.
 * Massively reduces network traffic between mappers and reducers.
 */
public class NodesCombiner extends Reducer<LongWritable, LongWritable, LongWritable, LongWritable> {

  private final LongWritable out = new LongWritable();
  private long totalTimeNs = 0;

  @Override
  protected void reduce(LongWritable key, Iterable<LongWritable> vals, Context ctx)
      throws IOException, InterruptedException {
    long startTime = System.nanoTime();
    long count = 0, wins = 0;
    for (LongWritable v : vals) {
      long p = v.get();
      count += unpackCount(p);
      wins += unpackWins(p);
      // Count input bytes (key + value = 8 + 8 = 16 bytes)
      ctx.getCounter(NodesEdgesMetrics.NodesMetrics.COMBINER_INPUT_BYTES)
          .increment(16);
    }
    out.set(pack(count, wins));
    ctx.write(key, out);
    // Count output bytes (key + value = 8 + 8 = 16 bytes)
    ctx.getCounter(NodesEdgesMetrics.NodesMetrics.COMBINER_OUTPUT_BYTES)
        .increment(16);
    // Accumulate execution time
    totalTimeNs += (System.nanoTime() - startTime);
  }

  @Override
  protected void cleanup(Context ctx) throws IOException, InterruptedException {
    // Convert accumulated time to milliseconds
    ctx.getCounter(NodesEdgesMetrics.NodesMetrics.COMBINER_TIME_MS)
        .increment(totalTimeNs / 1_000_000);
  }
}
