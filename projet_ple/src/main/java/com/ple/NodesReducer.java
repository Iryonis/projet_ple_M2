package com.ple;

import static com.ple.ArchetypeUtils.*;

import java.io.IOException;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

/**
 * Reducer for nodes - aggregates all values for each archetype.
 * 
 * Output format: archetype;count;wins
 */
public class NodesReducer extends Reducer<LongWritable, LongWritable, NullWritable, Text> {

  private final Text out = new Text();
  private int k;

  @Override
  protected void setup(Context ctx) {
    k = ctx.getConfiguration().getInt("archetype.size", 7);
  }

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
      ctx.getCounter(NodesEdgesMetrics.NodesMetrics.REDUCER_INPUT_BYTES)
          .increment(16);
    }
    out.set(toHex(key.get(), k) + ";" + count + ";" + wins);
    ctx.write(NullWritable.get(), out);
    // Count output bytes (output text length)
    ctx.getCounter(NodesEdgesMetrics.NodesMetrics.REDUCER_OUTPUT_BYTES)
        .increment(out.getLength());
    // Measure execution time
    ctx.getCounter(NodesEdgesMetrics.NodesMetrics.REDUCER_TIME_MS)
        .increment((System.nanoTime() - startTime) / 1_000_000);
  }
}
