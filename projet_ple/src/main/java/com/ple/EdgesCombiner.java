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

  @Override
  protected void reduce(EdgeKey key, Iterable<LongWritable> vals, Context ctx)
      throws IOException, InterruptedException {
    long count = 0, wins = 0;
    for (LongWritable v : vals) {
      long p = v.get();
      count += unpackCount(p);
      wins += unpackWins(p);
    }
    out.set(pack(count, wins));
    ctx.write(key, out);
  }
}
