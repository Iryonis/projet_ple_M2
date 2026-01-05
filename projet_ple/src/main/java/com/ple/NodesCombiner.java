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

  @Override
  protected void reduce(LongWritable key, Iterable<LongWritable> vals, Context ctx)
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
