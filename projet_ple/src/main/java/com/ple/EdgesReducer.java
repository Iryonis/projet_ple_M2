package com.ple;

import static com.ple.ArchetypeUtils.*;

import java.io.IOException;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

/**
 * Reducer for edges - aggregates all values for each (source,target) pair.
 * 
 * Output format: source;target;count;wins
 */
public class EdgesReducer extends Reducer<EdgeKey, LongWritable, NullWritable, Text> {

  private final Text out = new Text();
  private int k;

  @Override
  protected void setup(Context ctx) {
    k = ctx.getConfiguration().getInt("archetype.size", 7);
  }

  @Override
  protected void reduce(EdgeKey key, Iterable<LongWritable> vals, Context ctx)
      throws IOException, InterruptedException {
    long count = 0, wins = 0;
    for (LongWritable v : vals) {
      long p = v.get();
      count += unpackCount(p);
      wins += unpackWins(p);
    }
    out.set(
      toHex(key.getSource(), k) + ";" +
      toHex(key.getTarget(), k) + ";" +
      count + ";" + wins
    );
    ctx.write(NullWritable.get(), out);
  }
}
