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
    long count = 0, wins = 0;
    for (LongWritable v : vals) {
      long p = v.get();
      count += unpackCount(p);
      wins += unpackWins(p);
    }
    out.set(toHex(key.get(), k) + ";" + count + ";" + wins);
    ctx.write(NullWritable.get(), out);
  }
}
