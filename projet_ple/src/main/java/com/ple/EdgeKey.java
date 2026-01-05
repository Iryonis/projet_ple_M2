package com.ple;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;

/**
 * Composite key for edges: (source, target).
 * Enables proper reduction by (source,target) pair.
 * 
 * Binary format: 16 bytes (2 longs).
 */
public class EdgeKey implements WritableComparable<EdgeKey> {

  private long source;
  private long target;

  public EdgeKey() {}

  public void set(long source, long target) {
    this.source = source;
    this.target = target;
  }

  public long getSource() {
    return source;
  }

  public long getTarget() {
    return target;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeLong(source);
    out.writeLong(target);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    source = in.readLong();
    target = in.readLong();
  }

  @Override
  public int compareTo(EdgeKey o) {
    int cmp = Long.compare(source, o.source);
    return cmp != 0 ? cmp : Long.compare(target, o.target);
  }

  @Override
  public int hashCode() {
    return (int) ((source * 31) ^ target);
  }

  @Override
  public boolean equals(Object obj) {
    if (obj instanceof EdgeKey) {
      EdgeKey o = (EdgeKey) obj;
      return source == o.source && target == o.target;
    }
    return false;
  }

  /**
   * Raw comparator for EdgeKey - avoids deserialization during sort.
   * Major performance optimization for shuffle phase.
   */
  public static class Comparator extends WritableComparator {

    public Comparator() {
      super(EdgeKey.class, true);
    }

    @Override
    public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
      long src1 = readLong(b1, s1);
      long src2 = readLong(b2, s2);
      int cmp = Long.compare(src1, src2);
      if (cmp != 0) return cmp;
      long tgt1 = readLong(b1, s1 + 8);
      long tgt2 = readLong(b2, s2 + 8);
      return Long.compare(tgt1, tgt2);
    }
  }

  // Register the raw comparator
  static {
    WritableComparator.define(EdgeKey.class, new Comparator());
  }
}
