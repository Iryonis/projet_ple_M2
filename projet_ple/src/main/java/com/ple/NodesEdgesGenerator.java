package com.ple;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

/**
 * ULTRA-OPTIMIZED MapReduce for nodes and edges generation.
 *
 * ARCHITECTURE: 2 SEPARATE JOBS
 * - Job 1 (Nodes): Key=archetype, aggregates (count, wins)
 * - Job 2 (Edges): Key=(source,target), aggregates (count, wins)
 *
 * CRITICAL OPTIMIZATIONS:
 * 1. NO Gson: Manual JSON parsing via GameWritable
 * 2. ALL C(8,k) combinations generated (subject requirement)
 * 3. STREAMING generation: No array allocation
 * 4. FIXED countingSort bug: Separate buffers
 * 5. LongWritable keys: Binary encoding
 * 6. SEPARATE JOBS: Proper reduction for nodes AND edges
 * 7. COMBINER: For nodes AND edges (massive network reduction)
 * 8. COMPRESSION: Snappy on map output
 * 9. ANTI-EXPLOSION: Warning for k<=5
 * 10. Non-recursive loops: Unrolled for k=1..8
 * 11. EdgeKey: Composite key for proper edge reduction
 *
 * OUTPUT:
 * - <output>/nodes/part-r-* : archetype;count;wins
 * - <output>/edges/part-r-* : source;target;count;wins
 */
public class NodesEdgesGenerator extends Configured implements Tool {

  // ==================== COUNTERS ====================

  public static enum NodesMetrics {
    GAMES_PROCESSED,
    GAMES_SKIPPED,
    NODES_EMITTED,
  }

  public static enum EdgesMetrics {
    GAMES_PROCESSED,
    GAMES_SKIPPED,
    EDGES_EMITTED,
  }

  // ==================== EDGE COMPOSITE KEY ====================

  /**
   * Composite key for edges: (source, target).
   * Enables proper reduction by (source,target) pair.
   */
  public static class EdgeKey implements WritableComparable<EdgeKey> {

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
  }

  /** Raw comparator for EdgeKey - avoids deserialization */
  public static class EdgeKeyComparator extends WritableComparator {

    public EdgeKeyComparator() {
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

  // ==================== IN-MAPPER COMBINING CONSTANTS ====================

  /** Max entries in local aggregation map before flush (tune based on heap) */
  private static final int MAX_MAP_SIZE = 500_000;

  /** Flush threshold as percentage of max */
  private static final float FLUSH_THRESHOLD = 0.9f;

  // ==================== JOB 1: NODES ====================

  public static class NodesMapper
    extends Mapper<NullWritable, Text, LongWritable, LongWritable> {

    private final GameWritable game = new GameWritable();
    private final LongWritable outKey = new LongWritable();
    private final LongWritable outVal = new LongWritable();
    private int k;
    private final byte[] buf0 = new byte[8], buf1 = new byte[8];
    private final int[] cnt = new int[256];
    private Counter processed, skipped, emitted;

    // IN-MAPPER COMBINING: Local aggregation map
    // Key: archetype (long), Value: packed (count << 32 | wins)
    private HashMap<Long, long[]> localMap;
    private long localEmitCount;
    private Context context;

    @Override
    protected void setup(Context ctx) {
      k = ctx.getConfiguration().getInt("archetype.size", 7);
      processed = ctx.getCounter(NodesMetrics.GAMES_PROCESSED);
      skipped = ctx.getCounter(NodesMetrics.GAMES_SKIPPED);
      emitted = ctx.getCounter(NodesMetrics.NODES_EMITTED);
      // Initialize local map with expected capacity
      localMap = new HashMap<>(MAX_MAP_SIZE / 2);
      localEmitCount = 0;
      context = ctx;
    }

    @Override
    protected void map(NullWritable key, Text val, Context ctx)
      throws IOException, InterruptedException {
      if (!game.parseFromJson(val.toString())) {
        skipped.increment(1);
        return;
      }
      processed.increment(1);
      byte[] s0 = sort(game.deck0, buf0), s1 = sort(game.deck1, buf1);
      emitNodes(s0, game.winner == 0 ? 1 : 0);
      emitNodes(s1, game.winner == 1 ? 1 : 0);

      // Flush if map is getting too large
      if (localMap.size() >= MAX_MAP_SIZE * FLUSH_THRESHOLD) {
        flush(ctx);
      }
    }

    @Override
    protected void cleanup(Context ctx)
      throws IOException, InterruptedException {
      flush(ctx);
      emitted.increment(localEmitCount);
    }

    /** Flush local map to context */
    private void flush(Context ctx) throws IOException, InterruptedException {
      for (Map.Entry<Long, long[]> e : localMap.entrySet()) {
        outKey.set(e.getKey());
        long[] v = e.getValue();
        outVal.set((v[0] << 32) | v[1]);
        ctx.write(outKey, outVal);
        localEmitCount++;
      }
      localMap.clear();
    }

    private byte[] sort(byte[] d, byte[] o) {
      Arrays.fill(cnt, 0);
      for (int i = 0; i < 8; i++) cnt[d[i] & 0xFF]++;
      int idx = 0;
      for (int v = 0; v < 256; v++) while (cnt[v]-- > 0) o[idx++] = (byte) v;
      return o;
    }

    /** Aggregate locally instead of emitting immediately */
    private void emit(long arch, int w) {
      long[] agg = localMap.get(arch);
      if (agg == null) {
        agg = new long[2]; // [count, wins]
        localMap.put(arch, agg);
      }
      agg[0]++;
      agg[1] += w;
    }

    private void emitNodes(byte[] d, int w) {
      switch (k) {
        case 1:
          for (int i = 0; i < 8; i++) emit(d[i] & 0xFFL, w);
          break;
        case 2:
          for (int i = 0; i < 7; i++) for (int j = i + 1; j < 8; j++) emit(
            ((long) (d[i] & 0xFF) << 8) | (d[j] & 0xFF),
            w
          );
          break;
        case 3:
          for (int i = 0; i < 6; i++) for (int j = i + 1; j < 7; j++) for (
            int k = j + 1;
            k < 8;
            k++
          ) emit(
            ((long) (d[i] & 0xFF) << 16) |
            ((long) (d[j] & 0xFF) << 8) |
            (d[k] & 0xFF),
            w
          );
          break;
        case 4:
          for (int i = 0; i < 5; i++) for (int j = i + 1; j < 6; j++) for (
            int k = j + 1;
            k < 7;
            k++
          ) for (int l = k + 1; l < 8; l++) emit(
            ((long) (d[i] & 0xFF) << 24) |
            ((long) (d[j] & 0xFF) << 16) |
            ((long) (d[k] & 0xFF) << 8) |
            (d[l] & 0xFF),
            w
          );
          break;
        case 5:
          for (int i = 0; i < 4; i++) for (int j = i + 1; j < 5; j++) for (
            int k = j + 1;
            k < 6;
            k++
          ) for (int l = k + 1; l < 7; l++) for (
            int m = l + 1;
            m < 8;
            m++
          ) emit(
            ((long) (d[i] & 0xFF) << 32) |
            ((long) (d[j] & 0xFF) << 24) |
            ((long) (d[k] & 0xFF) << 16) |
            ((long) (d[l] & 0xFF) << 8) |
            (d[m] & 0xFF),
            w
          );
          break;
        case 6:
          for (int i = 0; i < 3; i++) for (int j = i + 1; j < 4; j++) for (
            int k = j + 1;
            k < 5;
            k++
          ) for (int l = k + 1; l < 6; l++) for (
            int m = l + 1;
            m < 7;
            m++
          ) for (int n = m + 1; n < 8; n++) emit(
            ((long) (d[i] & 0xFF) << 40) |
            ((long) (d[j] & 0xFF) << 32) |
            ((long) (d[k] & 0xFF) << 24) |
            ((long) (d[l] & 0xFF) << 16) |
            ((long) (d[m] & 0xFF) << 8) |
            (d[n] & 0xFF),
            w
          );
          break;
        case 7:
          for (int i = 0; i < 2; i++) for (int j = i + 1; j < 3; j++) for (
            int k = j + 1;
            k < 4;
            k++
          ) for (int l = k + 1; l < 5; l++) for (
            int m = l + 1;
            m < 6;
            m++
          ) for (int n = m + 1; n < 7; n++) for (
            int o = n + 1;
            o < 8;
            o++
          ) emit(
            ((long) (d[i] & 0xFF) << 48) |
            ((long) (d[j] & 0xFF) << 40) |
            ((long) (d[k] & 0xFF) << 32) |
            ((long) (d[l] & 0xFF) << 24) |
            ((long) (d[m] & 0xFF) << 16) |
            ((long) (d[n] & 0xFF) << 8) |
            (d[o] & 0xFF),
            w
          );
          break;
        case 8:
          emit(
            ((long) (d[0] & 0xFF) << 56) |
            ((long) (d[1] & 0xFF) << 48) |
            ((long) (d[2] & 0xFF) << 40) |
            ((long) (d[3] & 0xFF) << 32) |
            ((long) (d[4] & 0xFF) << 24) |
            ((long) (d[5] & 0xFF) << 16) |
            ((long) (d[6] & 0xFF) << 8) |
            (d[7] & 0xFF),
            w
          );
          break;
      }
    }
  }

  /** Combiner for nodes - aggregates locally before shuffle */
  public static class NodesCombiner
    extends Reducer<LongWritable, LongWritable, LongWritable, LongWritable> {

    private final LongWritable out = new LongWritable();

    @Override
    protected void reduce(
      LongWritable key,
      Iterable<LongWritable> vals,
      Context ctx
    ) throws IOException, InterruptedException {
      long count = 0, wins = 0;
      for (LongWritable v : vals) {
        long p = v.get();
        count += (p >> 32) & 0xFFFFFFFFL;
        wins += p & 0xFFFFFFFFL;
      }
      out.set((count << 32) | wins);
      ctx.write(key, out);
    }
  }

  public static class NodesReducer
    extends Reducer<LongWritable, LongWritable, NullWritable, Text> {

    private final Text out = new Text();
    private int k;

    @Override
    protected void setup(Context ctx) {
      k = ctx.getConfiguration().getInt("archetype.size", 7);
    }

    @Override
    protected void reduce(
      LongWritable key,
      Iterable<LongWritable> vals,
      Context ctx
    ) throws IOException, InterruptedException {
      long count = 0, wins = 0;
      for (LongWritable v : vals) {
        long p = v.get();
        count += (p >> 32) & 0xFFFFFFFFL;
        wins += p & 0xFFFFFFFFL;
      }
      out.set(hex(key.get(), k) + ";" + count + ";" + wins);
      ctx.write(NullWritable.get(), out);
    }

    private String hex(long v, int n) {
      StringBuilder sb = new StringBuilder(n * 2);
      for (int i = n - 1; i >= 0; i--) {
        int b = (int) ((v >> (i * 8)) & 0xFF);
        sb.append(Character.forDigit((b >> 4) & 0xF, 16));
        sb.append(Character.forDigit(b & 0xF, 16));
      }
      return sb.toString();
    }
  }

  // ==================== JOB 2: EDGES ====================

  public static class EdgesMapper
    extends Mapper<NullWritable, Text, EdgeKey, LongWritable> {

    private final GameWritable game = new GameWritable();
    private final EdgeKey outKey = new EdgeKey();
    private final LongWritable outVal = new LongWritable();
    private int k;
    private final byte[] buf0 = new byte[8], buf1 = new byte[8];
    private final int[] cnt = new int[256];
    private Counter processed, skipped, emitted;

    // IN-MAPPER COMBINING: Local aggregation map for edges
    // Key: packed (source << 64 bits conceptually, but we use 2 longs)
    // Using a custom key class would be expensive, so we use a long-pair encoding
    private HashMap<Long, HashMap<Long, long[]>> localEdgeMap;
    private long localEmitCount;
    private int localMapEntries;

    @Override
    protected void setup(Context ctx) {
      k = ctx.getConfiguration().getInt("archetype.size", 7);
      processed = ctx.getCounter(EdgesMetrics.GAMES_PROCESSED);
      skipped = ctx.getCounter(EdgesMetrics.GAMES_SKIPPED);
      emitted = ctx.getCounter(EdgesMetrics.EDGES_EMITTED);
      // Nested map: source -> (target -> [count, wins])
      localEdgeMap = new HashMap<>(16384);
      localEmitCount = 0;
      localMapEntries = 0;
    }

    @Override
    protected void map(NullWritable key, Text val, Context ctx)
      throws IOException, InterruptedException {
      if (!game.parseFromJson(val.toString())) {
        skipped.increment(1);
        return;
      }
      processed.increment(1);
      byte[] s0 = sort(game.deck0, buf0), s1 = sort(game.deck1, buf1);
      emitEdges(s0, s1, game.winner);

      // Flush if map is getting too large
      if (localMapEntries >= MAX_MAP_SIZE * FLUSH_THRESHOLD) {
        flush(ctx);
      }
    }

    @Override
    protected void cleanup(Context ctx)
      throws IOException, InterruptedException {
      flush(ctx);
      emitted.increment(localEmitCount);
    }

    /** Flush local edge map to context */
    private void flush(Context ctx) throws IOException, InterruptedException {
      for (Map.Entry<Long, HashMap<Long, long[]>> srcEntry : localEdgeMap.entrySet()) {
        long src = srcEntry.getKey();
        for (Map.Entry<Long, long[]> tgtEntry : srcEntry
          .getValue()
          .entrySet()) {
          long tgt = tgtEntry.getKey();
          long[] v = tgtEntry.getValue();
          outKey.set(src, tgt);
          outVal.set((v[0] << 32) | v[1]);
          ctx.write(outKey, outVal);
          localEmitCount++;
        }
      }
      localEdgeMap.clear();
      localMapEntries = 0;
    }

    private byte[] sort(byte[] d, byte[] o) {
      Arrays.fill(cnt, 0);
      for (int i = 0; i < 8; i++) cnt[d[i] & 0xFF]++;
      int idx = 0;
      for (int v = 0; v < 256; v++) while (cnt[v]-- > 0) o[idx++] = (byte) v;
      return o;
    }

    /** Aggregate locally instead of emitting immediately */
    private void emit(long src, long tgt, int w) {
      HashMap<Long, long[]> tgtMap = localEdgeMap.get(src);
      if (tgtMap == null) {
        tgtMap = new HashMap<>(256);
        localEdgeMap.put(src, tgtMap);
      }
      long[] agg = tgtMap.get(tgt);
      if (agg == null) {
        agg = new long[2]; // [count, wins]
        tgtMap.put(tgt, agg);
        localMapEntries++;
      }
      agg[0]++;
      agg[1] += w;
    }

    // Archetype computation helpers
    private long a1(byte[] d, int i) {
      return d[i] & 0xFFL;
    }

    private long a2(byte[] d, int i, int j) {
      return ((long) (d[i] & 0xFF) << 8) | (d[j] & 0xFF);
    }

    private long a3(byte[] d, int i, int j, int k) {
      return (
        ((long) (d[i] & 0xFF) << 16) |
        ((long) (d[j] & 0xFF) << 8) |
        (d[k] & 0xFF)
      );
    }

    private long a4(byte[] d, int i, int j, int k, int l) {
      return (
        ((long) (d[i] & 0xFF) << 24) |
        ((long) (d[j] & 0xFF) << 16) |
        ((long) (d[k] & 0xFF) << 8) |
        (d[l] & 0xFF)
      );
    }

    private long a5(byte[] d, int i, int j, int k, int l, int m) {
      return (
        ((long) (d[i] & 0xFF) << 32) |
        ((long) (d[j] & 0xFF) << 24) |
        ((long) (d[k] & 0xFF) << 16) |
        ((long) (d[l] & 0xFF) << 8) |
        (d[m] & 0xFF)
      );
    }

    private long a6(byte[] d, int i, int j, int k, int l, int m, int n) {
      return (
        ((long) (d[i] & 0xFF) << 40) |
        ((long) (d[j] & 0xFF) << 32) |
        ((long) (d[k] & 0xFF) << 24) |
        ((long) (d[l] & 0xFF) << 16) |
        ((long) (d[m] & 0xFF) << 8) |
        (d[n] & 0xFF)
      );
    }

    private long a7(byte[] d, int i, int j, int k, int l, int m, int n, int o) {
      return (
        ((long) (d[i] & 0xFF) << 48) |
        ((long) (d[j] & 0xFF) << 40) |
        ((long) (d[k] & 0xFF) << 32) |
        ((long) (d[l] & 0xFF) << 24) |
        ((long) (d[m] & 0xFF) << 16) |
        ((long) (d[n] & 0xFF) << 8) |
        (d[o] & 0xFF)
      );
    }

    private long a8(byte[] d) {
      return (
        ((long) (d[0] & 0xFF) << 56) |
        ((long) (d[1] & 0xFF) << 48) |
        ((long) (d[2] & 0xFF) << 40) |
        ((long) (d[3] & 0xFF) << 32) |
        ((long) (d[4] & 0xFF) << 24) |
        ((long) (d[5] & 0xFF) << 16) |
        ((long) (d[6] & 0xFF) << 8) |
        (d[7] & 0xFF)
      );
    }

    private void emitEdges(byte[] d0, byte[] d1, int w) {
      int w0 = w == 0 ? 1 : 0, w1 = w == 1 ? 1 : 0;
      switch (k) {
        case 1:
          for (int i = 0; i < 8; i++) {
            long x = a1(d0, i);
            for (int j = 0; j < 8; j++) {
              long y = a1(d1, j);
              emit(x, y, w0);
              emit(y, x, w1);
            }
          }
          break;
        case 2:
          for (int i = 0; i < 7; i++) for (int j = i + 1; j < 8; j++) {
            long x = a2(d0, i, j);
            for (int a = 0; a < 7; a++) for (int b = a + 1; b < 8; b++) {
              long y = a2(d1, a, b);
              emit(x, y, w0);
              emit(y, x, w1);
            }
          }
          break;
        case 3:
          for (int i = 0; i < 6; i++) for (int j = i + 1; j < 7; j++) for (
            int k = j + 1;
            k < 8;
            k++
          ) {
            long x = a3(d0, i, j, k);
            for (int a = 0; a < 6; a++) for (int b = a + 1; b < 7; b++) for (
              int c = b + 1;
              c < 8;
              c++
            ) {
              long y = a3(d1, a, b, c);
              emit(x, y, w0);
              emit(y, x, w1);
            }
          }
          break;
        case 4:
          for (int i = 0; i < 5; i++) for (int j = i + 1; j < 6; j++) for (
            int k = j + 1;
            k < 7;
            k++
          ) for (int l = k + 1; l < 8; l++) {
            long x = a4(d0, i, j, k, l);
            for (int a = 0; a < 5; a++) for (int b = a + 1; b < 6; b++) for (
              int c = b + 1;
              c < 7;
              c++
            ) for (int e = c + 1; e < 8; e++) {
              long y = a4(d1, a, b, c, e);
              emit(x, y, w0);
              emit(y, x, w1);
            }
          }
          break;
        case 5:
          for (int i = 0; i < 4; i++) for (int j = i + 1; j < 5; j++) for (
            int k = j + 1;
            k < 6;
            k++
          ) for (int l = k + 1; l < 7; l++) for (int m = l + 1; m < 8; m++) {
            long x = a5(d0, i, j, k, l, m);
            for (int a = 0; a < 4; a++) for (int b = a + 1; b < 5; b++) for (
              int c = b + 1;
              c < 6;
              c++
            ) for (int e = c + 1; e < 7; e++) for (int f = e + 1; f < 8; f++) {
              long y = a5(d1, a, b, c, e, f);
              emit(x, y, w0);
              emit(y, x, w1);
            }
          }
          break;
        case 6:
          for (int i = 0; i < 3; i++) for (int j = i + 1; j < 4; j++) for (
            int k = j + 1;
            k < 5;
            k++
          ) for (int l = k + 1; l < 6; l++) for (
            int m = l + 1;
            m < 7;
            m++
          ) for (int n = m + 1; n < 8; n++) {
            long x = a6(d0, i, j, k, l, m, n);
            for (int a = 0; a < 3; a++) for (int b = a + 1; b < 4; b++) for (
              int c = b + 1;
              c < 5;
              c++
            ) for (int e = c + 1; e < 6; e++) for (
              int f = e + 1;
              f < 7;
              f++
            ) for (int g = f + 1; g < 8; g++) {
              long y = a6(d1, a, b, c, e, f, g);
              emit(x, y, w0);
              emit(y, x, w1);
            }
          }
          break;
        case 7:
          for (int i = 0; i < 2; i++) for (int j = i + 1; j < 3; j++) for (
            int k = j + 1;
            k < 4;
            k++
          ) for (int l = k + 1; l < 5; l++) for (
            int m = l + 1;
            m < 6;
            m++
          ) for (int n = m + 1; n < 7; n++) for (int o = n + 1; o < 8; o++) {
            long x = a7(d0, i, j, k, l, m, n, o);
            for (int a = 0; a < 2; a++) for (int b = a + 1; b < 3; b++) for (
              int c = b + 1;
              c < 4;
              c++
            ) for (int e = c + 1; e < 5; e++) for (
              int f = e + 1;
              f < 6;
              f++
            ) for (int g = f + 1; g < 7; g++) for (int h = g + 1; h < 8; h++) {
              long y = a7(d1, a, b, c, e, f, g, h);
              emit(x, y, w0);
              emit(y, x, w1);
            }
          }
          break;
        case 8:
          {
            long x = a8(d0), y = a8(d1);
            emit(x, y, w0);
            emit(y, x, w1);
          }
          break;
      }
    }
  }

  /** Combiner for edges - aggregates by (source,target) locally */
  public static class EdgesCombiner
    extends Reducer<EdgeKey, LongWritable, EdgeKey, LongWritable> {

    private final LongWritable out = new LongWritable();

    @Override
    protected void reduce(
      EdgeKey key,
      Iterable<LongWritable> vals,
      Context ctx
    ) throws IOException, InterruptedException {
      long count = 0, wins = 0;
      for (LongWritable v : vals) {
        long p = v.get();
        count += (p >> 32) & 0xFFFFFFFFL;
        wins += p & 0xFFFFFFFFL;
      }
      out.set((count << 32) | wins);
      ctx.write(key, out);
    }
  }

  public static class EdgesReducer
    extends Reducer<EdgeKey, LongWritable, NullWritable, Text> {

    private final Text out = new Text();
    private int k;

    @Override
    protected void setup(Context ctx) {
      k = ctx.getConfiguration().getInt("archetype.size", 7);
    }

    @Override
    protected void reduce(
      EdgeKey key,
      Iterable<LongWritable> vals,
      Context ctx
    ) throws IOException, InterruptedException {
      long count = 0, wins = 0;
      for (LongWritable v : vals) {
        long p = v.get();
        count += (p >> 32) & 0xFFFFFFFFL;
        wins += p & 0xFFFFFFFFL;
      }
      out.set(
        hex(key.getSource(), k) +
        ";" +
        hex(key.getTarget(), k) +
        ";" +
        count +
        ";" +
        wins
      );
      ctx.write(NullWritable.get(), out);
    }

    private String hex(long v, int n) {
      StringBuilder sb = new StringBuilder(n * 2);
      for (int i = n - 1; i >= 0; i--) {
        int b = (int) ((v >> (i * 8)) & 0xFF);
        sb.append(Character.forDigit((b >> 4) & 0xF, 16));
        sb.append(Character.forDigit(b & 0xF, 16));
      }
      return sb.toString();
    }
  }

  // ==================== DRIVER ====================

  @Override
  public int run(String[] args) throws Exception {
    if (args.length < 3 || args.length > 4) {
      System.err.println(
        "Usage: nodesedges <input> <output> <k> [numReducers]"
      );
      System.err.println("  input:        SequenceFile from DataCleaner");
      System.err.println(
        "  output:       Output directory (creates /nodes and /edges)"
      );
      System.err.println(
        "  k:            Archetype size (1-8, recommended: 6-7)"
      );
      System.err.println("  numReducers:  Optional (default: 10)");
      System.err.println();
      System.err.println("OPTIMIZATIONS:");
      System.err.println("  ✓ 2 SEPARATE JOBS (proper reduction)");
      System.err.println("  ✓ COMBINER for nodes AND edges");
      System.err.println("  ✓ EdgeKey composite for proper edge aggregation");
      System.err.println("  ✓ SNAPPY compression on map output");
      System.err.println("  ✓ NO Gson (manual JSON parsing)");
      return 1;
    }

    Configuration conf = getConf();
    int k = Integer.parseInt(args[2]);
    int numReducers = args.length == 4 ? Integer.parseInt(args[3]) : 10;

    if (k < 1 || k > 8) {
      System.err.println("Error: k must be 1-8");
      return 1;
    }

    conf.setInt("archetype.size", k);
    conf.setBoolean("mapreduce.map.output.compress", true);
    conf.set(
      "mapreduce.map.output.compress.codec",
      "org.apache.hadoop.io.compress.SnappyCodec"
    );

    // CRITICAL: Increase map output buffer to avoid ArrayIndexOutOfBoundsException
    // Default is 100MB, we need more for high-volume edge emission
    conf.setInt("mapreduce.task.io.sort.mb", 512);
    conf.setFloat("mapreduce.map.sort.spill.percent", 0.9f);

    int comb = binomial(8, k);
    int edges = comb * comb * 2;

    System.out.println(
      "╔════════════════════════════════════════════════════╗"
    );
    System.out.println("║     NodesEdgesGenerator - 2 JOBS OPTIMIZED      ║");
    System.out.println(
      "╠════════════════════════════════════════════════════╣"
    );
    System.out.printf(
      "║  k=%d, C(8,%d)=%d archetypes/deck                  ║%n",
      k,
      k,
      comb
    );
    System.out.printf(
      "║  Nodes/game: %d, Edges/game: %,d               ║%n",
      comb * 2,
      edges
    );
    System.out.printf(
      "║  Reducers: %d                                       ║%n",
      numReducers
    );
    if (k <= 5) {
      System.out.println(
        "╠════════════════════════════════════════════════════╣"
      );
      System.out.println(
        "║  ⚠️  WARNING: k≤5 generates MASSIVE edge volume!   ║"
      );
      System.out.printf(
        "║  For 37M games: ~%,d BILLION edges to shuffle! ║%n",
        37L * edges / 1_000_000_000
      );
      System.out.println(
        "║  Consider k=6 or k=7 for better performance.       ║"
      );
    }
    System.out.println(
      "║  ✓ IN-MAPPER COMBINING (local aggregation)         ║"
    );
    System.out.println(
      "╚════════════════════════════════════════════════════╝"
    );

    Path input = new Path(args[0]);
    Path outBase = new Path(args[1]);
    Path nodesOut = new Path(outBase, "nodes");
    Path edgesOut = new Path(outBase, "edges");
    FileSystem fs = FileSystem.get(conf);

    long t0 = System.currentTimeMillis();

    // ===== JOB 1: NODES =====
    System.out.println("\n━━━ JOB 1: NODES ━━━");
    if (fs.exists(nodesOut)) fs.delete(nodesOut, true);

    Job j1 = Job.getInstance(conf, "NodesEdges_NODES (k=" + k + ")");
    j1.setJarByClass(NodesEdgesGenerator.class);
    j1.setInputFormatClass(SequenceFileInputFormat.class);
    FileInputFormat.addInputPath(j1, input);
    j1.setMapperClass(NodesMapper.class);
    j1.setMapOutputKeyClass(LongWritable.class);
    j1.setMapOutputValueClass(LongWritable.class);
    j1.setCombinerClass(NodesCombiner.class);
    j1.setReducerClass(NodesReducer.class);
    j1.setOutputKeyClass(NullWritable.class);
    j1.setOutputValueClass(Text.class);
    j1.setNumReduceTasks(numReducers);
    FileOutputFormat.setOutputPath(j1, nodesOut);

    long n0 = System.currentTimeMillis();
    if (!j1.waitForCompletion(true)) {
      System.err.println("NODES failed!");
      return 1;
    }
    long n1 = System.currentTimeMillis();
    System.out.printf(
      "NODES: %.1fs, %d games, %d nodes emitted%n",
      (n1 - n0) / 1000.0,
      j1.getCounters().findCounter(NodesMetrics.GAMES_PROCESSED).getValue(),
      j1.getCounters().findCounter(NodesMetrics.NODES_EMITTED).getValue()
    );

    // ===== JOB 2: EDGES =====
    System.out.println("\n━━━ JOB 2: EDGES ━━━");
    if (fs.exists(edgesOut)) fs.delete(edgesOut, true);

    Job j2 = Job.getInstance(conf, "NodesEdges_EDGES (k=" + k + ")");
    j2.setJarByClass(NodesEdgesGenerator.class);
    j2.setInputFormatClass(SequenceFileInputFormat.class);
    FileInputFormat.addInputPath(j2, input);
    j2.setMapperClass(EdgesMapper.class);
    j2.setMapOutputKeyClass(EdgeKey.class);
    j2.setMapOutputValueClass(LongWritable.class);
    j2.setCombinerClass(EdgesCombiner.class);
    j2.setSortComparatorClass(EdgeKeyComparator.class);
    j2.setReducerClass(EdgesReducer.class);
    j2.setOutputKeyClass(NullWritable.class);
    j2.setOutputValueClass(Text.class);
    j2.setNumReduceTasks(numReducers);
    FileOutputFormat.setOutputPath(j2, edgesOut);

    long e0 = System.currentTimeMillis();
    if (!j2.waitForCompletion(true)) {
      System.err.println("EDGES failed!");
      return 1;
    }
    long e1 = System.currentTimeMillis();
    System.out.printf(
      "EDGES: %.1fs, %d games, %d edges emitted%n",
      (e1 - e0) / 1000.0,
      j2.getCounters().findCounter(EdgesMetrics.GAMES_PROCESSED).getValue(),
      j2.getCounters().findCounter(EdgesMetrics.EDGES_EMITTED).getValue()
    );

    long t1 = System.currentTimeMillis();
    System.out.println(
      "\n╔════════════════════════════════════════════════════╗"
    );
    System.out.printf(
      "║  TOTAL TIME: %.1f seconds                          ║%n",
      (t1 - t0) / 1000.0
    );
    System.out.println(
      "║  Output: " + outBase + "/nodes, " + outBase + "/edges"
    );
    System.out.println(
      "╚════════════════════════════════════════════════════╝"
    );
    return 0;
  }

  private static int binomial(int n, int k) {
    if (k > n - k) k = n - k;
    int r = 1;
    for (int i = 0; i < k; i++) r = r * (n - i) / (i + 1);
    return r;
  }

  public static void main(String[] args) throws Exception {
    System.exit(ToolRunner.run(new NodesEdgesGenerator(), args));
  }
}
