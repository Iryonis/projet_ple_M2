package com.ple;

import static com.ple.ArchetypeUtils.*;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.Mapper;

/**
 * Mapper for edges generation.
 * 
 * For each game, generates all C(8,k)² directed matchups between deck archetypes.
 * Uses in-mapper combining with nested HashMap for massive shuffle reduction.
 * 
 * Input: SequenceFile<NullWritable, Text> (JSON game records)
 * Output: <EdgeKey(source, target), packed(count, wins)>
 */
public class EdgesMapper extends Mapper<NullWritable, Text, EdgeKey, LongWritable> {

  // Reusable objects
  private final GameWritable game = new GameWritable();
  private final EdgeKey outKey = new EdgeKey();
  private final LongWritable outVal = new LongWritable();

  // Counting sort buffers
  private final byte[] buf0 = new byte[8];
  private final byte[] buf1 = new byte[8];
  private final int[] cnt = new int[256];

  // Configuration
  private int k;

  // Counters
  private Counter processed, skipped, emitted;

  // IN-MAPPER COMBINING: Nested map source -> (target -> [count, wins])
  private HashMap<Long, HashMap<Long, long[]>> localEdgeMap;
  private long localEmitCount;
  private int localMapEntries;

  @Override
  protected void setup(Context ctx) {
    k = ctx.getConfiguration().getInt("archetype.size", 7);
    processed = ctx.getCounter(NodesEdgesMetrics.EdgesMetrics.GAMES_PROCESSED);
    skipped = ctx.getCounter(NodesEdgesMetrics.EdgesMetrics.GAMES_SKIPPED);
    emitted = ctx.getCounter(NodesEdgesMetrics.EdgesMetrics.EDGES_EMITTED);
    localEdgeMap = new HashMap<>(16384);
    localEmitCount = 0;
    localMapEntries = 0;
  }

  @Override
  protected void map(NullWritable key, Text val, Context ctx)
      throws IOException, InterruptedException {
    long startTime = System.nanoTime();
    
    // Count input bytes
    ctx.getCounter(NodesEdgesMetrics.EdgesMetrics.MAPPER_INPUT_BYTES)
        .increment(val.getLength());
    
    if (!game.parseFromJson(val.toString())) {
      skipped.increment(1);
      ctx.getCounter(NodesEdgesMetrics.EdgesMetrics.MAPPER_TIME_MS)
          .increment((System.nanoTime() - startTime) / 1_000_000);
      return;
    }
    processed.increment(1);

    byte[] s0 = sort(game.deck0, buf0, cnt);
    byte[] s1 = sort(game.deck1, buf1, cnt);
    emitEdges(s0, s1, game.winner);

    // Flush if map is getting too large
    if (localMapEntries >= MAX_MAP_SIZE * FLUSH_THRESHOLD) {
      flush(ctx);
    }
    
    // Measure execution time
    ctx.getCounter(NodesEdgesMetrics.EdgesMetrics.MAPPER_TIME_MS)
        .increment((System.nanoTime() - startTime) / 1_000_000);
  }

  @Override
  protected void cleanup(Context ctx) throws IOException, InterruptedException {
    flush(ctx);
    emitted.increment(localEmitCount);
  }

  /** Flush local edge map to context */
  private void flush(Context ctx) throws IOException, InterruptedException {
    for (Map.Entry<Long, HashMap<Long, long[]>> srcEntry : localEdgeMap.entrySet()) {
      long src = srcEntry.getKey();
      for (Map.Entry<Long, long[]> tgtEntry : srcEntry.getValue().entrySet()) {
        long tgt = tgtEntry.getKey();
        long[] v = tgtEntry.getValue();
        outKey.set(src, tgt);
        outVal.set(pack(v[0], v[1]));
        ctx.write(outKey, outVal);
        // Count output bytes (EdgeKey = 16 bytes + value = 8 bytes = 24 bytes)
        ctx.getCounter(NodesEdgesMetrics.EdgesMetrics.MAPPER_OUTPUT_BYTES)
            .increment(24);
        localEmitCount++;
      }
    }
    localEdgeMap.clear();
    localMapEntries = 0;
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

  /** Generate all C(8,k)² directed edges between two decks */
  private void emitEdges(byte[] d0, byte[] d1, int winner) {
    int w0 = winner == 0 ? 1 : 0;
    int w1 = winner == 1 ? 1 : 0;

    switch (k) {
      case 1:
        for (int i = 0; i < 8; i++) {
          long x = encode1(d0, i);
          for (int j = 0; j < 8; j++) {
            long y = encode1(d1, j);
            emit(x, y, w0);
            emit(y, x, w1);
          }
        }
        break;
      case 2:
        for (int i = 0; i < 7; i++)
          for (int j = i + 1; j < 8; j++) {
            long x = encode2(d0, i, j);
            for (int a = 0; a < 7; a++)
              for (int b = a + 1; b < 8; b++) {
                long y = encode2(d1, a, b);
                emit(x, y, w0);
                emit(y, x, w1);
              }
          }
        break;
      case 3:
        for (int i = 0; i < 6; i++)
          for (int j = i + 1; j < 7; j++)
            for (int k = j + 1; k < 8; k++) {
              long x = encode3(d0, i, j, k);
              for (int a = 0; a < 6; a++)
                for (int b = a + 1; b < 7; b++)
                  for (int c = b + 1; c < 8; c++) {
                    long y = encode3(d1, a, b, c);
                    emit(x, y, w0);
                    emit(y, x, w1);
                  }
            }
        break;
      case 4:
        for (int i = 0; i < 5; i++)
          for (int j = i + 1; j < 6; j++)
            for (int k = j + 1; k < 7; k++)
              for (int l = k + 1; l < 8; l++) {
                long x = encode4(d0, i, j, k, l);
                for (int a = 0; a < 5; a++)
                  for (int b = a + 1; b < 6; b++)
                    for (int c = b + 1; c < 7; c++)
                      for (int e = c + 1; e < 8; e++) {
                        long y = encode4(d1, a, b, c, e);
                        emit(x, y, w0);
                        emit(y, x, w1);
                      }
              }
        break;
      case 5:
        for (int i = 0; i < 4; i++)
          for (int j = i + 1; j < 5; j++)
            for (int k = j + 1; k < 6; k++)
              for (int l = k + 1; l < 7; l++)
                for (int m = l + 1; m < 8; m++) {
                  long x = encode5(d0, i, j, k, l, m);
                  for (int a = 0; a < 4; a++)
                    for (int b = a + 1; b < 5; b++)
                      for (int c = b + 1; c < 6; c++)
                        for (int e = c + 1; e < 7; e++)
                          for (int f = e + 1; f < 8; f++) {
                            long y = encode5(d1, a, b, c, e, f);
                            emit(x, y, w0);
                            emit(y, x, w1);
                          }
                }
        break;
      case 6:
        for (int i = 0; i < 3; i++)
          for (int j = i + 1; j < 4; j++)
            for (int k = j + 1; k < 5; k++)
              for (int l = k + 1; l < 6; l++)
                for (int m = l + 1; m < 7; m++)
                  for (int n = m + 1; n < 8; n++) {
                    long x = encode6(d0, i, j, k, l, m, n);
                    for (int a = 0; a < 3; a++)
                      for (int b = a + 1; b < 4; b++)
                        for (int c = b + 1; c < 5; c++)
                          for (int e = c + 1; e < 6; e++)
                            for (int f = e + 1; f < 7; f++)
                              for (int g = f + 1; g < 8; g++) {
                                long y = encode6(d1, a, b, c, e, f, g);
                                emit(x, y, w0);
                                emit(y, x, w1);
                              }
                  }
        break;
      case 7:
        for (int i = 0; i < 2; i++)
          for (int j = i + 1; j < 3; j++)
            for (int k = j + 1; k < 4; k++)
              for (int l = k + 1; l < 5; l++)
                for (int m = l + 1; m < 6; m++)
                  for (int n = m + 1; n < 7; n++)
                    for (int o = n + 1; o < 8; o++) {
                      long x = encode7(d0, i, j, k, l, m, n, o);
                      for (int a = 0; a < 2; a++)
                        for (int b = a + 1; b < 3; b++)
                          for (int c = b + 1; c < 4; c++)
                            for (int e = c + 1; e < 5; e++)
                              for (int f = e + 1; f < 6; f++)
                                for (int g = f + 1; g < 7; g++)
                                  for (int h = g + 1; h < 8; h++) {
                                    long y = encode7(d1, a, b, c, e, f, g, h);
                                    emit(x, y, w0);
                                    emit(y, x, w1);
                                  }
                    }
        break;
      case 8:
        {
          long x = encode8(d0);
          long y = encode8(d1);
          emit(x, y, w0);
          emit(y, x, w1);
        }
        break;
    }
  }
}
