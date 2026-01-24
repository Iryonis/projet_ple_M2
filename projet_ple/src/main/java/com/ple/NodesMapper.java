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
 * Mapper for nodes generation.
 * 
 * For each game, generates all C(8,k) archetypes for both decks.
 * Uses in-mapper combining for massive reduction of shuffle data.
 * 
 * Input: SequenceFile<NullWritable, Text> (JSON game records)
 * Output: <archetype (long), packed(count, wins)>
 */
public class NodesMapper extends Mapper<NullWritable, Text, LongWritable, LongWritable> {

  // Reusable objects to avoid GC pressure
  private final GameWritable game = new GameWritable();
  private final LongWritable outKey = new LongWritable();
  private final LongWritable outVal = new LongWritable();

  // Counting sort buffers
  private final byte[] buf0 = new byte[8];
  private final byte[] buf1 = new byte[8];
  private final int[] cnt = new int[256];

  // Configuration
  private int k;

  // Counters
  private Counter processed, skipped, emitted;

  // IN-MAPPER COMBINING: Local aggregation map
  // Key: archetype (long), Value: [count, wins]
  private HashMap<Long, long[]> localMap;
  private long localEmitCount;
  private long totalTimeNs = 0;

  @Override
  protected void setup(Context ctx) {
    k = ctx.getConfiguration().getInt("archetype.size", 7);
    processed = ctx.getCounter(NodesEdgesMetrics.NodesMetrics.GAMES_PROCESSED);
    skipped = ctx.getCounter(NodesEdgesMetrics.NodesMetrics.GAMES_SKIPPED);
    emitted = ctx.getCounter(NodesEdgesMetrics.NodesMetrics.NODES_EMITTED);
    localMap = new HashMap<>(MAX_MAP_SIZE / 2);
    localEmitCount = 0;
  }

  @Override
  protected void map(NullWritable key, Text val, Context ctx)
      throws IOException, InterruptedException {
    long startTime = System.nanoTime();
    
    // Count input bytes
    ctx.getCounter(NodesEdgesMetrics.NodesMetrics.MAPPER_INPUT_BYTES)
        .increment(val.getLength());
    
    if (!game.parseFromJson(val.toString())) {
      skipped.increment(1);
      totalTimeNs += (System.nanoTime() - startTime);
      return;
    }
    processed.increment(1);

    byte[] s0 = sort(game.deck0, buf0, cnt);
    byte[] s1 = sort(game.deck1, buf1, cnt);

    emitNodes(s0, game.winner == 0 ? 1 : 0);
    emitNodes(s1, game.winner == 1 ? 1 : 0);

    // Flush if map is getting too large
    if (localMap.size() >= MAX_MAP_SIZE * FLUSH_THRESHOLD) {
      flush(ctx);
    }
    
    // Accumulate execution time
    totalTimeNs += (System.nanoTime() - startTime);
  }

  @Override
  protected void cleanup(Context ctx) throws IOException, InterruptedException {
    flush(ctx);
    emitted.increment(localEmitCount);
    // Convert accumulated time to milliseconds
    ctx.getCounter(NodesEdgesMetrics.NodesMetrics.MAPPER_TIME_MS)
        .increment(totalTimeNs / 1_000_000);
  }

  /** Flush local map to context */
  private void flush(Context ctx) throws IOException, InterruptedException {
    for (Map.Entry<Long, long[]> e : localMap.entrySet()) {
      outKey.set(e.getKey());
      long[] v = e.getValue();
      outVal.set(pack(v[0], v[1]));
      ctx.write(outKey, outVal);
      // Count output bytes (key + value = 8 + 8 = 16 bytes)
      ctx.getCounter(NodesEdgesMetrics.NodesMetrics.MAPPER_OUTPUT_BYTES)
          .increment(16);
      localEmitCount++;
    }
    localMap.clear();
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

  /** Generate all C(8,k) archetypes for a deck */
  private void emitNodes(byte[] d, int w) {
    switch (k) {
      case 1:
        for (int i = 0; i < 8; i++) {
          emit(encode1(d, i), w);
        }
        break;
      case 2:
        for (int i = 0; i < 7; i++)
          for (int j = i + 1; j < 8; j++) {
            emit(encode2(d, i, j), w);
          }
        break;
      case 3:
        for (int i = 0; i < 6; i++)
          for (int j = i + 1; j < 7; j++)
            for (int k = j + 1; k < 8; k++) {
              emit(encode3(d, i, j, k), w);
            }
        break;
      case 4:
        for (int i = 0; i < 5; i++)
          for (int j = i + 1; j < 6; j++)
            for (int k = j + 1; k < 7; k++)
              for (int l = k + 1; l < 8; l++) {
                emit(encode4(d, i, j, k, l), w);
              }
        break;
      case 5:
        for (int i = 0; i < 4; i++)
          for (int j = i + 1; j < 5; j++)
            for (int k = j + 1; k < 6; k++)
              for (int l = k + 1; l < 7; l++)
                for (int m = l + 1; m < 8; m++) {
                  emit(encode5(d, i, j, k, l, m), w);
                }
        break;
      case 6:
        for (int i = 0; i < 3; i++)
          for (int j = i + 1; j < 4; j++)
            for (int k = j + 1; k < 5; k++)
              for (int l = k + 1; l < 6; l++)
                for (int m = l + 1; m < 7; m++)
                  for (int n = m + 1; n < 8; n++) {
                    emit(encode6(d, i, j, k, l, m, n), w);
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
                      emit(encode7(d, i, j, k, l, m, n, o), w);
                    }
        break;
      case 8:
        emit(encode8(d), w);
        break;
    }
  }
}
