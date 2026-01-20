package com.ple;

/**
 * Centralized metrics (counters) for NodesEdges jobs.
 */
public final class NodesEdgesMetrics {

  private NodesEdgesMetrics() {} // Utility class

  /** Counters for nodes job */
  public static enum NodesMetrics {
    GAMES_PROCESSED,
    GAMES_SKIPPED,
    NODES_EMITTED,
    // Performance metrics
    MAPPER_INPUT_BYTES,
    MAPPER_OUTPUT_BYTES,
    MAPPER_TIME_MS,
    COMBINER_INPUT_BYTES,
    COMBINER_OUTPUT_BYTES,
    COMBINER_TIME_MS,
    REDUCER_INPUT_BYTES,
    REDUCER_OUTPUT_BYTES,
    REDUCER_TIME_MS
  }

  /** Counters for edges job */
  public static enum EdgesMetrics {
    GAMES_PROCESSED,
    GAMES_SKIPPED,
    EDGES_EMITTED,
    // Performance metrics
    MAPPER_INPUT_BYTES,
    MAPPER_OUTPUT_BYTES,
    MAPPER_TIME_MS,
    COMBINER_INPUT_BYTES,
    COMBINER_OUTPUT_BYTES,
    COMBINER_TIME_MS,
    REDUCER_INPUT_BYTES,
    REDUCER_OUTPUT_BYTES,
    REDUCER_TIME_MS
  }
}
