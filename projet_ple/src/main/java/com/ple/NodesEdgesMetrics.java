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
    NODES_EMITTED
  }

  /** Counters for edges job */
  public static enum EdgesMetrics {
    GAMES_PROCESSED,
    GAMES_SKIPPED,
    EDGES_EMITTED
  }
}
