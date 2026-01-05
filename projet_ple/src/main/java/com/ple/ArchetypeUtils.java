package com.ple;

import java.util.Arrays;

/**
 * Utility class for archetype (deck subset) operations.
 * Centralizes counting sort and hex conversion logic.
 * 
 * An archetype is a subset of k cards from a deck of 8 cards,
 * encoded as a long for efficient storage and comparison.
 */
public final class ArchetypeUtils {

  private ArchetypeUtils() {} // Utility class

  // ==================== CONSTANTS ====================

  /** Max entries in local aggregation map before flush */
  public static final int MAX_MAP_SIZE = 500_000;

  /** Flush threshold as percentage of max */
  public static final float FLUSH_THRESHOLD = 0.9f;

  // ==================== COUNTING SORT ====================

  /**
   * Counting sort for 8 bytes (card IDs 0-255).
   * O(8 + 256) = O(1) complexity - faster than comparison sort.
   * 
   * @param input  Source deck (8 bytes)
   * @param output Output buffer (8 bytes) - must be different from input
   * @param cnt    Reusable count array (256 elements)
   * @return The output buffer for chaining
   */
  public static byte[] sort(byte[] input, byte[] output, int[] cnt) {
    Arrays.fill(cnt, 0);
    for (int i = 0; i < 8; i++) {
      cnt[input[i] & 0xFF]++;
    }
    int idx = 0;
    for (int v = 0; v < 256; v++) {
      while (cnt[v]-- > 0) {
        output[idx++] = (byte) v;
      }
    }
    return output;
  }

  // ==================== HEX CONVERSION ====================

  /**
   * Convert a packed long archetype to hex string.
   * 
   * @param value The packed archetype value
   * @param numBytes Number of bytes (k) in the archetype
   * @return Hex string representation (2*k characters)
   */
  public static String toHex(long value, int numBytes) {
    StringBuilder sb = new StringBuilder(numBytes * 2);
    for (int i = numBytes - 1; i >= 0; i--) {
      int b = (int) ((value >> (i * 8)) & 0xFF);
      sb.append(Character.forDigit((b >> 4) & 0xF, 16));
      sb.append(Character.forDigit(b & 0xF, 16));
    }
    return sb.toString();
  }

  // ==================== ARCHETYPE ENCODING ====================

  /** Encode 1-card archetype as long */
  public static long encode1(byte[] d, int i) {
    return d[i] & 0xFFL;
  }

  /** Encode 2-card archetype as long */
  public static long encode2(byte[] d, int i, int j) {
    return ((long) (d[i] & 0xFF) << 8) | (d[j] & 0xFF);
  }

  /** Encode 3-card archetype as long */
  public static long encode3(byte[] d, int i, int j, int k) {
    return ((long) (d[i] & 0xFF) << 16) |
           ((long) (d[j] & 0xFF) << 8) |
           (d[k] & 0xFF);
  }

  /** Encode 4-card archetype as long */
  public static long encode4(byte[] d, int i, int j, int k, int l) {
    return ((long) (d[i] & 0xFF) << 24) |
           ((long) (d[j] & 0xFF) << 16) |
           ((long) (d[k] & 0xFF) << 8) |
           (d[l] & 0xFF);
  }

  /** Encode 5-card archetype as long */
  public static long encode5(byte[] d, int i, int j, int k, int l, int m) {
    return ((long) (d[i] & 0xFF) << 32) |
           ((long) (d[j] & 0xFF) << 24) |
           ((long) (d[k] & 0xFF) << 16) |
           ((long) (d[l] & 0xFF) << 8) |
           (d[m] & 0xFF);
  }

  /** Encode 6-card archetype as long */
  public static long encode6(byte[] d, int i, int j, int k, int l, int m, int n) {
    return ((long) (d[i] & 0xFF) << 40) |
           ((long) (d[j] & 0xFF) << 32) |
           ((long) (d[k] & 0xFF) << 24) |
           ((long) (d[l] & 0xFF) << 16) |
           ((long) (d[m] & 0xFF) << 8) |
           (d[n] & 0xFF);
  }

  /** Encode 7-card archetype as long */
  public static long encode7(byte[] d, int i, int j, int k, int l, int m, int n, int o) {
    return ((long) (d[i] & 0xFF) << 48) |
           ((long) (d[j] & 0xFF) << 40) |
           ((long) (d[k] & 0xFF) << 32) |
           ((long) (d[l] & 0xFF) << 24) |
           ((long) (d[m] & 0xFF) << 16) |
           ((long) (d[n] & 0xFF) << 8) |
           (d[o] & 0xFF);
  }

  /** Encode 8-card archetype (full deck) as long */
  public static long encode8(byte[] d) {
    return ((long) (d[0] & 0xFF) << 56) |
           ((long) (d[1] & 0xFF) << 48) |
           ((long) (d[2] & 0xFF) << 40) |
           ((long) (d[3] & 0xFF) << 32) |
           ((long) (d[4] & 0xFF) << 24) |
           ((long) (d[5] & 0xFF) << 16) |
           ((long) (d[6] & 0xFF) << 8) |
           (d[7] & 0xFF);
  }

  // ==================== MATH ====================

  /**
   * Compute binomial coefficient C(n,k).
   */
  public static int binomial(int n, int k) {
    if (k > n - k) k = n - k;
    int r = 1;
    for (int i = 0; i < k; i++) {
      r = r * (n - i) / (i + 1);
    }
    return r;
  }

  // ==================== VALUE PACKING ====================

  /**
   * Pack count and wins into a single long.
   * Format: (count << 32) | wins
   * 
   * @param count Number of occurrences
   * @param wins Number of wins
   * @return Packed value
   */
  public static long pack(long count, long wins) {
    return (count << 32) | (wins & 0xFFFFFFFFL);
  }

  /**
   * Extract count from packed value.
   */
  public static long unpackCount(long packed) {
    return (packed >> 32) & 0xFFFFFFFFL;
  }

  /**
   * Extract wins from packed value.
   */
  public static long unpackWins(long packed) {
    return packed & 0xFFFFFFFFL;
  }
}
