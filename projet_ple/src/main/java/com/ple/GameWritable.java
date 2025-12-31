package com.ple;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import org.apache.hadoop.io.Writable;

/**
 * Custom Writable for game data.
 * Avoids JSON parsing completely - uses binary format.
 *
 * Format: deck0 (8 bytes), deck1 (8 bytes), winner (1 byte) = 17 bytes total
 *
 * Each deck is 8 card IDs (1 byte each, hex 00-FF).
 */
public class GameWritable implements Writable {

  public byte[] deck0 = new byte[8];
  public byte[] deck1 = new byte[8];
  public byte winner; // 0 or 1

  @Override
  public void write(DataOutput out) throws IOException {
    out.write(deck0);
    out.write(deck1);
    out.writeByte(winner);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    in.readFully(deck0);
    in.readFully(deck1);
    winner = in.readByte();
  }

  /**
   * Parse from JSON string (backward compatibility with existing SequenceFile).
   * Uses manual parsing instead of Gson for 10x+ speedup.
   *
   * Expected JSON format:
   * {"players":[{"deck":"0102030405060708"},{"deck":"090a0b0c0d0e0f10"}],"winner":0}
   *
   * @param json The JSON string to parse
   * @return true if parsing succeeded, false otherwise
   */
  public boolean parseFromJson(String json) {
    try {
      // Find first deck
      int deck0Idx = json.indexOf("\"deck\":\"");
      if (deck0Idx == -1) return false;
      deck0Idx += 8; // Skip '"deck":"'
      if (deck0Idx + 16 > json.length()) return false;

      // Parse deck0 (16 hex chars -> 8 bytes)
      for (int i = 0; i < 8; i++) {
        int hi = Character.digit(json.charAt(deck0Idx + i * 2), 16);
        int lo = Character.digit(json.charAt(deck0Idx + i * 2 + 1), 16);
        if (hi == -1 || lo == -1) return false;
        deck0[i] = (byte) ((hi << 4) | lo);
      }

      // Find second deck
      int deck1Idx = json.indexOf("\"deck\":\"", deck0Idx + 16);
      if (deck1Idx == -1) return false;
      deck1Idx += 8;
      if (deck1Idx + 16 > json.length()) return false;

      // Parse deck1
      for (int i = 0; i < 8; i++) {
        int hi = Character.digit(json.charAt(deck1Idx + i * 2), 16);
        int lo = Character.digit(json.charAt(deck1Idx + i * 2 + 1), 16);
        if (hi == -1 || lo == -1) return false;
        deck1[i] = (byte) ((hi << 4) | lo);
      }

      // Find winner
      int winnerIdx = json.indexOf("\"winner\":");
      if (winnerIdx == -1) return false;
      winnerIdx += 9; // Skip '"winner":'

      // Skip whitespace
      while (
        winnerIdx < json.length() &&
        (json.charAt(winnerIdx) == ' ' || json.charAt(winnerIdx) == '\t')
      ) {
        winnerIdx++;
      }

      if (winnerIdx >= json.length()) return false;
      char winnerChar = json.charAt(winnerIdx);
      if (winnerChar != '0' && winnerChar != '1') return false;
      winner = (byte) (winnerChar - '0');

      return true;
    } catch (Exception e) {
      return false;
    }
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("GameWritable{deck0=");
    for (byte b : deck0) sb.append(String.format("%02x", b & 0xFF));
    sb.append(", deck1=");
    for (byte b : deck1) sb.append(String.format("%02x", b & 0xFF));
    sb.append(", winner=").append(winner).append("}");
    return sb.toString();
  }
}
