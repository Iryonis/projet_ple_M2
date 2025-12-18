package com.ple;

import org.apache.hadoop.util.ProgramDriver;

public class App {

  public static void main(String[] args) throws Exception {
    ProgramDriver pgd = new ProgramDriver();
    int exitCode = -1;
    try {
      pgd.addClass("clean", DataCleaner.class, "cleaning data");
      pgd.addClass(
        "nodesedges",
        NodesEdgesGenerator.class,
        "generate nodes and edges from cleaned data"
      );
      exitCode = pgd.run(args);
    } catch (Throwable e1) {
      e1.printStackTrace();
    }
    System.exit(exitCode);
  }
}
