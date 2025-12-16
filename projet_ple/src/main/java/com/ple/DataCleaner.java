package com.ple;

import com.google.gson.Gson; // Gson for small files / Try Jackson for bigger ?
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.io.IOException;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

/**
 * Job MapReduce pour nettoyer et valider des données de parties de jeu.
 * Vérifie que chaque ligne JSON est valide et respecte les règles métier.
 */
public class DataCleaner extends Configured implements Tool {

  /**
   * Mapper qui valide chaque ligne JSON et extrait une clé unique pour détecter
   * les doublons.
   */
  public static class CleanMapper
    extends Mapper<LongWritable, Text, Text, Text> {

    private final Gson gson = new Gson();
    private final Text outputKey = new Text();
    private final Text outputValue = new Text();

    @Override
    protected void map(LongWritable key, Text value, Context context)
      throws IOException, InterruptedException {
      String line = value.toString().trim();

      // PARSING JSON UNE SEULE FOIS
      JsonObject game;
      try {
        game = gson.fromJson(line, JsonObject.class);
        if (game == null) {
          return;
        }
      } catch (Exception e) {
        return;
      }

      // Validation et extraction en un seul passage
      String uniqueKey = validateAndExtract(game);
      if (uniqueKey == null) {
        return;
      }

      // Utiliser les données déjà extraites
      outputKey.set(uniqueKey);
      outputValue.set(line);

      context.write(outputKey, outputValue);
    }

    /**
     * Valide le JSON et extrait toutes les données nécessaires en un seul passage.
     * Retourne la clé unique ou null si invalide.
     */
    private String validateAndExtract(JsonObject game) {
      try {
        // Fail-fast : extraire directement sans vérifications multiples
        JsonArray players = game.getAsJsonArray("players");
        if (players == null || players.size() != 2) {
          return null;
        }

        JsonObject player0 = players.get(0).getAsJsonObject();
        JsonObject player1 = players.get(1).getAsJsonObject();

        // Extraire directement les champs nécessaires
        String utag0 = player0.get("utag").getAsString();
        String utag1 = player1.get("utag").getAsString();
        String deck0 = player0.get("deck").getAsString();
        String deck1 = player1.get("deck").getAsString();
        String dateStr = game.get("date").getAsString();
        int round = game.get("round").getAsInt();

        // Validation rapide des decks (8 cartes exactement)
        if (deck0.length() != 16 || deck1.length() != 16) {
          return null;
        }

        // Extraire la date (juste YYYY-MM-DD)
        String dateOnly = dateStr.substring(0, 10);

        // Trier les utags pour normaliser la clé
        String sortedUtag0, sortedUtag1;
        if (utag0.compareTo(utag1) < 0) {
          sortedUtag0 = utag0;
          sortedUtag1 = utag1;
        } else {
          sortedUtag0 = utag1;
          sortedUtag1 = utag0;
        }

        // Retourner directement la clé
        return String.format(
          "%s|%s|%s|%d",
          sortedUtag0,
          sortedUtag1,
          dateOnly,
          round
        );
      } catch (Exception e) {
        return null;
      }
    }
  }

  /**
   * Combiner qui élimine les doublons exacts localement avant l'envoi au Reducer.
   * Réduit le volume de données transitant sur le réseau.
   */
  public static class CleanCombiner extends Reducer<Text, Text, Text, Text> {

    private final Set<String> seenLines = new HashSet<>();

    @Override
    protected void reduce(Text key, Iterable<Text> values, Context context)
      throws IOException, InterruptedException {
      seenLines.clear();

      for (Text value : values) {
        String line = value.toString();

        // Déduplication simple avec la ligne complète
        if (!seenLines.contains(line)) {
          seenLines.add(line);
          context.write(key, value);
        }
      }
    }
  }

  public static class CleanReducer
    extends Reducer<Text, Text, NullWritable, Text> {

    private final Gson gson = new Gson();
    private static final long TIME_THRESHOLD_SECONDS = 10;
    private final Set<Integer> seenHashes = new HashSet<>();

    @Override
    protected void reduce(Text key, Iterable<Text> values, Context context)
      throws IOException, InterruptedException {
      List<GameEntry> games = new ArrayList<>();

      // Collecter toutes les parties avec cette clé
      for (Text value : values) {
        String line = value.toString();

        // Utiliser hashCode() natif (bien plus rapide que SHA-256)
        int lineHash = line.hashCode();

        // Éliminer les doublons exacts via hash
        if (seenHashes.contains(lineHash)) {
          continue;
        }
        seenHashes.add(lineHash);

        try {
          JsonObject game = gson.fromJson(line, JsonObject.class);
          String dateStr = game.get("date").getAsString();
          Instant timestamp = Instant.parse(dateStr);
          games.add(new GameEntry(line, timestamp));
        } catch (Exception e) {
          // Si erreur de parsing, on garde quand même la ligne
          context.write(NullWritable.get(), value);
        }
      }

      // Éliminer les parties avec timestamps proches (±10 secondes)
      for (int i = 0; i < games.size(); i++) {
        GameEntry current = games.get(i);
        boolean isDuplicate = false;

        // Vérifier si une partie antérieure proche existe déjà
        for (int j = 0; j < i; j++) {
          GameEntry previous = games.get(j);
          long secondsDiff = Math.abs(
            ChronoUnit.SECONDS.between(previous.timestamp, current.timestamp)
          );

          if (secondsDiff <= TIME_THRESHOLD_SECONDS) {
            isDuplicate = true;
            break;
          }
        }

        if (!isDuplicate) {
          context.write(NullWritable.get(), new Text(current.line));
        }
      }

      // Nettoyer le HashSet pour éviter l'accumulation mémoire
      seenHashes.clear();
    }

    private static class GameEntry {

      final String line;
      final Instant timestamp;

      GameEntry(String line, Instant timestamp) {
        this.line = line;
        this.timestamp = timestamp;
      }
    }
  }

  @Override
  public int run(String[] args) throws Exception {
    if (args.length != 2) {
      System.err.println("Usage: DataCleaner <input> <output>");
      return 1;
    }

    long startTime = System.currentTimeMillis();
    System.out.println("=== Démarrage du job Data Cleaner ===");

    Configuration conf = getConf();
    Job job = Job.getInstance(conf, "Data Cleaner");
    job.setJarByClass(DataCleaner.class);

    // Configuration du Mapper
    job.setMapperClass(CleanMapper.class);
    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(Text.class);

    // Configuration du Combiner
    job.setCombinerClass(CleanCombiner.class);

    // Configuration du Reducer
    job.setReducerClass(CleanReducer.class);
    job.setOutputKeyClass(NullWritable.class);
    job.setOutputValueClass(Text.class);

    // Chemins d'entrée/sortie
    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));

    boolean success = job.waitForCompletion(true);

    long endTime = System.currentTimeMillis();
    long durationMs = endTime - startTime;
    long seconds = durationMs / 1000;
    long minutes = seconds / 60;
    long remainingSeconds = seconds % 60;

    System.out.println("\n=== Job terminé ===");
    System.out.printf(
      "Temps d'exécution: %d min %d sec (%.2f sec)%n",
      minutes,
      remainingSeconds,
      durationMs / 1000.0
    );

    return success ? 0 : 1;
  }

  public static void main(String[] args) throws Exception {
    System.exit(ToolRunner.run(new DataCleaner(), args));
  }
}
