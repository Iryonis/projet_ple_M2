package com.ple;

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

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonArray;
import com.google.gson.JsonSyntaxException;

import java.io.IOException;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Job MapReduce pour nettoyer et valider des données de parties de jeu.
 * Vérifie que chaque ligne JSON est valide et respecte les règles métier.
 */
public class DataCleaner extends Configured implements Tool {

    /**
     * Mapper qui valide chaque ligne JSON et extrait une clé unique pour détecter
     * les doublons.
     */
    public static class CleanMapper extends Mapper<LongWritable, Text, Text, Text> {
        private final Gson gson = new Gson();
        private final Text outputKey = new Text();
        private final Text outputValue = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String line = value.toString().trim();

            // Validation 1 : JSON bien formaté
            JsonObject game;
            try {
                game = gson.fromJson(line, JsonObject.class);
                if (game == null) {
                    context.getCounter("Validation", "Null JSON").increment(1);
                    return;
                }
            } catch (JsonSyntaxException e) {
                context.getCounter("Validation", "Invalid JSON").increment(1);
                return;
            }

            // Validation 2 : Vérifier que les deux joueurs ont exactement 8 cartes
            if (!hasValidDecks(game, context)) {
                return;
            }

            // Créer une clé unique pour détecter les doublons
            String uniqueKey = buildUniqueKey(game);
            outputKey.set(uniqueKey);
            outputValue.set(line);

            context.write(outputKey, outputValue);
            context.getCounter("Validation", "Valid Lines").increment(1);
        }

        /**
         * Vérifie que les deux joueurs ont exactement 8 cartes dans leur deck.
         */
        private boolean hasValidDecks(JsonObject game, Context context) {
            try {
                JsonArray players = game.getAsJsonArray("players");
                if (players == null || players.size() != 2) {
                    context.getCounter("Validation", "Invalid Players Count").increment(1);
                    return false;
                }

                for (int i = 0; i < 2; i++) {
                    JsonObject player = players.get(i).getAsJsonObject();
                    String deck = player.get("deck").getAsString();

                    // Chaque carte est représentée par 2 caractères hexadécimaux
                    int cardCount = deck.length() / 2;
                    if (cardCount != 8) {
                        context.getCounter("Validation", "Invalid Deck Size Player" + i).increment(1);
                        return false;
                    }
                }
                return true;
            } catch (Exception e) {
                context.getCounter("Validation", "Deck Validation Error").increment(1);
                return false;
            }
        }

        /**
         * Construit une clé unique pour identifier les parties dupliquées.
         * Format : player1_utag|player2_utag|round (sans la date pour grouper les
         * parties proches)
         */
        private String buildUniqueKey(JsonObject game) {
            try {
                JsonArray players = game.getAsJsonArray("players");
                String player1 = players.get(0).getAsJsonObject().get("utag").getAsString();
                String player2 = players.get(1).getAsJsonObject().get("utag").getAsString();
                int round = game.get("round").getAsInt();

                // Clé sans date pour grouper les parties des mêmes joueurs au même round
                return String.format("%s|%s|%d", player1, player2, round);
            } catch (Exception e) {
                return game.toString(); // Fallback en cas d'erreur
            }
        }
    }

    /**
     * Reducer qui élimine les doublons exacts et les parties dupliquées
     * (mêmes joueurs, même round, dates proches à ±5 secondes).
     */
    public static class CleanReducer extends Reducer<Text, Text, NullWritable, Text> {
        private static final long TIME_THRESHOLD_SECONDS = 5;
        private final Gson gson = new Gson();
        private final Set<String> seenLines = new HashSet<>();

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            List<GameEntry> games = new ArrayList<>();

            // Collecter toutes les parties avec cette clé
            for (Text value : values) {
                String line = value.toString();

                // Éliminer les doublons exacts
                if (seenLines.contains(line)) {
                    context.getCounter("Validation", "Exact Duplicates").increment(1);
                    continue;
                }
                seenLines.add(line);

                try {
                    JsonObject game = gson.fromJson(line, JsonObject.class);
                    String dateStr = game.get("date").getAsString();
                    Instant timestamp = Instant.parse(dateStr);
                    games.add(new GameEntry(line, timestamp));
                } catch (Exception e) {
                    // Si erreur de parsing, on garde quand même la ligne
                    context.write(NullWritable.get(), value);
                    context.getCounter("Output", "Written (parse error)").increment(1);
                }
            }

            // Éliminer les parties avec timestamps proches (±5 secondes)
            for (int i = 0; i < games.size(); i++) {
                GameEntry current = games.get(i);
                boolean isDuplicate = false;

                // Vérifier si une partie antérieure proche existe déjà
                for (int j = 0; j < i; j++) {
                    GameEntry previous = games.get(j);
                    long secondsDiff = Math.abs(ChronoUnit.SECONDS.between(
                            previous.timestamp, current.timestamp));

                    if (secondsDiff <= TIME_THRESHOLD_SECONDS) {
                        isDuplicate = true;
                        context.getCounter("Validation", "Time Duplicate").increment(1);
                        break;
                    }
                }

                if (!isDuplicate) {
                    context.write(NullWritable.get(), new Text(current.line));
                    context.getCounter("Output", "Unique Lines Written").increment(1);
                }
            }
        }

        /**
         * Classe pour associer une ligne JSON à son timestamp.
         */
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
        System.out.printf("Temps d'exécution: %d min %d sec (%.2f sec)%n",
                minutes, remainingSeconds, durationMs / 1000.0);

        return success ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        System.exit(ToolRunner.run(new DataCleaner(), args));
    }
}
