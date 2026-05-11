"""
Distributed transformation stage (PySpark).

Reads raw bronze TSV from S3, applies the same cleaning + event aggregation
logic as the local pandas pipeline using PySpark, and writes silver + gold
Parquet tables back to S3.

This is the project's distributed-compute layer. It mirrors the local
pandas pipeline so the deployed Streamlit app can rely on either source.

Run (local Spark, reads/writes to S3):
    AWS_PROFILE=default \
    python src/spark_transform.py \
        --bronze s3://kava-chess-pipeline/bronze/kava_chess_games.tsv \
        --silver s3://kava-chess-pipeline/silver/games \
        --gold   s3://kava-chess-pipeline/gold/event_features

You can also run it against local paths for development:
    python src/spark_transform.py \
        --bronze data/raw/kava_chess_games.tsv \
        --silver data/silver/games_spark \
        --gold   data/gold/event_features_spark
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType


NAME_MERGES = {
    "harold": "Gonzalez, Harold",
    "gonzalez, harold": "Gonzalez, Harold",
    "cruz, omar": "Cruz, Omar",
}


def _norm_name_expr(col):
    """SQL expression that normalizes a player name column."""
    trimmed = F.trim(F.regexp_replace(col, r"\s+", " "))
    lowered = F.lower(trimmed)
    merged = (
        F.when(lowered == F.lit("harold"), F.lit("Gonzalez, Harold"))
        .when(lowered == F.lit("gonzalez, harold"), F.lit("Gonzalez, Harold"))
        .when(lowered == F.lit("cruz, omar"), F.lit("Cruz, Omar"))
        .otherwise(trimmed)
    )
    # null out empties and literal "Null"
    return F.when(
        (lowered == F.lit("null")) | (trimmed == F.lit("")) | trimmed.isNull(),
        F.lit(None).cast("string"),
    ).otherwise(merged)


def build_spark(app_name: str = "kava-chess-attendance") -> SparkSession:
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.session.timeZone", "America/New_York")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
        .config("spark.jars.packages",
                "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262")
    )
    return builder.getOrCreate()


def clean_games(spark: SparkSession, bronze_path: str):
    raw = (
        spark.read.option("header", True).option("sep", "\t").csv(bronze_path)
    )
    # Drop repeated header rows
    raw = raw.filter(~((F.trim(F.col("Date")) == "Date") & (F.trim(F.col("White")) == "White")))
    raw = raw.filter(F.trim(F.col("Date")) != "")

    df = raw.select(
        F.to_date(F.trim(F.col("Date")), "MMM d, yyyy").alias("event_date"),
        F.trim(F.col("Time")).alias("game_time_raw"),
        _norm_name_expr(F.col("White")).alias("white_player"),
        _norm_name_expr(F.col("Black")).alias("black_player"),
        F.trim(F.col("Winner")).alias("winner_raw"),
        F.coalesce(F.trim(F.col("Comments")), F.lit("")).alias("comments"),
    ).filter(F.col("event_date").isNotNull())

    df = df.withColumn(
        "result_kind",
        F.when(F.lower(F.col("winner_raw")) == F.lit("draw"), F.lit("draw"))
        .when((F.col("winner_raw").isNull()) | (F.col("winner_raw") == F.lit("")), F.lit("unknown"))
        .otherwise(F.lit("decisive")),
    ).withColumn(
        "winner_player",
        F.when(F.col("result_kind") == F.lit("draw"), F.lit(None).cast("string"))
        .otherwise(_norm_name_expr(F.col("winner_raw"))),
    )

    df = df.withColumn(
        "result_type",
        F.when(F.col("result_kind") == F.lit("draw"), F.lit("draw"))
        .when(F.col("winner_player") == F.col("white_player"), F.lit("white_won"))
        .when(F.col("winner_player") == F.col("black_player"), F.lit("black_won"))
        .otherwise(F.lit("unknown")),
    )

    # Drop fully empty player rows
    df = df.filter(F.col("white_player").isNotNull() | F.col("black_player").isNotNull())

    # Dedup
    df = df.dropDuplicates(
        ["event_date", "game_time_raw", "white_player", "black_player", "winner_player"]
    )

    return df.select(
        "event_date",
        "game_time_raw",
        "white_player",
        "black_player",
        "winner_player",
        "result_type",
        "result_kind",
        "comments",
    )


def build_event_features(spark: SparkSession, silver_df):
    # Long-form attendees: one row per (event_date, player)
    whites = silver_df.select(
        "event_date", F.col("white_player").alias("player")
    ).filter(F.col("player").isNotNull())
    blacks = silver_df.select(
        "event_date", F.col("black_player").alias("player")
    ).filter(F.col("player").isNotNull())
    attendees = whites.union(blacks).dropDuplicates(["event_date", "player"])

    # First-appearance date per player (for new vs returning)
    first_seen = attendees.groupBy("player").agg(F.min("event_date").alias("first_seen"))
    attendees = attendees.join(first_seen, "player", "left")
    attendees = attendees.withColumn(
        "is_new", (F.col("event_date") == F.col("first_seen")).cast(IntegerType())
    )

    per_event_players = attendees.groupBy("event_date").agg(
        F.countDistinct("player").alias("unique_players"),
        F.sum("is_new").alias("new_players_count"),
    )
    per_event_players = per_event_players.withColumn(
        "returning_players_count",
        F.col("unique_players") - F.col("new_players_count"),
    )

    per_event_games = silver_df.groupBy("event_date").agg(
        F.count(F.lit(1)).alias("num_games"),
        F.sum(F.when(F.col("result_type") == "draw", 1).otherwise(0)).alias("num_draws"),
    )

    ev = per_event_players.join(per_event_games, "event_date", "outer")
    ev = ev.withColumn("attendance_count", F.col("unique_players")).withColumn(
        "games_per_player",
        F.when(F.col("unique_players") > 0, F.col("num_games") / F.col("unique_players")).otherwise(0.0),
    ).withColumn(
        "draw_rate",
        F.when(F.col("num_games") > 0, F.col("num_draws") / F.col("num_games")).otherwise(0.0),
    )

    w = Window.orderBy("event_date")
    w3 = w.rowsBetween(-3, -1)
    w5 = w.rowsBetween(-5, -1)
    w10 = w.rowsBetween(-10, -1)

    ev = (
        ev.withColumn("previous_event_attendance", F.lag("attendance_count").over(w))
        .withColumn("attendance_two_events_ago", F.lag("attendance_count", 2).over(w))
        .withColumn("rolling_3_event_attendance", F.avg("attendance_count").over(w3))
        .withColumn("rolling_5_event_attendance", F.avg("attendance_count").over(w5))
        .withColumn("rolling_10_event_attendance", F.avg("attendance_count").over(w10))
        .withColumn("previous_event_num_games", F.lag("num_games").over(w))
        .withColumn("previous_event_unique_players", F.lag("unique_players").over(w))
        .withColumn("previous_event_new_players_count", F.lag("new_players_count").over(w))
        .withColumn("previous_event_returning_players_count", F.lag("returning_players_count").over(w))
        .withColumn("previous_event_draw_rate", F.lag("draw_rate").over(w))
        .withColumn("previous_event_games_per_player", F.lag("games_per_player").over(w))
        .withColumn("rolling_3_avg_num_games", F.avg("num_games").over(w3))
        .withColumn("rolling_3_avg_new_players", F.avg("new_players_count").over(w3))
        .withColumn("rolling_3_avg_returning_players", F.avg("returning_players_count").over(w3))
        .withColumn("days_since_last_event", F.datediff(F.col("event_date"), F.lag("event_date").over(w)))
        .withColumn(
            "biweekly_event_indicator",
            ((F.col("days_since_last_event") >= 12) & (F.col("days_since_last_event") <= 16)).cast(IntegerType()),
        )
        .withColumn(
            "weekly_event_indicator",
            ((F.col("days_since_last_event") >= 5) & (F.col("days_since_last_event") <= 9)).cast(IntegerType()),
        )
        .withColumn(
            "back_to_back_event_indicator",
            (F.col("days_since_last_event") <= 3).cast(IntegerType()),
        )
        .withColumn(
            "first_event_after_long_break",
            (F.col("days_since_last_event") >= 30).cast(IntegerType()),
        )
        .withColumn("year", F.year("event_date"))
        .withColumn("month", F.month("event_date"))
        .withColumn("day_of_month", F.dayofmonth("event_date"))
        .withColumn("day_of_week", F.dayofweek("event_date") - 1)  # 0..6 Mon..Sun-ish
        .withColumn("is_sunday", (F.dayofweek("event_date") == 1).cast(IntegerType()))
        .withColumn("week_of_year", F.weekofyear("event_date"))
        .withColumn("is_beginning_of_month", (F.col("day_of_month") <= 7).cast(IntegerType()))
        .withColumn("is_end_of_month", (F.col("day_of_month") >= 24).cast(IntegerType()))
    )

    median_attendance = ev.approxQuantile("attendance_count", [0.5], 0.01)[0]
    ev = ev.withColumn(
        "high_turnout",
        (F.col("attendance_count") >= F.lit(median_attendance)).cast(IntegerType()),
    ).withColumn("_median_attendance_used", F.lit(median_attendance))

    return ev.orderBy("event_date")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bronze", required=True, help="bronze TSV path (s3:// or local)")
    ap.add_argument("--silver", required=True, help="silver output Parquet path")
    ap.add_argument("--gold", required=True, help="gold output Parquet path")
    args = ap.parse_args(argv)

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    print(f"[spark] reading bronze: {args.bronze}")
    silver = clean_games(spark, args.bronze)
    n_silver = silver.count()
    print(f"[spark] silver rows: {n_silver:,}")

    silver.write.mode("overwrite").parquet(args.silver)
    print(f"[spark] wrote silver -> {args.silver}")

    gold = build_event_features(spark, silver)
    n_gold = gold.count()
    print(f"[spark] gold rows: {n_gold:,}")

    gold.write.mode("overwrite").parquet(args.gold)
    print(f"[spark] wrote gold -> {args.gold}")

    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
