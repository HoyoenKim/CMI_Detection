import os

import polars as pl
from tqdm import tqdm

from preprocess.util import deg_to_rad, add_feature, save_each_series
from preprocess.const import FEATURE_NAMES

ANGLEZ_MEAN = -8.810476
ANGLEZ_STD = 35.521877
ENMO_MEAN = 0.041315
ENMO_STD = 0.101829
ANGLEZ_DIFF_MEAN =  -2.669245e-07
ANGLEZ_DIFF_STD = 11.409339
ENMO_DIFF_MEAN = -9.252390e-12
ENMO_DIFF_STD = 0.083003

def cal_const(series_lf):
    global ANGLEZ_MEAN
    global ANGLEZ_STD
    global ENMO_MEAN
    global ENMO_STD
    global ANGLEZ_DIFF_MEAN
    global ANGLEZ_DIFF_STD
    global ENMO_DIFF_MEAN
    global ENMO_DIFF_STD

    # Calculate the differences
    series_df = series_lf.with_columns([
        pl.col('anglez'),
        pl.col('enmo'),
        ((pl.col('anglez') - pl.col('anglez').shift(1)).fill_nan(0)).abs().alias('anglez_diff'),
        ((pl.col('enmo') - pl.col('enmo').shift(1)).fill_nan(0)).abs().alias('enmo_diff')
    ]).collect(streaming=True)

    # Calculate mean and standard deviation for the differences
    angle_stats = series_df.select([
        pl.mean('anglez').alias('anglez_mean'),
        pl.std('anglez').alias('anglez_std')
    ]).to_dict()
    enmo_stats = series_df.select([
        pl.mean('enmo').alias('enmo_mean'),
        pl.std('enmo').alias('enmo_std')
    ]).to_dict()
    angle_diff_stats = series_df.select([
        pl.mean('anglez_diff').alias('anglez_diff_mean'),
        pl.std('anglez_diff').alias('anglez_diff_std')
    ]).to_dict()
    enmo_diff_stats = series_df.select([
        pl.mean('enmo_diff').alias('enmo_diff_mean'),
        pl.std('enmo_diff').alias('enmo_diff_std')
    ]).to_dict()

    # Extract the values
    ANGLEZ_MEAN = angle_stats['anglez_mean'][0]
    ANGLEZ_STD  = angle_stats['anglez_std'][0]
    ENMO_MEAN  =  enmo_stats['enmo_mean'][0]
    ENMO_STD   =  enmo_stats['enmo_std'][0]
    ANGLEZ_DIFF_MEAN = angle_diff_stats['anglez_diff_mean'][0]
    ANGLEZ_DIFF_STD = angle_diff_stats['anglez_diff_std'][0]
    ENMO_DIFF_MEAN = enmo_diff_stats['enmo_diff_mean'][0]
    ENMO_DIFF_STD = enmo_diff_stats['enmo_diff_std'][0]

    # Print the calculated values
    print("ANGLEZ_MEAN:", ANGLEZ_MEAN)
    print("ANGLEZ_STD:",  ANGLEZ_STD)
    print("ENMO_MEAN:",  ENMO_MEAN)
    print("ENMO_STD:",   ENMO_STD)
    print("ANGLEZ_DIFF_MEAN:", ANGLEZ_DIFF_MEAN)
    print("ANGLEZ_DIFF_STD:", ANGLEZ_DIFF_STD)
    print("ENMO_DIFF_MEAN:", ENMO_DIFF_MEAN)
    print("ENMO_DIFF_STD:", ENMO_DIFF_STD)


def preprocess(train_series_path, train_series_save_dir, phase, env, save = True):
    series_lf = pl.scan_parquet(train_series_path, low_memory=True)
    if env == "colab":
        series_df = (
            series_lf.with_columns(
                pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%z", utc=True),
                deg_to_rad(pl.col("anglez")).alias("anglez_rad"),
                pl.col("anglez"),
                pl.col("enmo"),
                (pl.col('anglez') - pl.col('anglez').shift(1) ).alias('anglez_diff'),
                (pl.col('enmo') - pl.col('enmo').shift(1)).alias('enmo_diff')
            )
            .select(
                [
                    pl.col("series_id"),
                    pl.col("anglez"),
                    pl.col("enmo"),
                    pl.col("timestamp"),
                    pl.col("anglez_rad"),
                    pl.col("anglez_diff"),
                    pl.col("enmo_diff"),
                ]
            )
            .collect(streaming=True)
            .sort(by=["series_id", "timestamp"])
        )
    else:
        series_df = (
            series_lf.with_columns(
                pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z"),
                deg_to_rad(pl.col("anglez")).alias("anglez_rad"),
                pl.col("anglez"),
                pl.col("enmo"),
                (pl.col('anglez') - pl.col('anglez').shift(1) ).alias('anglez_diff'),
                (pl.col('enmo') - pl.col('enmo').shift(1)).alias('enmo_diff')
            )
            .select(
                [
                    pl.col("series_id"),
                    pl.col("anglez"),
                    pl.col("enmo"),
                    pl.col("timestamp"),
                    pl.col("anglez_rad"),
                    pl.col("anglez_diff"),
                    pl.col("enmo_diff"),
                ]
            )
            .collect(streaming=True)
            .sort(by=["series_id", "timestamp"])
        )
    n_unique = series_df.get_column("series_id").n_unique()

    if save:
        if not os.path.isdir(train_series_save_dir):
            os.mkdir(train_series_save_dir)
        if env == "colab":
            for series_id, this_series_df in tqdm(series_df.groupby("series_id"), total=n_unique):
                this_series_df = add_feature(this_series_df, FEATURE_NAMES)
                this_series_save_path = os.path.join(train_series_save_dir, series_id)
                if not os.path.isdir(this_series_save_path):
                    os.mkdir(this_series_save_path)
                save_each_series(this_series_df, FEATURE_NAMES, this_series_save_path)
        else:
            for series_id, this_series_df in tqdm(series_df.group_by("series_id"), total=n_unique):
                this_series_df = add_feature(this_series_df, FEATURE_NAMES)
                this_series_save_path = os.path.join(train_series_save_dir, series_id)
                if not os.path.isdir(this_series_save_path):
                    os.mkdir(this_series_save_path)
                save_each_series(this_series_df, FEATURE_NAMES, this_series_save_path)

