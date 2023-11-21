import os

import polars as pl
from tqdm import tqdm

from preprocess.const import ANGLEZ_MEAN, ANGLEZ_STD, ENMO_MEAN, ENMO_STD, FEATURE_NAMES  
from preprocess.util import deg_to_rad, add_feature, save_each_series

def preprocess(train_series_path, train_series_save_dir, save = True):
    series_lf = pl.scan_parquet(train_series_path, low_memory=True)
    series_df = (
        series_lf.with_columns(
            pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z"),
            deg_to_rad(pl.col("anglez")).alias("anglez_rad"),
            (pl.col("anglez") - ANGLEZ_MEAN) / ANGLEZ_STD,
            (pl.col("enmo") - ENMO_MEAN) / ENMO_STD,
        )
        .select(
            [
                pl.col("series_id"),
                pl.col("anglez"),
                pl.col("enmo"),
                pl.col("timestamp"),
                pl.col("anglez_rad"),
            ]
        )
        .collect(streaming=True)
        .sort(by=["series_id", "timestamp"])
    )
    n_unique = series_df.get_column("series_id").n_unique()

    if save:
        if not os.path.isdir(train_series_save_dir):
            os.mkdir(train_series_save_dir)
        for series_id, this_series_df in tqdm(series_df.group_by("series_id"), total=n_unique):
            this_series_df = add_feature(this_series_df, FEATURE_NAMES)
            this_series_save_path = os.path.join(train_series_save_dir, series_id)
            if not os.path.isdir(this_series_save_path):
                os.mkdir(this_series_save_path)
            save_each_series(this_series_df, FEATURE_NAMES, this_series_save_path)

