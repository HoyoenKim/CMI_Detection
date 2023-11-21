import os

import numpy as np
import polars as pl

def deg_to_rad(x):
    return np.pi / 180 * x

def to_coord(x, MAX, name):
    rad = 2 * np.pi * (x % MAX) / MAX
    x_sin = rad.sin()
    x_cos = rad.cos()

    return [x_sin.alias(f"{name}_sin"), x_cos.alias(f"{name}_cos")]

def add_feature(series_df, FEATURE_NAMES):
    series_df = (
        series_df.with_row_count("step")
        .with_columns(
            *to_coord(pl.col("timestamp").dt.hour(), 24, "hour"),
            *to_coord(pl.col("timestamp").dt.month(), 12, "month"),
            *to_coord(pl.col("timestamp").dt.minute(), 60, "minute"),
            pl.col("step") / pl.count("step"),
            pl.col('anglez_rad').sin().alias('anglez_sin'),
            pl.col('anglez_rad').cos().alias('anglez_cos'),
        )
        .select("series_id", *FEATURE_NAMES)
    )
    return series_df

def save_each_series(this_series_df, FEATURE_NAMES, this_series_save_path):
    for col_name in FEATURE_NAMES:
        x = this_series_df.get_column(col_name).to_numpy(zero_copy_only=True)
        np.save(os.path.join(this_series_save_path, f"{col_name}.npy"), x)