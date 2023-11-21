import numpy as np
import polars as pl
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from datamodule.util import load_features, load_chunk_features
from datamodule.util import get_train_ds, get_valid_ds

class SleepDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.processed_dir = self.cfg["dir"]["processed_dir"]
        self.event_df = pl.read_csv(self.cfg["dir"]["train_events_path"]).drop_nulls()
        self.train_event_df = self.event_df.filter(
            pl.col("series_id").is_in(self.cfg["split"]["train_series_ids"])
        )
        self.valid_event_df = self.event_df.filter(
            pl.col("series_id").is_in(self.cfg["split"]["valid_series_ids"])
        )

        # train data
        self.train_features = load_features(
            feature_names=self.cfg["features"],
            series_ids=self.cfg["split"]["train_series_ids"],
            processed_dir=self.cfg["dir"]["processed_dir"],
        )

        # valid data
        self.valid_chunk_features = load_chunk_features(
            duration=self.cfg["duration"],
            feature_names=self.cfg["features"],
            series_ids=self.cfg["split"]["valid_series_ids"],
            processed_dir=self.cfg["dir"]["processed_dir"],
        )

    def train_dataloader(self):
        train_dataset = get_train_ds(
            cfg=self.cfg,
            event_df=self.train_event_df,
            features=self.train_features,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg["dataset"]["batch_size"],
            shuffle=True,
            num_workers=self.cfg["dataset"]["num_workers"],
            pin_memory=True,
            drop_last=True,
        )
        return train_loader

    def val_dataloader(self):
        valid_dataset = get_valid_ds(
            cfg=self.cfg,
            event_df=self.valid_event_df,
            chunk_features=self.valid_chunk_features,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.cfg["dataset"]["batch_size"],
            shuffle=False,
            num_workers=self.cfg["dataset"]["num_workers"],
            pin_memory=True,
        )
        return valid_loader
