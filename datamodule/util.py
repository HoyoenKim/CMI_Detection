import os

import numpy as np

from common.util import pad_if_needed
from datamodule.seg import SegTestDataset, SegTrainDataset, SegValidDataset
#from datamodule.detr import DETRTestDataset, DETRTrainDataset, DETRValidDataset
#from datamodule.centernet import CenterNetTestDataset, CenterNetTrainDataset, CenterNetValidDataset

def load_features(feature_names, series_ids, processed_dir):
    features = {}
    for series_id in series_ids:
        series_dir = os.path.join(processed_dir, series_id)
        this_feature = []
        for feature_name in feature_names:
            this_feature.append(np.load(os.path.join(series_dir, f"{feature_name}.npy")))
        features[series_id] = np.stack(this_feature, axis=1)

    return features

def load_chunk_features(duration, feature_names, series_ids, processed_dir):
    features = {}
    for series_id in series_ids:
        series_dir = os.path.join(processed_dir, series_id)
        this_feature = []
        for feature_name in feature_names:
            this_feature.append(np.load(os.path.join(series_dir, f"{feature_name}.npy")))
        this_feature = np.stack(this_feature, axis=1)
        num_chunks = (len(this_feature) // duration) + 1
        for i in range(num_chunks):
            chunk_feature = this_feature[i * duration : (i + 1) * duration]
            chunk_feature = pad_if_needed(chunk_feature, duration, pad_value=0)
            features[f"{series_id}_{i:07}"] = chunk_feature

    return features

def get_train_ds(cfg, event_df, features):
    if cfg["dataset"]["name"] == "seg":
        return SegTrainDataset(cfg=cfg, features=features, event_df=event_df)
    else:
        raise ValueError(f"Invalid dataset name: {cfg['dataset']['name']}")

def get_valid_ds(cfg, event_df, chunk_features):
    if cfg["dataset"]["name"] == "seg":
        return SegValidDataset(cfg=cfg, chunk_features=chunk_features, event_df=event_df)
    else:
        raise ValueError(f"Invalid dataset name: {cfg['dataset']['name']}")

def get_test_ds(cfg, chunk_features):
    if cfg["dataset"]["name"] == "seg":
        return SegTestDataset(cfg=cfg, chunk_features=chunk_features)
    else:
        raise ValueError(f"Invalid dataset name: {cfg['dataset']['name']}")
