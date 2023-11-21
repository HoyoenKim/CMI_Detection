import numpy as np
import polars as pl
import torch
import os

from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

from datamodule.util  import load_chunk_features, get_test_ds
from model.base import BaseModel
from model.common import get_model
from common.util import nearest_valid_size, trace
from common.postprocess import post_process_for_seg

def load_model(cfg, device):
    num_timesteps = nearest_valid_size(int(cfg["duration"] * cfg["upsample_rate"]), cfg["downsample_rate"])
    model = get_model(cfg, feature_dim=len(cfg["features"]), n_classes=len(cfg["labels"]), num_timesteps=num_timesteps // cfg["downsample_rate"], test=True)

    # load weights
    weight_path = "./best_model.pth"
    model.load_state_dict(torch.load(weight_path, map_location=device))
    print('load weight from "{}"'.format(weight_path))

    return model

def get_test_dataloader(cfg):
    """get test dataloader

    Args:
        cfg (DictConfig): config

    Returns:
        DataLoader: test dataloader
    """
    feature_dir = cfg["dir"]["processed_dir"]
    series_ids = [x for x in os.listdir(feature_dir)]
    chunk_features = load_chunk_features(
        duration=cfg["duration"],
        feature_names=cfg["features"],
        series_ids=series_ids,
        processed_dir=cfg["dir"]["processed_dir"],
    )
    test_dataset = get_test_ds(cfg, chunk_features=chunk_features)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=False,
    )
    return test_dataloader

def do_inference(duration, loader, model, device, use_amp):
    model = model.to(device)
    model.eval()

    preds = []
    keys = []
    for batch in tqdm(loader, desc="inference"):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                x = batch["feature"].to(device)
                output = model.predict(
                    x,
                    org_duration=duration,
                )
            if output.preds is None:
                raise ValueError("output.preds is None")
            else:
                key = batch["key"]
                preds.append(output.preds.detach().cpu().numpy())
                keys.extend(key)

    preds = np.concatenate(preds)

    return keys, preds

def inference(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_everything(cfg["seed"])
    test_dataloader = get_test_dataloader(cfg)
    model = load_model(cfg, device)
    keys, preds = do_inference(cfg["duration"], test_dataloader, model, device, use_amp=cfg["use_amp"])
    print('f')
    sub_df = post_process_for_seg(keys, preds, score_th=cfg["pp"]["score_th"], distance=cfg["pp"]["distance"])
    print('g')
    sub_df.write_csv("./submission.csv")