from typing import Union

import torch.nn as nn

from model.feature_extractor.cnn import CNNSpectrogram
#from model.feature_extractor.panns import PANNsFeatureExtractor
#from model.feature_extractor.lstm import LSTMFeatureExtractor
#from model.feature_extractor.spectrogram import SpecFeatureExtractor

from model.decoder.unet1ddecoder import UNet1DDecoder
#from model.decoder.lstmdecoder import LSTMDecoder
#from model.decoder.transformerdecoder import TransformerDecoder
#from model.decoder.mlpdecoder import MLPDecoder

from model.spec2Dcnn import Spec2DCNN
#from model.spec1D import Spec1D
#from model.detr2D import DETR2DCNN
#from model.centernet import CenterNet

def get_feature_extractor(cfg, feature_dim, num_timesteps):
    feature_extractor = None
    if cfg["name"] == "CNNSpectrogram":
        feature_extractor = CNNSpectrogram(
            in_channels=feature_dim, output_size=num_timesteps, **cfg["params"]
        )
    else:
        raise ValueError(f"Invalid feature extractor name: {cfg['name']}")

    return feature_extractor

def get_decoder(cfg, n_channels, n_classes, num_timesteps):
    decoder = None
    if cfg["name"] == "UNet1DDecoder":
        decoder = UNet1DDecoder(
            n_channels=n_channels,
            n_classes=n_classes,
            duration=num_timesteps,
            **cfg["params"],
        )
    else:
        raise ValueError(f"Invalid decoder name: {cfg['name']}")

    return decoder

def get_model(cfg, feature_dim, n_classes, num_timesteps, test = False):
    model = None
    if cfg["model"]["name"] == "Spec2DCNN":
        feature_extractor = get_feature_extractor(
            cfg["feature_extractor"], feature_dim, num_timesteps
        )
        decoder = get_decoder(cfg["decoder"], feature_extractor.height, n_classes, num_timesteps)
        if test:
            cfg["model"]["params"]["encoder_weights"] = None
        model = Spec2DCNN(
            feature_extractor=feature_extractor,
            decoder=decoder,
            in_channels=feature_extractor.out_chans,
            mixup_alpha=cfg["aug"]["mixup_alpha"],
            cutmix_alpha=cfg["aug"]["cutmix_alpha"],
            **cfg["model"]["params"],
        )
    else:
        raise ValueError(f"Invalid model name: {cfg['params']['name']}")

    return model
