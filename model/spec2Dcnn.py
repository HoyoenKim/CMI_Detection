from typing import Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torchvision.transforms.functional import resize

from common.cutmix import Cutmix
from common.mixup import Mixup
from model.base import BaseModel

import matplotlib.pyplot as plt

def plot_feature_extractor_output(x, batch_idx=0, channel_idx=0, filename="./feature_extractor_output.png"):
    # Select the specific batch and channel
    data = x[batch_idx, channel_idx, :, :].detach().cpu().numpy()

    plt.imshow(data, aspect='auto')
    plt.title(f"Feature Extractor Output - Batch {batch_idx}, Channel {channel_idx}")
    plt.xlabel("Time")
    plt.ylabel("Features")
    plt.colorbar()
    plt.savefig(filename)  # Save the image
    plt.clf()  # Clear the figure

def plot_encoder_output(x, batch_idx=0, filename="./encoder_output.png"):
    # Select the specific batch
    data = x[batch_idx, :, :].detach().cpu().numpy()

    plt.imshow(data, aspect='auto')
    plt.title(f"Encoder Output - Batch {batch_idx}")
    plt.xlabel("Time")
    plt.ylabel("Encoded Features")
    plt.colorbar()
    plt.savefig(filename)  # Save the image
    plt.clf()  # Clear the figure

def plot_decoder_output(logits, batch_idx=0, filename="./decoder_output.png"):
    # Select the specific batch
    data = logits[batch_idx, :, :].detach().cpu().numpy()

    plt.imshow(data.T, aspect='auto')  # Transpose to have classes on y-axis
    plt.title(f"Decoder Output - Batch {batch_idx}")
    plt.xlabel("Time")
    plt.ylabel("Classes")
    plt.colorbar()
    plt.savefig(filename)  # Save the image
    plt.clf()  # Clear the figure

class Spec2DCNN(BaseModel):
    def __init__(
        self,
        feature_extractor: nn.Module,
        decoder: nn.Module,
        encoder_name: str,
        in_channels: int,
        encoder_weights: Optional[str] = None,
        mixup_alpha: float = 0.5,
        cutmix_alpha: float = 0.5,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.encoder = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
        )
        self.decoder = decoder
        self.mixup = Mixup(mixup_alpha)
        self.cutmix = Cutmix(cutmix_alpha)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def _forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        do_mixup: bool = False,
        do_cutmix: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.feature_extractor(x)  # (batch_size, n_channels, height, n_timesteps)
        plot_feature_extractor_output(x, 0, 0)

        if do_mixup and labels is not None:
            x, labels = self.mixup(x, labels)
        if do_cutmix and labels is not None:
            x, labels = self.cutmix(x, labels)

        x = self.encoder(x).squeeze(1)  # (batch_size, height, n_timesteps)
        plot_encoder_output(x, 0)

        logits = self.decoder(x)  # (batch_size, n_timesteps, n_classes)
        plot_decoder_output(logits, 0)

        if labels is not None:
            return logits, labels
        else:
            return logits

    def _logits_to_proba_per_step(self, logits: torch.Tensor, org_duration: int) -> torch.Tensor:
        preds = logits.sigmoid()
        return resize(preds, size=[org_duration, preds.shape[-1]], antialias=False)[:, :, [1, 2]]

    def _correct_labels(self, labels: torch.Tensor, org_duration: int) -> torch.Tensor:
        return resize(labels, size=[org_duration, labels.shape[-1]], antialias=False)[:, :, [1, 2]]
