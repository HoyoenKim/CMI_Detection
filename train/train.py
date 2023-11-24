import sys
import logging
from pathlib import Path

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    TQDMProgressBar
)
import numpy as np
#from pytorch_lightning.loggers import WandbLogger

from datamodule.datamodule import SleepDataModule
from model.model import PLSleepModel

class MyProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar
    
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"
)
LOGGER = logging.getLogger(Path(__file__).name)

def train(cfg, env):
    seed_everything(cfg["seed"])

    # init lightning model
    datamodule = SleepDataModule(cfg)
    LOGGER.info("Set Up DataModule")
    model = PLSleepModel(cfg, datamodule.valid_event_df, len(cfg["features"]), len(cfg["labels"]), cfg["duration"])

    # set callbacks
    checkpoint_cb = ModelCheckpoint(
        verbose=True,
        monitor=cfg["trainer"]["monitor"],
        mode=cfg["trainer"]["monitor_mode"],
        save_top_k=1,
        save_last=False,
    )
    lr_monitor = LearningRateMonitor("epoch")
    if env == "colab":
        progress_bar = MyProgressBar()
    else:
        progress_bar = RichProgressBar()
    model_summary = RichModelSummary(max_depth=2)

    # init experiment logger
    #pl_logger = WandbLogger(
    #    name=cfg["exp_name"],
    #    project="child-mind-institute-detect-sleep-states",
    #)
    #pl_logger.log_hyperparams(cfg)

    trainer = Trainer(
        # env
        default_root_dir=Path.cwd(),
        # num_nodes=cfg["trainer"]["num_gpus"],
        accelerator=cfg["trainer"]["accelerator"],
        precision=16 if cfg["trainer"]["use_amp"] else 32,
        # training
        fast_dev_run=cfg["trainer"]["debug"],  # run only 1 train batch and 1 val batch
        max_epochs=cfg["trainer"]["epochs"],
        max_steps=cfg["trainer"]["epochs"] * len(datamodule.train_dataloader()),
        gradient_clip_val=cfg["trainer"]["gradient_clip_val"],
        accumulate_grad_batches=cfg["trainer"]["accumulate_grad_batches"],
        callbacks=[checkpoint_cb, lr_monitor, progress_bar, model_summary],
        #logger=pl_logger,
        # resume_from_checkpoint=resume_from,
        num_sanity_val_steps=0,
        log_every_n_steps=int(len(datamodule.train_dataloader()) * 0.1),
        sync_batchnorm=True,
        check_val_every_n_epoch=cfg["trainer"]["check_val_every_n_epoch"],
    )

    trainer.fit(model, datamodule=datamodule)

    # load best weights
    model = PLSleepModel.load_from_checkpoint(
        checkpoint_cb.best_model_path,
        cfg=cfg,
        val_event_df=datamodule.valid_event_df,
        feature_dim=len(cfg["features"]),
        num_classes=len(cfg["labels"]),
        duration=cfg["duration"],
    )
    weights_path = cfg["dir"]["model_path"]  # type: ignore
    LOGGER.info(f"Extracting and saving best weights: {weights_path}")
    torch.save(model.model.state_dict(), weights_path)


def do_train(model, dataloader, optimizer, criterion, device, cfg):
    model.train()
    total_loss = 0
    for batch in dataloader:
        inputs, labels = batch['feature'].to(device), batch['label'].to(device)

        optimizer.zero_grad()
        do_mixup = np.random.rand() < cfg["aug"]["mixup_prob"]
        do_cutmix = np.random.rand() < cfg["aug"]["cutmix_prob"]
        outputs = model(inputs, labels, do_mixup, do_cutmix)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def do_validate(model, dataloader, criterion, device, cfg):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch['feature'].to(device), batch['label'].to(device)
            do_mixup = np.random.rand() < cfg["aug"]["mixup_prob"]
            do_cutmix = np.random.rand() < cfg["aug"]["cutmix_prob"]
            outputs = model(inputs, labels, do_mixup, do_cutmix)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train2(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 모듈 초기화
    datamodule = SleepDataModule(cfg)

    # 모델 초기화
    model = PLSleepModel(cfg, datamodule.valid_event_df, len(cfg["features"]), len(cfg["labels"]), cfg["duration"]).to(device)

    # 옵티마이저 및 손실 함수 설정
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=cfg["optimizer"]["lr"])
    criterion = model.model.loss_fn
    best_val_loss = float('inf')
    for epoch in range(cfg["trainer"]['epochs']):
        train_loss = do_train(model, datamodule.train_dataloader(), optimizer, criterion, device, cfg)
        val_loss = do_validate(model, datamodule.val_dataloader(), criterion, device, cfg)

        print(f"Epoch {epoch}, Train Loss: {train_loss}, Validation Loss: {val_loss}")

        # 검증 손실이 개선되었는지 확인하고 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            weights_path = cfg["dir"]["model_path"]
            torch.save(model.model.state_dict(), weights_path)
