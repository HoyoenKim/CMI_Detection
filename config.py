from const import train_series_ids, valid_series_ids, new_train_series_ids, new_valid_series_ids

def gen_config(phase, env, dataset, feature_extractor, model, decoder, postprocess_config, feature_arg, num_workers = 2, downsample_rate = 2, batch_size = 32, epochs = 50, series_ids = 0):
    # Dataset
    seg = {
      "name": "seg",
      "batch_size": batch_size,
      "num_workers": num_workers,
      "offset": 10,
      "sigma": 10,
      "bg_sampling_rate": 0.5
    }
    detr = {
      "name": "detr",
      "batch_size": batch_size,
      "num_workers": num_workers,
      "offset": 10,
      "sigma": 10,
      "bg_sampling_rate": 0.5
    }
    centernet = {
      "name": "centernet",
      "batch_size": batch_size,
      "num_workers": num_workers,
      "offset": 10,
      "sigma": 10,
      "bg_sampling_rate": 0.5
    }
    # Feature Extractor
    CNN = {
      "name": "CNNSpectrogram",
      "params": {
        "base_filters": 64,
        "kernel_sizes": [
          32,
          16,
          downsample_rate
        ],
        "stride": downsample_rate,
        "sigmoid": True,
        "reinit": True
      }
    }
    PANNs = {
      "name": "PANNsFeatureExtractor",
      "params": {
        "base_filters": 64,
        "kernel_sizes": [
          32,
          16,
          downsample_rate
        ],
        "stride": downsample_rate,
        "sigmoid": True,
        "reinit": True,
        "win_length": None
      }
    }
    LSTM = {
      "name": "LSTMFeatureExtractor",
      "params": {
        "hidden_size": 64,
        "num_layers": 2,
        "bidirectional": True,
        "stride": downsample_rate
      }
    }
    Spec = {
      "name": "SpecFeatureExtractor",
      "params": {
        "height": 64,
        "hop_length": downsample_rate,
        "win_length": None
      }
    }
    # Model
    Spec2DCNN = {
      "name": "Spec2DCNN",
      "params": {
        "encoder_name": "resnet34",
        "encoder_weights": "imagenet"
      }
    }
    Spec1D = {
      "name": "Spec1D",
      "params": None
    }
    DETR2DCNN = {
      "name": "DETR2DCNN",
      "params": {
        "encoder_name": "resnet34",
        "encoder_weights": "imagenet",
        "max_det": 15,
        "hidden_dim": 256,
        "nheads": 8,
        "num_encoder_layers": 4,
        "num_decoder_layers": 4
      }
    }
    CenterNet = {
      "name": "CenterNet",
      "params": {
        "encoder_name": "resnet34",
        "encoder_weights": "imagenet",
        "keypoint_weight": 1,
        "offset_weight": 1,
        "bbox_size_weight": 1
      }
    }
    TransformerAutoModel = {
      "name": "TransformerAutoModel",
      "params": {
        "model_name": "microsoft/deberta-v3-small",
        "hidden_size": 120,
        "stride": downsample_rate,
      }
    }
    # Decoder
    UNet1DDecoder = {
      "name": "UNet1DDecoder",
      "params": {
        "bilinear": False,
        "se": False,
        "res": False,
        "scale_factor": 2,
        "dropout": 0.2
      }
    }
    LSTMDecoder = {
      "name": "LSTMDecoder",
      "params": {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "bidirectional": True
      }
    }
    TransformerDecoder = {
      "name": "TransformerDecoder",
      "params": {
        "hidden_size": 256,
        "num_layers": 4,
        "nhead": 4,
        "dropout": 0.2
      }
    }
    TransformerCNNDecoder = {
      "name": "TransformerCNNDecoder",
      "params": {
        "hidden_size": 256,
        "num_layers": 4,
        "nhead": 4,
        "dropout": 0.2
      }
    }
    MLPDecoder = {
      "name": "MLPDecoder",
      "params": {
          "hidden_size": 128,
          "num_layers": 3,
      }
    }
    # Common
    if(series_ids == 1):
      print("use new series ids")
      split_config = {
          "train_series_ids": new_train_series_ids,
          "valid_series_ids": new_valid_series_ids
      } 
    else:
      print("use old series ids")
      split_config = {
          "train_series_ids": train_series_ids,
          "valid_series_ids": valid_series_ids
      } 
    
    aug_config = {
        "mixup_prob": 0.0,
        "mixup_alpha": 0.4,
        "cutmix_prob": 0.0,
        "cutmix_alpha": 0.4,
    }
    optimizer_config = {
        "lr": 0.0005,
    }
    scheduler_config = {
        "num_warmup_steps": 0,
    }
    trainer_config = {
        "epochs": epochs,
        "accelerator": "auto",
        "use_amp": True,
        "debug": False,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 1,
        "monitor": "val_loss",
        "monitor_mode": "min",
        "check_val_every_n_epoch": 1,
    }
    labels = [
        "awake",
        "event_onset",
        "event_wakeup",
    ]

    decoder_config = None
    train_dir_config = None
    test_dir_config = None
    model_name_postfix = f"{dataset}_{feature_extractor}_{model}_{decoder}_{feature_arg}"
    if env == "local" or env == "colab":
        train_dir_config = {
            "processed_dir": './data',
            "train_events_path": './raw_data/train_events.csv',
            "train_series_path": "./raw_data/train_series.parquet",
            "test_series_path": "./raw_data/test_series.parquet",
            "model_path": f"./best_model_{model_name_postfix}.pth",
            "submission_path": "./submission.csv",
        }
        test_dir_config = {
            "processed_dir": './data_test',
            "train_events_path": './raw_data/train_events.csv',
            "train_series_path": "./raw_data/train_series.parquet",
            "test_series_path": "./raw_data/test_series.parquet",
            "model_path": "./best_model.pth",
            "submission_path": "./submission.csv",
        }
    elif env == "kaggle":
        train_dir_config = {
            "processed_dir": "/kaggle/working/output",
            "train_events_path": "/kaggle/input/child-mind-institute-detect-sleep-states/train_events.csv",
            "train_series_path": "/kaggle/input/child-mind-institute-detect-sleep-states/train_series.parquet",
            "test_series_path": "/kaggle/input/child-mind-institute-detect-sleep-states/test_series.parquet",
            "model_path": f"/kaggle/working/best_model_{model_name_postfix}.pth",
            "submission_path": "/kaggle/working/submission.csv",
        }
        test_dir_config = {
            "processed_dir": "/kaggle/working/output_test",
            "train_events_path": "/kaggle/input/child-mind-institute-detect-sleep-states/train_events.csv",
            "train_series_path": "/kaggle/input/child-mind-institute-detect-sleep-states/train_series.parquet",
            "test_series_path": "/kaggle/input/child-mind-institute-detect-sleep-states/test_series.parquet",
            "model_path": "/kaggle/input/cmi-best-model/best_model.pth",
            "submission_path": "/kaggle/working/submission.csv",
        }

    dataset_config = None
    if dataset == "seg":
        dataset_config = seg
    elif dataset == "detr":
        dataset_config = detr
    elif dataset == "centernet":
        dataset_config = centernet

    feature_extractor_config = None
    if feature_extractor == "CNN":
        feature_extractor_config = CNN
    elif feature_extractor == "PANNs":
        feature_extractor_config = PANNs
    elif feature_extractor == "LSTM":
        feature_extractor_config = LSTM
    elif feature_extractor == "Spec":
        feature_extractor_config = Spec

    model_config = None
    if model == "Spec2DCNN":
        model_config = Spec2DCNN
    elif model == "Spec1D":
        model_config = Spec1D
    elif model == "DETR2DCNN":
        model_config = DETR2DCNN
    elif model == "CenterNet":
        model_config = CenterNet
    elif model == "TransformerAutoModel":
        model_config = TransformerAutoModel
    
    decoder_config = None
    if decoder == "UNet1DDecoder":
        decoder_config = UNet1DDecoder
    elif decoder == "LSTMDecoder":
        decoder_config = LSTMDecoder
    elif decoder == "TransformerDecoder":
        decoder_config = TransformerDecoder
    elif decoder == "MLPDecoder":
        decoder_config = MLPDecoder
    elif decoder == "TransformerCNNDecoder":
        decoder_config = TransformerCNNDecoder

    features = feature_arg.split(',')
    if phase == "train":
        train_config = {
            "seed": 42,
            "dir": train_dir_config,
            "dataset": dataset_config,
            "feature_extractor": feature_extractor_config,
            "model": model_config,
            "decoder": decoder_config,

            "duration": 5760,
            "downsample_rate": downsample_rate,
            "upsample_rate": 1,

            "split": split_config,
            "features": features,
            "labels": labels,

            "aug": aug_config,
            "optimizer": optimizer_config,
            "scheduler": scheduler_config,
            "trainer": trainer_config,

            "exp_name": "exp00",
            "pp": postprocess_config,
        }
        return train_config
    elif phase == "test":
        inference_config = {
            "seed": 42,
            "dir": test_dir_config,
            "dataset": dataset_config,
            "feature_extractor": feature_extractor_config,
            "model": model_config,
            "decoder": decoder_config,

            "duration": 5760,
            "downsample_rate": downsample_rate,
            "upsample_rate": 1,

            "features": features,
            "labels": labels,

            "aug": aug_config,
            "batch_size": batch_size,
            "num_workers": num_workers,

            "pp": postprocess_config,
            "use_amp": True,
        }
        return inference_config