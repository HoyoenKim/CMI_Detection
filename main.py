import sys

from preprocess.preprocess import preprocess
from train.train import train
from inference.inference import inference
from const import train_series_ids, valid_series_ids

train_dir_config = {
    "processed_dir": './data',
    "train_events_path": './raw_data/train_events.csv'
}
split_config = {
    "train_series_ids": train_series_ids,
    "valid_series_ids": valid_series_ids
}
dataset_config = {
    "name": "seg",
    "batch_size": 32,
    "num_workers": 24,
    "offset": 10,
    "sigma": 10,
    "bg_sampling_rate": 0.5
}
aug_config = {
    "mixup_prob": 0.0,
    "mixup_alpha": 0.4,
    "cutmix_prob": 0.0,
    "cutmix_alpha": 0.4,
}
postprocess_config = {
    "score_th": 0.005,
    "distance": 100,
}
optimizer_config = {
    "lr": 0.0005,
}
scheduler_config = {
    "num_warmup_steps": 0,
}
model_config = {
    "name": "Spec2DCNN",
    "params": {
        "encoder_name": "resnet34",
        "encoder_weights": "imagenet",
    }
}
feature_extractor_config = {
    "name": "CNNSpectrogram",
    "params": {
        "base_filters": 64,
        "kernel_sizes": [
            32,
            16,
            2, #down_sample rate 
        ],
        "stride": 2, #down_sample rate 
        "sigmoid": True,
        "reinit": True,
    }
}
decoder_config = {
    "name": "UNet1DDecoder",
    "params": {
        "bilinear": False,
        "se": False,
        "res": False,
        "scale_factor": 2,
        "dropout": 0.2,
    }
}
trainer_config = {
    "epochs": 50,
    "accelerator": "auto",
    "use_amp": True,
    "debug": False,
    "gradient_clip_val": 1.0,
    "accumulate_grad_batches": 1,
    "monitor": "val_loss",
    "monitor_mode": "min",
    "check_val_every_n_epoch": 1,
}
features = [
    "anglez",
    "enmo",
    #"step",
    #"month_sin",
    #"month_cos",
    "hour_sin",
    "hour_cos",
    #"minute_sin",
    #"minute_cos",
    #"anglez_sin",
    #"anglez_cos",
]
labels = [
    "awake",
    "event_onset",
    "event_wakeup",
]
train_config = {
    "seed": 42,
    "dir": train_dir_config,
    "split": split_config,
    "features": features,
    "labels": labels,
    "duration": 5760,
    "downsample_rate": 2,
    "upsample_rate": 1,
    "dataset": dataset_config,
    "aug": aug_config,
    "pp": postprocess_config,
    "optimizer": optimizer_config,
    "scheduler": scheduler_config,
    "model": model_config,
    "feature_extractor": feature_extractor_config,
    "decoder": decoder_config,
    "exp_name": "exp00",
    "trainer": trainer_config,
}

test_dir_config = {
    "processed_dir": './data_test',
}
inference_config = {
    "seed": 42,
    "dir": test_dir_config,
    "features": features,
    "labels": labels,
    "batch_size": 32,
    "num_workers": 8,
    "duration": 5760,
    "downsample_rate": 2,
    "upsample_rate": 1,
    "dataset": dataset_config,
    "aug": aug_config,
    "pp": postprocess_config,
    "model": model_config,
    "feature_extractor": feature_extractor_config,
    "decoder": decoder_config,
    "use_amp": True,
}

train_series_path = "./raw_data/train_series.parquet"
train_series_save_dir = "./data"
test_series_path = "./raw_data/test_series.parquet"
test_series_save_dir = "./data_test"
model_path = "./best_model.pth"
submission_path = "./submission.csv"

if __name__ == "__main__":
    do_preprocess = False
    do_train = False
    do_test_preprocess = False
    do_inference = True

    try:
        do_preprocess = sys.argv[1]
        do_train = sys.argv[2]
        do_test_preprocess = sys.argv[3]
        do_inference = sys.argv[4]
        env = sys.argv[5]
        if env == "kaggle":
            train_series_path = "/kaggle/input/child-mind-institute-detect-sleep-states/train_series.parquet"
            train_series_save_dir = "/kaggle/working/output"
            test_series_path = "/kaggle/input/child-mind-institute-detect-sleep-states/test_series.parquet"
            test_series_save_dir = "/kaggle/working/output_test"
            model_path = "/kaggle/input/cmi-best-model/best_model.pth"
            submission_path = "/kaggle/working/submission.csv"
            train_config["dir"] = {
                "processed_dir": train_series_save_dir,
                "train_events_path": train_series_path
            }
            inference_config["dir"] = {
                "processed_dir": test_series_save_dir,
            }
            
    except:
        print("Please Type the boolean values")

    # preprocess    
    if do_preprocess == "1":
        preprocess(train_series_path, train_series_save_dir)
    
    # train
    if do_train == "1":
        train(train_config)

    # inference
    if do_test_preprocess == "1":
        preprocess(test_series_path, test_series_save_dir)    
    if do_inference == "1":
        inference(inference_config, model_path, submission_path)

    
    