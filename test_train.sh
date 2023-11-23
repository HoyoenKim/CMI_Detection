#!/bin/bash

datasets=("seg" "detr" "centernet")
feature_extractors=("CNN" "PANNs" "LSTM" "Spec")
models=("Spec2DCNN" "Spec1D" "DETR2DCNN" "CenterNet")
decoders=("UNet1DDecoder" "LSTMDecoder" "TransformerDecoder" "MLPDecoder")

env="local"
num_workers=2
downsample_rate=2
batch_size=32
distance=40
score_th=0.05

feature_args="anglez,enmo,hour_sin,hour_cos,anglez_diff,enmo_diff"
epochs=50

# Loop through each combination of configurations
for dataset in "${datasets[@]}"; do
    for feature_extractor in "${feature_extractors[@]}"; do
        for model in "${models[@]}"; do
            for decoder in "${decoders[@]}"; do
                # Call your Python script with the current configuration
                echo "python3 main.py \"train\" 0 1 $dataset $feature_extractor $model $decoder $env $num_workers $downsample_rate $batch_size $distance $score_th $feature_args $epochs"
                python3 main.py "train" 0 1 $dataset $feature_extractor $model $decoder $env $num_workers $downsample_rate $batch_size $distance $score_th $feature_args $epochs
            done
        done
    done
done