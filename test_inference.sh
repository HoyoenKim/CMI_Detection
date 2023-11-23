#!/bin/bash

dataset="seg"
feature_extractor="CNN"
model="Spec2DCNN"
decoder="UNet1DDecoder"

env="local"
num_workers=2
downsample_rate=2
batch_size=32
distance=40
score_th=0.05
feature_args="anglez,enmo,hour_sin,hour_cos"
epochs=50

echo "python3 main.py \"test\" 0 1 $dataset $feature_extractor $model $decoder $env $num_workers $downsample_rate $batch_size $distance $score_th $feature_args $epochs"
python3 main.py "test" 0 1 $dataset $feature_extractor $model $decoder $env $num_workers $downsample_rate $batch_size $distance $score_th $feature_args $epochs