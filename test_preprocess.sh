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

echo "python3 main.py \"train\" 1 0 $dataset $feature_extractor $model $decoder $env $num_workers $downsample_rate $batch_size $distance $score_th"
#python3 main.py "train" 1 0 $dataset $feature_extractor $model $decoder $env $num_workers $downsample_rate $batch_size $distance $score_th

echo "python3 main.py \"test\" 1 0 $dataset $feature_extractor $model $decoder $env $num_workers $downsample_rate $batch_size $distance $score_th"
python3 main.py "test" 1 0 $dataset $feature_extractor $model $decoder $env $num_workers $downsample_rate $batch_size $distance $score_th