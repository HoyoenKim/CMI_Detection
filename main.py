import sys

from preprocess.preprocess import preprocess
from train.train import train, train2
from inference.inference import inference
from config import gen_config

import torch

if __name__ == "__main__":
    do_preprocess = False
    do_train = False
    do_test_preprocess = False
    do_inference = True

    try:
        phase = sys.argv[1] # train, test
        
        do_preprocess = sys.argv[2] # 0: undo, 1: do
        do_main = sys.argv[3] # 0: undo, 1: do train or do test

        dataset = sys.argv[4] # seg, detr, centernet
        feature_extractor = sys.argv[5] # CNN, PANNs, LSTM, Spec
        model = sys.argv[6] # Spec2DCNN, Spec1D, DETR2DCNN, CenterNet
        decoder = sys.argv[7] # UNet1DDecoder, LSTMDecoder, TransformerDecoder, MLPDecoder
        
        env = sys.argv[8]
        num_workers = int(sys.argv[9])
        downsample_rate = int(sys.argv[10])
        batch_size = int(sys.argv[11])

        distance = int(sys.argv[12])
        score_th = float(sys.argv[13])
        postprocess_config = {
            "score_th": score_th,
            "distance": distance,
        }
        print(f"Post-process: distance with {distance} and score_th with {score_th}")
        feature_arg = sys.argv[14]
        features = feature_arg.split(',')
        print("Features: ", features)
        epochs = int(sys.argv[15])

        if len(sys.argv) == 16:
            series_id_new = 0
        else:
            series_id_new = 1
    except:
        print(sys.argv)
        print("Please Type the boolean values")
    
    config = gen_config(phase, env, dataset, feature_extractor, model, decoder, postprocess_config, feature_arg, num_workers, downsample_rate, batch_size, epochs, series_id_new)
    if phase == "train":
        if do_preprocess == "1":
            # preprocess 
            preprocess(config["dir"]["train_series_path"], config["dir"]["processed_dir"], phase, env)
        if do_main == "1":
            # train
            if env == "colab":
                torch.set_float32_matmul_precision('high')
            
            if model == "DETR2DCNN" or model == "CenterNet":
                print("Other train fn")
                train2(config)
            else:
                train(config, env)
    elif phase == "test": 
        if do_preprocess == "1":
            # preprocess 
            preprocess(config["dir"]["test_series_path"], config["dir"]["processed_dir"], phase, env)    
        if do_main == "1":
            # inference
            inference(config, config["dir"]["model_path"], config["dir"]["submission_path"])