
# @Author : Yong Zheng


import argparse
import time
import torch
import pandas as pd
import multiprocessing as mcpu
from deepcarskit.quick_start import run
from logging import getLogger
from itertools import product

# Experiment Setup
# mlflow.set_tracking_uri("../mlflow")
# mlflow.set_experiment("DeepCarsKit - TripAdvisor")


if __name__ == '__main__':
    print('GPU availability: ', torch.cuda.is_available())

    n_gpu = torch.cuda.device_count()
    print('Num of GPU: ', n_gpu)

    if n_gpu>0:
        print(torch.cuda.get_device_name(0))
        print('Current GPU index: ', torch.cuda.current_device())

    logger = getLogger()
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', type=str, default='config.yaml', help='config files')

    args, _ = parser.parse_known_args()

    config_list = args.config_files.strip().split(' ') if args.config_files else None

    learning_rates = [10**(-i) for i in range(3, 6)]
    learners = ["adam", "RMSprop"]
    models = ["NeuCMFii", "NeuCMFww", "NeuCMF0i", "NeuCMFi0", "NeuCMF0w", "NeuCMFw0"]
    embedding_sizes = [32]
    weight_decays = [0.0]
    train_batch_sizes = [500]

    hyper = pd.DataFrame(columns=['learning_rate', 'learner', 'model', 'embedding_size', 'weight_decay', 'train_batch_size', 'mae'])

    prod = product(learning_rates, learners, models, embedding_sizes, weight_decays, train_batch_sizes)
    for learning_rate, learner, model, embedding_size, weight_decay, train_batch_size in prod:
        custom_config_dict = {
            "learning_rate": learning_rate,
            "learner": learner,
            "model": model,
            "embedding_size": embedding_size,
            "weight_decay": weight_decay,
            "train_batch_size": train_batch_size
        }

        metrics = run(config_file_list=config_list)
        metrics["best_valid_result"]

        hyper = hyper.append([learning_rate, learner, model, embedding_size, weight_decay, train_batch_size, metrics["best_valid_result"]])

    print(hyper.head())
    hyper.to_csv('hyper.csv', index=False)

    t1 = time.time()
    total = t1 - t0
    logger.info('time cost: '+ f': {total}s')