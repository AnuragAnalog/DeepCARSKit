
# @Author : Yong Zheng

import os
import time
import torch
import optuna
import argparse
import pandas as pd
import multiprocessing as mcpu

from itertools import product
from logging import getLogger
from deepcarskit.quick_start import run

def remove_all_files(dir):
    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

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

    dataset_type = 'frappe'
    model_type = 'neucmf'

    filename = f"./hypers/version2/{dataset_type}_hyper_{model_type}_optuna.csv"
    
    def objective(trial):
        hyper = pd.read_csv(filename)
        learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-1)
        learner = trial.suggest_categorical('learner', ['adam', 'RMSprop'])
        if model_type == 'neucmf':
            model = trial.suggest_categorical('model', ['NeuCMFii', 'NeuCMFww', 'NeuCMFi0', 'NeuCMF0i', 'NeuCMFw0', 'NeuCMF0w'])
        elif model_type == 'fms':
            model = trial.suggest_categorical('model', ['FM', 'DeepFM'])
        embedding_size = trial.suggest_int('embedding_size', 32, 512)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-1)
        train_batch_size = trial.suggest_int('train_batch_size', 500, 1000, step=50)
        epoch = trial.suggest_int('epoch', 10, 30)

        custom_config_dict = {
            "learning_rate": learning_rate,
            "learner": learner,
            "model": model,
            "embedding_size": embedding_size,
            "weight_decay": weight_decay,
            "train_batch_size": train_batch_size,
            "epochs": epoch
        }

        metrics = run(config_file_list=config_list, custom_config_dict=custom_config_dict)
        hyper = pd.concat([hyper, pd.DataFrame(
            [[learning_rate, learner, model, embedding_size, weight_decay, train_batch_size, epoch, metrics['best_valid_result']['mae'], metrics['best_valid_result']['rmse'], metrics['best_valid_result']['auc']]], columns=['learning_rate', 'learner', 'model', 'embedding_size', 'weight_decay', 'train_batch_size', 'epoch', 'mae', 'rmse', 'auc_roc']
        )])

        hyper.to_csv(filename, index=False)
        remove_all_files('./log/best')
        remove_all_files('./log/')

        return metrics['best_valid_result']['mae']

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=200)

    t1 = time.time()
    total = t1 - t0
    logger.info('time cost: '+ f': {total}s')