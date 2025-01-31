
# @Author : Yong Zheng


import argparse
import time
import torch
import optuna
import pandas as pd
import multiprocessing as mcpu
from deepcarskit.quick_start import run
from logging import getLogger
from itertools import product

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

    # def objective(trial):
        # filename = 'frappe_hyper_neucmf_optuna.csv'
        # hyper = pd.read_csv(f"./hypers/{filename}")
        # learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        # learner = trial.suggest_categorical('learner', ['adam', 'RMSprop'])
        # model = trial.suggest_categorical('model', ['NeuCMFii', 'NeuCMFww', 'NeuCMFi0', 'NeuCMF0i', 'NeuCMFw0', 'NeuCMF0w'])
        # embedding_size = trial.suggest_categorical('embedding_size', [32, 64, 128, 256, 512])
        # weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-1)
        # train_batch_size = trial.suggest_categorical('train_batch_size', [500, 1000, 2000, 4000])
# 
        # custom_config_dict = {
            # "learning_rate": learning_rate,
            # "learner": learner,
            # "model": model,
            # "embedding_size": embedding_size,
            # "weight_decay": weight_decay,
            # "train_batch_size": train_batch_size
        # }
# 
        # metrics = run(config_file_list=config_list, custom_config_dict=custom_config_dict)
        # hyper = pd.concat([hyper, pd.DataFrame(
            # [[learning_rate, learner, model, embedding_size, weight_decay, train_batch_size, metrics['best_valid_result']['mae']]], columns=['learning_rate', 'learner', 'model', 'embedding_size', 'weight_decay', 'train_batch_size', 'mae']
        # )])
# 
        # hyper.to_csv(f"./hypers/{filename}", index=False)
# 
        # return metrics['best_valid_result']['mae']
# 
    # study = optuna.create_study(direction='minimize')
    # study.optimize(objective, n_trials=1000)

    run(config_file_list=config_list)

    t1 = time.time()
    total = t1 - t0
    logger.info('time cost: '+ f': {total}s')