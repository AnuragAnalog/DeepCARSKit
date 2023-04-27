
# @Author : Yong Zheng


import os
import time
import torch
import shutil
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

    dataset_type = 'ta'
    model_type = 'neucmf'

    filename = f"./hypers/version2/{dataset_type}_hyper_{model_type}_gs.csv"
    hyper = pd.read_csv(filename)

    learning_rate = [1e-1]
    learner = ['adam', 'RMSprop']
    epochs = [30]
    embedding_size = [32, 128, 512]
    weight_decay = [1e-3, 1e-1]
    train_batch_size = [500, 1000]

    if model_type == 'neucmf':
        model = ['NeuCMFii', 'NeuCMFww', 'NeuCMFi0', 'NeuCMF0i', 'NeuCMFw0', 'NeuCMF0w']
    elif model_type == 'fms':
        model = ['FM', 'DeepFM']

    for lr, e, l, m, es, wd, tbs in product(learning_rate, epochs, learner, model, embedding_size, weight_decay, train_batch_size):
        custom_config_dict = {
            "learning_rate": lr,
            "epochs": e,
            "learner": l,
            "model": m,
            "embedding_size": es,
            "weight_decay": wd,
            "train_batch_size": tbs
        }

        metrics = run(config_file_list=config_list, custom_config_dict=custom_config_dict)
        hyper = pd.concat([hyper, pd.DataFrame(
            [[lr, e, l, m, es, wd, tbs, metrics['best_valid_result']['mae'], metrics['best_valid_result']['rmse'], metrics['best_valid_result']['auc']]], columns=['learning_rate', 'epoch', 'learner', 'model', 'embedding_size', 'weight_decay', 'train_batch_size', 'mae', 'rmse', 'auc_roc']
        )])
        hyper.to_csv(filename, index=False)

        remove_all_files('./log/best')
        remove_all_files('./log/')

    t1 = time.time()
    total = t1 - t0
    logger.info('time cost: '+ f': {total}s')