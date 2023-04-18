#!/usr/bin/python3

import optuna
import numpy as np
import pandas as pd

from itertools import product
from sklearn.metrics import mean_absolute_error
from matrix_factorization import KernelMF, train_update_test_split

# Constants
use_cols = ['user:token', 'item:token', 'cnt:float']

def apply_mf(data_dict, kernel_n_factors, kernel_n_epochs, kernel_lr, kernel_reg, update_epochs, update_lr):
    # Train, update, test split
    X_train, y_train, X_update, y_update, X_test, y_test = data_dict['X_train'], data_dict['y_train'], data_dict['X_update'], data_dict['y_update'], data_dict['X_test'], data_dict['y_test']

    # Train model
    matrix_fact = KernelMF(n_factors=kernel_n_factors, n_epochs=kernel_n_epochs, lr=kernel_lr, reg=kernel_reg)
    matrix_fact.fit(X_train, y_train)

    # Update model
    matrix_fact.update_users(X_update, y_update, n_epochs=update_epochs, lr=update_lr)

    # Predict
    y_pred = matrix_fact.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    print('MAE: {}'.format(mae))

    return mae

if __name__ == '__main__':
    # Load data and Rename columns
    data = pd.read_csv('./dataset/frappe/frappe.inter', usecols=use_cols)
    data.columns=['user_id', 'item_id', 'rating']

    # Drop duplicates
    data = data.iloc[data[['user_id', 'item_id']].drop_duplicates().index, :]

    # Train and Test Split
    X_train, y_train, X_update, y_update, X_test, y_test = train_update_test_split(data, frac_new_users=0.15)
    data_dict = {'X_train': X_train, 'y_train': y_train, 'X_update': X_update, 'y_update': y_update, 'X_test': X_test, 'y_test': y_test}

    # Optuna
    def objective(trial):
        filename = 'frappe_hyper_mf_optuna.csv'
        hyper = pd.read_csv(f"./hypers/{filename}")
        kernel_n_factor = trial.suggest_int('kernel_n_factor', 10, 500)
        kernel_n_epoch = trial.suggest_int('kernel_n_epoch', 10, 200)
        kernel_lr = trial.suggest_loguniform('kernel_lr', 1e-4, 1e-1)
        kernel_reg = trial.suggest_loguniform('kernel_reg', 1e-4, 1e-1)
        update_epoch = trial.suggest_int('update_epoch', 10, 200)
        update_lr = trial.suggest_loguniform('update_lr', 1e-4, 1e-1)

        mae = apply_mf(data_dict, kernel_n_factor, kernel_n_epoch, kernel_lr, kernel_reg, update_epoch, update_lr)
        hyper = pd.concat([hyper, pd.DataFrame({'kernel_n_factor': [kernel_n_factor], 'kernel_n_epoch': [kernel_n_epoch], 'kernel_lr': [kernel_lr], 'kernel_reg': [kernel_reg], 'update_epoch': [update_epoch], 'update_lr': [update_lr], 'mae': [mae]})])

        hyper.to_csv(f"./hypers/{filename}", index=False)
        return mae
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=1000)