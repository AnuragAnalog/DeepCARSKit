#!/usr/bin/python3

import numpy as np
import pandas as pd

from surprise import SVD
from surprise.reader import Reader
from surprise.dataset import Dataset
from surprise.model_selection import train_test_split, cross_validate

# Constants
use_cols = ['user:token', 'item:token', 'cnt:float']

if __name__ == '__main__':
    # Load the data
    data = pd.read_csv('./dataset/frappe/frappe.inter', usecols=use_cols)

    reader = Reader(rating_scale=(1, 5))
    dataset_loader = Dataset(reader)

    # Load the data
    dataset = dataset_loader.load_from_df(data, reader)

    # Cross validation
    print('Cross validation')
    print(cross_validate(SVD(), dataset, measures=['RMSE', 'MAE'], cv=5, verbose=True))