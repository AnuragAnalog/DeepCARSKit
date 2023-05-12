#!/usr/bin/env python
# coding: utf-8

# # DeepFM - no context

# In[1]:


# Required modules

import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf

from itertools import product

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# In[2]:


# Load the data

use_cols = ['user_id:token', 'item_id:token', 'rating:float']
data = pd.read_csv('./dataset/tripadvisor/tripadvisor.inter', usecols=use_cols)
data.head()


# In[3]:


# Encoding the user_id column

user_encoder = LabelEncoder()
data['user_id:token'] = user_encoder.fit_transform(data['user_id:token'].values)

# Encoding the item_id column
item_encoder = LabelEncoder()
data['item_id:token'] = item_encoder.fit_transform(data['item_id:token'].values)


# In[4]:


# Renaming columns

data.columns = ['user_id', 'item_id', 'rating']


# In[5]:


# Split the data into train and test sets

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


# In[6]:


# Define the number of unique users and movies

num_users = data['user_id'].nunique()
num_movies = data['item_id'].nunique()

# Define embedding size

embedding_size = 10


# In[7]:


# Model definition

def get_model(embedding_size, weight_decay=0.0):
    # Define the input shape
    input_shape = (train_data.shape[1] - 1,)

    l2_reg = tf.keras.regularizers.l2(weight_decay)

    # Define input layers
    user_input = tf.keras.layers.Input(shape=(1,))
    movie_input = tf.keras.layers.Input(shape=(1,))

    # Define user embedding
    user_embedding = tf.keras.layers.Embedding(num_users, embedding_size, input_length=1)(user_input)
    user_embedding = tf.keras.layers.Flatten()(user_embedding)

    # Define movie embedding
    movie_embedding = tf.keras.layers.Embedding(num_movies, embedding_size, input_length=1)(movie_input)
    movie_embedding = tf.keras.layers.Flatten()(movie_embedding)

    # Concatenate user and movie embeddings
    concat = tf.keras.layers.concatenate([user_embedding, movie_embedding])

    # Define FM part
    fm = tf.keras.layers.Dense(1, activation=None)(concat)

    # Define DNN part
    dnn = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2_reg)(concat)
    dnn = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2_reg)(dnn)
    dnn = tf.keras.layers.Dense(1, activation=None)(dnn)

    # Concatenate FM and DNN parts
    concat = tf.keras.layers.concatenate([fm, dnn])

    # Define output layer
    output = tf.keras.layers.Flatten()(concat)

    # Define the model
    model = tf.keras.models.Model(inputs=[user_input, movie_input], outputs=output)

    return model

model = get_model(10)


# In[8]:


# Compile the model

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])


# In[9]:


# Fit the model

history = model.fit([train_data['user_id'], train_data['item_id']], 
                    train_data['rating'], 
                    validation_data=([test_data['user_id'], test_data['item_id']], test_data['rating']),
                    epochs=100, batch_size=64)


# In[10]:


# Evaluate the model

model.evaluate([test_data['user_id'], test_data['item_id']], test_data['rating'])


# ## Cross validation

# In[ ]:


# Hyperparameter tuning

from sklearn.model_selection import KFold

filename = 'ta_hyper_deepfm_gs.csv'
hyper = pd.read_csv(f"./hypers/version3/{filename}")
learning_rate = [1e-6, 1e-3, 1e-1]
learner = ['adam', 'RMSprop']
epochs = [10, 20, 30]
embedding_size = [32, 128, 512]
weight_decay = [1e-3, 1e-1]
train_batch_size = [500, 1000]

outer_kf = KFold(n_splits=10)
test_metrics = dict(mse=list(), mae=list())

for i, (remain_index, test_index) in enumerate(outer_kf.split(data)):
    for lr, learner, embedding_size, weight_decay, epoch, train_batch_size in tqdm.tqdm(product(learning_rate, learner, embedding_size, weight_decay, epochs, train_batch_size)):
        train_metrics, valid_metrics = dict(mse=list(), mae=list()), dict(mse=list(), mae=list())
        inner_kf = KFold(n_splits=10)

        for train_index, valid_index in inner_kf.split(data.iloc[remain_index]):
            train_set = data.iloc[train_index]
            valid_set = data.iloc[valid_index]
            model = get_model(embedding_size=embedding_size, weight_decay=weight_decay)
            if learner == 'adam':
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse', metrics=['mae'])
            else:
                model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr), loss='mse', metrics=['mae'])

            model.fit([train_set['user_id'], train_set['item_id']], 
                      train_set['rating'], 
                      validation_data=([valid_set['user_id'], valid_set['item_id']], valid_set['rating']),
                      epochs=epoch, batch_size=train_batch_size, verbose=0)
            vmse, vmae = model.evaluate([valid_set['user_id'], valid_set['item_id']], valid_set['rating'])

            valid_metrics['mse'].append(vmse)
            valid_metrics['mae'].append(vmae)
        final_valid_mse = np.mean(valid_metrics['mse'])
        final_valid_mae = np.mean(valid_metrics['mae'])

        hyper = pd.concat([hyper, pd.DataFrame(
                [[lr, learner, embedding_size, weight_decay, train_batch_size, epoch, final_valid_mae, np.sqrt(final_valid_mse), i]], columns=['learning_rate', 'learner', 'embedding_size', 'weight_decay', 'train_batch_size', 'epoch', 'mae', 'rmse', 'test_no']
            )])
        hyper.to_csv(f"./hypers/version3/{filename}", index=False)

    hyper = pd.read_csv(f"./hypers/version3/{filename}")
    hyper = hyper[hyper['test_no'] == i]
    temp_hyper = hyper.groupby(['learning_rate', 'learner', 'embedding_size', 'weight_decay', 'train_batch_size', 'epoch']).agg({'mae': 'mean'}).reset_index().sort_values(by='mae', ascending=True)
    best_hyper = temp_hyper.iloc[0]

    remain_set = data.iloc[remain_index]
    test_set = data.iloc[test_index]
    model = get_model(embedding_size=best_hyper['embedding_size'], weight_decay=best_hyper['weight_decay'])
    if best_hyper['learner'] == 'adam':
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_hyper['learning_rate']), loss='mse', metrics=['mae'])
    else:
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=best_hyper['learning_rate']), loss='mse', metrics=['mae'])

    model.fit([remain_set['user_id'], remain_set['item_id']], 
            remain_set['rating'], 
            validation_data=([test_set['user_id'], test_set['item_id']], test_set['rating']),
            epochs=best_hyper["epoch"], batch_size=best_hyper["train_batch_size"], verbose=0)
    
    test_mse, test_mae = model.evaluate([test_set['user_id'], test_set['item_id']], test_set['rating'])
    hyper = pd.concat([hyper, pd.DataFrame(
            [[best_hyper['learning_rate'], best_hyper['learner'], best_hyper['embedding_size'], best_hyper['weight_decay'], best_hyper['train_batch_size'], best_hyper['epoch'], test_mae, np.sqrt(test_mse), -i]], columns=['learning_rate', 'learner', 'embedding_size', 'weight_decay', 'train_batch_size', 'epoch', 'mae', 'rmse', 'test_no']
        )])
    hyper.to_csv(f"./hypers/version3/test_{filename}", index=False)
