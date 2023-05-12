#!/usr/bin/python3
# coding: utf-8

import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class DeepFMNC():
    def __init__(self, n_users, n_items, embed_size=8, seed=42):
        self.n_users = n_users
        self.n_items = n_items
        self.seed = seed
        self.embed_size = embed_size

    def create_model(self, input_shape, layer_nums, weight_decay=0.0):
        # Define the input shape
        input_shape = (input_shape - 1,)

        l2_reg = tf.keras.regularizers.l2(weight_decay)

        # Define input layers
        user_input = tf.keras.layers.Input(shape=(1,))
        movie_input = tf.keras.layers.Input(shape=(1,))

        # Define user embedding
        user_embedding = tf.keras.layers.Embedding(self.n_users, self.embed_size, input_length=1)(user_input)
        user_embedding = tf.keras.layers.Flatten()(user_embedding)

        # Define movie embedding
        movie_embedding = tf.keras.layers.Embedding(self.n_items, self.embed_size, input_length=1)(movie_input)
        movie_embedding = tf.keras.layers.Flatten()(movie_embedding)

        # Concatenate user and movie embeddings
        concat = tf.keras.layers.concatenate([user_embedding, movie_embedding])

        # Define FM part
        fm = tf.keras.layers.Dense(1, activation=None)(concat)

        # Define DNN part
        for i, layer_num in enumerate(layer_nums):
            if i == 0:
                dnn = tf.keras.layers.Dense(layer_num, activation='relu', kernel_regularizer=l2_reg)(concat)
            else:
                dnn = tf.keras.layers.Dense(layer_num, activation='relu', kernel_regularizer=l2_reg)(dnn)
        dnn = tf.keras.layers.Dense(1, activation=None)(dnn)

        # Concatenate FM and DNN parts
        concat = tf.keras.layers.concatenate([fm, dnn])

        # Define output layer
        output = tf.keras.layers.Flatten()(concat)

        # Define the model
        model = tf.keras.models.Model(inputs=[user_input, movie_input], outputs=output)

        return model
    
    def train_model(self, train_data, epochs=10, batch_size=32, val_data=None):
        # Create model
        self.model = self.create_model(input_shape=train_data.shape[1], layer_nums=[64, 32, 16, 8], weight_decay=0.0001)

        # Compile model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])

        # Fit model
        history = self.model.fit(x=[train_data['user_id'], train_data['item_id']], y=train_data['rating'], batch_size=batch_size, epochs=epochs, validation_data=([val_data['user_id'], val_data['item_id']], val_data['rating']), verbose=1)

        return history
    
    def predict(self, test_data):
        # Predict
        predictions = self.model.predict([test_data['user_id'], test_data['item_id']])

        return predictions
    
    def evaluate(self, test_data):
        # Evaluate
        self.model.evaluate([test_data['user_id'], test_data['item_id']], test_data['rating'])

        return None
    
if __name__ == '__main__':
    # Load data
    use_cols = ['user_id:token', 'item_id:token', 'rating:float']
    data = pd.read_csv('./dataset/tripadvisor/tripadvisor.inter', usecols=use_cols)

    # Encoding the user_id column
    user_encoder = LabelEncoder()
    data['user_id:token'] = user_encoder.fit_transform(data['user_id:token'].values)

    # Encoding the item_id column
    item_encoder = LabelEncoder()
    data['item_id:token'] = item_encoder.fit_transform(data['item_id:token'].values)

    data.columns = ['user_id', 'item_id', 'rating']

    # Define the number of unique users and movies
    num_users = data['user_id'].nunique()
    num_movies = data['item_id'].nunique()

    # Define embedding size
    embedding_size = 10

    # Create train and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    input_size = train_data.shape[1]

    # Create model
    model = DeepFMNC(n_users=num_users, n_items=num_movies, embed_size=embedding_size)

    # Train model
    history = model.train_model(train_data, epochs=10, batch_size=32, val_data=test_data)

    # Predict
    predictions = model.predict(test_data)

    # Evaluate
    model.evaluate(test_data)