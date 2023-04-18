#!/usr/bin/python3

import os
import wget
import pandas as pd
import streamlit as st
import plotly.express as px

from zipfile import ZipFile

# URL for the data
url = 'https://archive.org/download/cars_results/params_info.zip'

# Download the data
if not os.path.exists('./params_info/'):
    os.mkdir('params_info')
    wget.download(url, 'params_info.zip')

    # Unzip the data
    with ZipFile('params_info.zip', 'r') as zf:
        zf.extractall('./')

    # Remove the zip file
    os.remove('params_info.zip')

# Set the page configuration
st.set_page_config(
    page_title='Results',
    page_icon=':bar_chart:',
    layout='wide',
    initial_sidebar_state='auto'
)

# Set the page title
st.title('Hyperparameter Results Information')

st.sidebar.header('Choose some basic settings')

# Choose the dataset
dataset_type = st.sidebar.selectbox(
    'Choose the dataset',
    ['TripAdvisor', 'Frappe']
)

# Choose the models
model_type = st.sidebar.selectbox(
    'Choose the model',
    ['Matrix Factorization', 'Factorization Machines', 'NeuCMFs']
)

# Choose the hyperparameters optimization framework
hyper_type = st.sidebar.selectbox(
    'Choose the hyperparameters optimization framework',
    ['Grid Search', 'Optuna']
)

fname = ''
if dataset_type == 'TripAdvisor':
    fname += 'ta_hyper'
elif dataset_type == 'Frappe':
    fname += 'frappe_hyper'

if model_type == 'Matrix Factorization':
    fname += '_mf'
elif model_type == 'Factorization Machines':
    fname += '_fms'
elif model_type == 'NeuCMFs':
    fname += '_neucmf'

if hyper_type == 'Grid Search':
    fname += '_gs.csv'
elif hyper_type == 'Optuna':
    fname += '_optuna.csv'

st.markdown("Applied **{model_type}** on **{dataset_type}** dataset with **{hyper_type}**.")

# Data Presentation
data_present = st.selectbox(
    "How would you like me to present the data?",
    ["Table", "Charts"]
)

# Select configuration file
st.write('## Hyperparameter Results Information')

# Read the data
try:
    data = pd.read_csv(f'./params_info/{fname}')
except FileNotFoundError:
    st.error('No data found. Please try again later.')
    st.stop()

# Present the data
if data_present == 'Table':
    # View the data

    st.write("You can click on the column names to sort the table.")
    st.dataframe(data)
elif data_present == 'Charts':
    # Select the columns to plot

    if model_type in ['NeuCMFs', 'Factorization Machines']:
        agg_df = data.groupby(['model']).agg({'mae': 'min'}).reset_index()

        st.plotly_chart(
            px.bar(
                agg_df,
                x='model',
                y='mae',
                title='Best MAE by Model'
            )
        )
    elif model_type == 'Matrix Factorization':
        st.error('No charts available for Matrix Factorization model.')