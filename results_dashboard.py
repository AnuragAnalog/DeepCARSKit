#!/usr/bin/python3

import os
import wget
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from zipfile import ZipFile

# Constant variables
DATASET_TYPE = ['TripAdvisor', 'Frappe']
MODEL_TYPE = ['Factorization Machines', 'NeuCMFs']
HYPER_TYPE = ['Grid Search', 'Optuna']
METRICS = ['MAE', 'RMSE', 'AUC_ROC']
PLOT_TYPE = ['Bar Plot', 'Contour Plot', 'Parallel Coordinates']

# URL for the data
url = 'https://archive.org/download/cars_results/params_info.zip'

# Download the data
if not os.path.exists('./params_info.zip'):
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
dataset_type = st.sidebar.selectbox('Choose the dataset', DATASET_TYPE)

# Choose the models
model_type = st.sidebar.selectbox('Choose the model', MODEL_TYPE)

# Choose the hyperparameters optimization framework
hyper_type = st.sidebar.selectbox('Choose the hyperparameters optimization framework', HYPER_TYPE)

fname = ''
if dataset_type == 'TripAdvisor':
    fname += 'ta_hyper'
elif dataset_type == 'Frappe':
    fname += 'frappe_hyper'

if model_type == 'Factorization Machines':
    fname += '_fms'
elif model_type == 'NeuCMFs':
    fname += '_neucmf'

if hyper_type == 'Grid Search':
    fname += '_gs.csv'
elif hyper_type == 'Optuna':
    fname += '_optuna.csv'

st.text(f"Applied {hyper_type} on {model_type} dataset with {dataset_type}.")

# Data Presentation
data_present = st.selectbox(
    "How would you like me to present the data?",
    ["Table", "Charts"]
)

# Select configuration file
st.write('## Hyperparameter Results Information')

# Read the data
data = pd.read_csv(f'./params_info/{fname}')

# Present the data
if data_present == 'Table':
    # View the data

    st.write("You can click on the column names to sort the table.")
    st.dataframe(data)
elif data_present == 'Charts':
    # Select the columns to plot

    st.sidebar.subheader('Choose the plot settings')
    plot_type = st.sidebar.selectbox(
        'Choose the type of plot', ['Bar Plot', 'Contour Plot', 'Parallel Coordinates']
    )

    metric = st.sidebar.selectbox(
        'Choose the performance measure', METRICS
    ).lower()

    if plot_type == "Bar Plot":
        agg_df = data.groupby(['model']).agg({metric: 'min'}).reset_index()
        fig = px.bar(agg_df, x='model', y=metric, title=f'Best {metric.upper()} by Model')
    elif plot_type == "Contour Plot":
        x_axis = st.sidebar.selectbox(
            'Choose the x-axis', [col for col in data.columns if col not in map(str.lower, METRICS)], key='x_axis'
        )
        y_axis = st.sidebar.selectbox(
            'Choose the y-axis', [col for col in data.columns if col not in list(map(str.lower, METRICS))+[x_axis]], key='y_axis'
        )

        contour_data = pd.pivot_table(data, values=metric, index=[x_axis], columns=[y_axis])
        contour_data.fillna(0, inplace=True)
        print(contour_data)
        fig = go.Figure(data=[go.Surface(z=contour_data.values, contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True))])
        fig.update_layout(title=f'{metric.upper()} Contour Plot', 
                        autosize=False,
                        scene=dict(xaxis=dict(title=x_axis), yaxis=dict(title=y_axis), zaxis=dict(title=metric))
                    )
    elif plot_type == "Parallel Coordinates":
        options = [col for col in data.columns if col not in map(str.lower, METRICS)]
        parallels = st.sidebar.multiselect("Select all Parallel Coordinates", default=options, options=options)

        fig = px.parallel_coordinates(data, color=metric, dimensions=parallels+[metric], color_continuous_scale=px.colors.diverging.Tealrose, title=f'{metric.upper()} Parallel Coordinates')

    st.plotly_chart(fig, use_container_width=True)