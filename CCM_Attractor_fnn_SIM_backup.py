#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 12:12:57 2023

@author: ofir
"""

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/ofir/Dropbox/Projects/Succession_CCM_LSTM_Embedding/fnn')

from fnn.models import MLPEmbedding, LSTMEmbedding
import numpy as np
import pandas as pd
import os
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from sklearn.preprocessing import MinMaxScaler
from fnn.regularizers import FNN
import pickle

# Define the base folder path
BaseFolder = "/media/ofir/StorageDevice/Dropbox/Projects/Succession_CCM_LSTM_Embedding/"

# Initialize parameters for time series
y1p, y2p, y3p, y4p = 0.4, 0.4, 0.4, 0.4
y1, y2, y3, y4 = [y1p], [y2p], [y3p], [y4p]

cycles = 1000
for i in range(cycles):
    y_1 = y1p * (3.9 - (3.9 * y1p))
    y_2 = y2p * (3.6 - (0.4 * y1p) - (3.6 * y2p))
    y_3 = y3p * (3.6 - (0.4 * y2p) - (3.6 * y3p))
    y_4 = y4p * (3.8 - (0.35 * y3p) - (3.8 * y4p))
    
    y1.append(y_1)
    y2.append(y_2)
    y3.append(y_3)
    y4.append(y_4)
    
    y1p, y2p, y3p, y4p = y_1, y_2, y_3, y_4

# Combine into DataFrame
df_timeseries = pd.DataFrame({'y1': y1, 'y2': y2, 'y3': y3, 'y4': y4})

# Normalize the data to 0-1 range
df_upsampled_normalized = pd.DataFrame(index=df_timeseries.index)
AllScalersDict = {}
for i in df_timeseries.columns:
    scaler = MinMaxScaler((0, 1))
    scaled_data = scaler.fit_transform(df_timeseries[i].values.reshape(-1, 1))
    df_upsampled_normalized[i] = [j[0] for j in scaled_data]
    AllScalersDict[i] = scaler

# Plot 3D scatter function (no lines, smaller dots, same color)
def Plot3D(coords, title, embedding_type):
    X, Y, Z = coords[:, 0], coords[:, 1], coords[:, 2]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot, smaller dots, all dots same color
    ax.scatter(X, Y, Z, s=10, color='blue')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Modify the title to reflect the type of embeddings being plotted
    ax.set_title(f"{title} ({embedding_type})")
    plt.show()

# Plot facets as 3D scatter without lines, smaller dots, same color
def plot3D_facets(coords, col, embedding_type):
    X, Y, Z = coords[:, 0], coords[:, 1], coords[:, 2]

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
    
    # Scatter plots, smaller dots, same color
    ax[0].scatter(X, Y, s=10, color='blue')
    ax[1].scatter(Y, Z, s=10, color='blue')
    ax[2].scatter(X, Z, s=10, color='blue')

    for a in ax:
        a.set_xlabel('Dimension 1')
        a.set_ylabel('Dimension 2')

    # Modify the title to reflect the type of embeddings being plotted
    fig.suptitle(f"{col} ({embedding_type})")
    plt.savefig(BaseFolder + col + "_atractor.png")

# Generate embeddings and plot them using the functions with descriptive titles
FullCols = df_upsampled_normalized.columns
Dict_embeddings = {}

# FNN Embeddings
for col in FullCols:
    # FNN Embeddings
    model = MLPEmbedding(3, time_window=6, latent_regularizer=FNN(0.01), random_state=12)
    embedding = model.fit_transform(df_upsampled_normalized[col][:].values, learning_rate=3e-3)
    
    Dict_embeddings[col] = embedding
    # Update title to include "FNN Embeddings"
    plot3D_facets(embedding, col, embedding_type="FNN Embeddings")
    Plot3D(embedding, col, embedding_type="FNN Embeddings")

# Save embeddings as pickle
with open(BaseFolder + 'All_ccm1_SIM_embeddings.pickle', 'wb') as handle:
    pickle.dump(Dict_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Reload and scatter plot embeddings for raw dataset
with open(BaseFolder + 'All_ccm1_SIM_embeddings.pickle', 'rb') as handle:
    Dict_embeddings = pickle.load(handle)

# Plot raw dataset time embeddings as 3D scatter (Raw Time Embeddings)
for col in FullCols:
    raw_data = df_upsampled_normalized[col].values.reshape(-1, 1)
    
    # For raw time series, we need to create 3D coordinates by shifting windows (lagging) across the series
    raw_embedding = np.column_stack([raw_data[:-2], raw_data[1:-1], raw_data[2:]])
    
    # Plot raw time embeddings
    plot3D_facets(raw_embedding, col, embedding_type="Raw Time Embeddings")
    Plot3D(raw_embedding, col, embedding_type="Raw Time Embeddings")

for col in FullCols:
    model = LSTMEmbedding(3, time_window=6, random_state=12)  # Example LSTM embedding
    embedding = model.fit_transform(df_upsampled_normalized[col][:].values, learning_rate=3e-3)

    Dict_embeddings[col] = embedding
    # Update title to include "LSTM Embeddings"
    plot3D_facets(embedding, col, embedding_type="LSTM Embeddings")
    Plot3D(embedding, col, embedding_type="LSTM Embeddings")



time_series_df = df_upsampled_normalized.copy()




import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

window_length = 10  # Length of each sequence
n_features = 1      # Number of features in the data

# Function to create sequences
def create_sequences(data, window_length):
    sequences = []
    for i in range(len(data) - window_length + 1):
        sequence = data[i:i + window_length]
        sequences.append(sequence)
    return np.array(sequences)

# Dictionary to store embeddings
Dict_embeddings = {}

for col in FullCols:
    # Extract the column data
    data = time_series_df[[col]].values  # Assuming time_series_df is a DataFrame

    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Create sequences
    sequences = create_sequences(scaled_data, window_length)

    # Reshape sequences for LSTM input
    X = sequences.reshape(sequences.shape[0], window_length, n_features)

    # Define the input layer
    inputs = Input(shape=(window_length, n_features))

    # LSTM layer
    lstm_out = LSTM(32, activation='relu', return_sequences=False)(inputs)

    # Dense layers
    dense_1 = Dense(16, activation='relu')(lstm_out)
    embeddings = Dense(3, activation='linear', name='embedding_layer')(dense_1)

    # Output layer (reconstruct the middle point of the sequence)
    output = Dense(1, activation='linear')(embeddings)

    # Define the model
    model = Model(inputs=inputs, outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Target is the middle value of each sequence
    y = X[:, window_length // 2, 0]

    # Train the model
    model.fit(X, y, epochs=50, verbose=0)

    # Create a separate model to extract embeddings
    embedding_model = Model(inputs=model.input, outputs=model.get_layer('embedding_layer').output)

    # Get embeddings for each sequence
    embeddings_list = embedding_model.predict(X)

    # Store embeddings
    Dict_embeddings[col] = embeddings_list

    # Extract x, y, z coordinates
    x_coords = embeddings_list[:, 0]
    y_coords = embeddings_list[:, 1]
    z_coords = embeddings_list[:, 2]

    # Plot the embeddings
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_coords, y_coords, z_coords, s=3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(col)
    plt.show()
        






