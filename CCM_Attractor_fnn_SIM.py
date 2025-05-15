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
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from fnn.regularizers import FNN
import skccm as ccm
from skccm.utilities import train_test_split






# Define the base folder path
BaseFolder = "/media/ofir/StorageDevice/Dropbox/Projects/Succession_CCM_LSTM_Embedding/"
#BaseFolder = "/home/ofir/Dropbox/Projects/Gideon/"

y1p = 0.4
y2p = 0.4
y3p = 0.4
y4p = 0.4

y1 = [y1p]
y2 = [y2p]
y3 = [y3p]
y4 = [y4p]

cycles = 1000

for i in range(0, cycles):
    y_1 = y1p * (3.9 - (3.9 * y1p))
    y_2 = y2p * (3.6 - (0.4 * y1p) - (3.6 * y2p))
    y_3 = y3p * (3.6 - (0.4 * y2p) - (3.6 * y3p))
    y_4 = y4p * (3.8 - (0.35 * y3p) - (3.8 * y4p))
   
    y1.append(y_1)
    y2.append(y_2)
    y3.append(y_3)
    y4.append(y_4)

    y1p = y_1
    y2p = y_2
    y3p = y_3
    y4p = y_4


concated_ = pd.DataFrame()
 
concated_['y1'] = y1
concated_['y2'] = y2
concated_['y3'] = y3
concated_['y4'] = y4



df_timeseries = concated_.copy()

#Normalize 0-1
df_upsampled_normalized = pd.DataFrame(index = df_timeseries.index)
#df_upsampled_normalized = df_concated_smoothed.copy()
AllScalersDict = {}
for i in df_timeseries.columns:
    scaler = MinMaxScaler((0,1))
    scaled_data = scaler.fit_transform(df_timeseries[i].values.reshape(-1, 1))
    df_upsampled_normalized[i] = [j[0] for j in scaled_data]
    AllScalersDict[i] = scaler

df_upsampled_normalized



def plot3D_facets(coords, col):    
    # Get the number of data points
    num_points = len(coords)
    
    marker_sizes = 500 
    
    
    
    # Create a color map based on chronological time
    color_map = plt.get_cmap('RdBu_r')  
    
    # Generate an array of colors based on the chronological order of data points
    colors = [color_map(i / (num_points - 1)) for i in range(num_points)]
    
    plt.figure()
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20*4, 20))
    ax[0].scatter(coords[:, 0], coords[:, 1], c=colors, s=marker_sizes)  # Use scatter to apply colors
    ax[0].plot(coords[:, 0], coords[:, 1])

    ax[1].scatter(coords[:, 1], coords[:, 2], c=colors, s=marker_sizes)
    ax[1].plot(coords[:, 1], coords[:, 2])
    
    
    ax[2].scatter(coords[:, 0], coords[:, 2], c=colors, s=marker_sizes)
    ax[2].plot(coords[:, 0], coords[:, 2])
    
    plt.title(col)
    plt.savefig(BaseFolder+col+"_atractor.png")




def density(coords, col):
    num_points = len(coords)
    marker_sizes = 500
    # Create a color map based on chronological time
    color_map = plt.get_cmap('RdBu_r')  
    # Generate an array of colors based on the chronological order of data points
    colors = [color_map(i / (num_points - 1)) for i in range(num_points)]
    # Create a 2D KDE plot for each pair of dimensions
    kde_plots = []
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20*4, 20))
    for i, (dim1, dim2) in enumerate([(0, 1), (1, 2), (0, 2)]):
        ax[i].scatter(coords[:, dim1], coords[:, dim2], c=colors, s=marker_sizes, cmap="RdBu_r", alpha=0.85)
        ax[i].plot(coords[:, dim1], coords[:, dim2])
        # Create a KDE plot as a heatmap
        kde = sns.kdeplot(coords[:, dim1], coords[:, dim2], ax=ax[i], cmap="viridis",
                          shade=True, shade_lowest=False, alpha=0.25)
    
        kde_plots.append(kde)
    ax.set_title(col)
    plt.show()


#####################

from mpl_toolkits.mplot3d import Axes3D



def Plot3D(coords):
    num_points = len(coords)
    marker_sizes = 100 
    # Create a color map based on chronological time
    color_map = plt.get_cmap('RdBu_r')  
    
    # Generate an array of colors based on the chronological order of data points
    colors = [color_map(i / (num_points - 1)) for i in range(num_points)]
    X=coords[:, 0]
    Y=coords[:, 1]
    Z=coords[:, 2]
    
    vmin = min(min(X), min(Y), min(Z))
    vmax = max(max(X), max(Y), max(Z))
    
    fig = plt.figure( figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    
    xflat = np.full_like(X, vmin)
    yflat = np.full_like(Y, vmax)
    zflat = np.full_like(Z, vmin)
    
    plt.plot(xflat, Y, Z, color="gray", alpha=0.5)
    plt.plot(X, yflat, Z, color="gray", alpha=0.5)
    plt.plot(X, Y, zflat, color="gray", alpha=0.5)
    
    print(colors)
    print(marker_sizes)
    
    ax.scatter(X,Y,Z, c=colors, s=marker_sizes)
    #ax.plot(X,Y,Z)
    
    
    plt.show()








FullCols = df_upsampled_normalized.columns
Dict_embeddings = {}

for col in FullCols:
    model = MLPEmbedding(3, time_window=15, latent_regularizer=FNN(0.1), random_state=12)
    
    # =============================================================================
    #model = LSTMEmbedding(3, 
    #                     time_window=15,
    #                     latent_regularizer=FNN(0.01),
    #                     random_state=0
    #                     )
    # =============================================================================   
    embedding = model.fit_transform(df_upsampled_normalized[col][:].values,
                                    learning_rate=3e-3) # make 3D embedding

    Dict_embeddings[col] = embedding


    plot3D_facets(embedding, col)

    Plot3D(embedding)

#Save embeddings as pickle
import pickle
#save amplified df as pickle to be read by the external process
with open(BaseFolder+'All_ccm1_SIM_embeddings.pickle', 'wb') as handle:
    pickle.dump(Dict_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)   




with open(BaseFolder + 'All_ccm1_SIM_embeddings.pickle', 'rb') as handle:
    Dict_embeddings = pickle.load(handle)


#Here feed CCM with ffn embeddings

#explore CCM


def amplifyData(df, subSetLength=600, jumpN=30):
    allDfs = []
    for i in list(range(1, len(df)-subSetLength, jumpN)):
        tmp = df.iloc[i:i+subSetLength]
        allDfs.append(tmp)
    return allDfs


def build_colsDict(df):
    dd = {}
    counter = 0
    for i in df.columns:
        counter=counter+1
        dd[i] = "col_"+str(counter)
        dd["col_"+str(counter)] = i
    return dd


amplified_dfs = amplifyData(df_upsampled_normalized, subSetLength=100, jumpN=100)
# =============================================================================
# 
# # first differencing
# #deltaX xmap Y
# for i, vali in enumerate(amplified_dfs):
#     for j in vali.columns:
#         if j in phytoCols_0_10_names:
#             #amplified_dfs[i][j] = np.log(amplified_dfs[i][j])
#             if not j in targetlist:
#                 amplified_dfs[i][j] = vali[j].diff()
#                 amplified_dfs[i][j] = amplified_dfs[i][j].dropna()
#                 
# =============================================================================
DictCols = build_colsDict(df_upsampled_normalized)

for i, vali in enumerate(amplified_dfs):
    vali.columns = [DictCols[i] for i in vali.columns]
    amplified_dfs[i] = vali



#save amplified df as pickle to be read by the external process
with open(BaseFolder+'All_ccm1_SIM_amplified_dfs.pickle', 'wb') as handle:
    pickle.dump(amplified_dfs, handle, protocol=pickle.HIGHEST_PROTOCOL)   

with open(BaseFolder+'All_ccm1_SIM_DictCols.pickle', 'wb') as handle:
    pickle.dump(DictCols, handle, protocol=pickle.HIGHEST_PROTOCOL)  

with open(BaseFolder+'All_ccm1_SIM_x1_x2_columns.pickle', 'wb') as handle:
    pickle.dump([FullCols, FullCols], handle, protocol=pickle.HIGHEST_PROTOCOL)     


os.system('python '+BaseFolder+'ccm_multiproc_fnn.py '+ BaseFolder + ' All_ccm1_SIM_' )

with open(BaseFolder + 'All_All_ccm1_SIM_' + 'results.pickle', 'rb') as handle:
    All_CCM_dfs = pickle.load(handle)



#check convergence
for counti, i in enumerate(All_CCM_dfs):
    All_CCM_dfs[counti] = list(All_CCM_dfs[counti])
    df_Scores = i[1]
    try:
        l=int(len(df_Scores)/2)
        if ((df_Scores["x1_mean"][l:].mean() >= df_Scores["x2_mean"][:l].mean()) and \
             (df_Scores["x1_mean"][-20:].mean() >= 0.05)):
            All_CCM_dfs[counti].append(True)
            print('true')
            print(All_CCM_dfs[counti][-2][-1][-4]+' ' +All_CCM_dfs[counti][-2][-1][-5])
        else:
            All_CCM_dfs[counti].append(False)
    except:
        All_CCM_dfs[counti].append(False)




# =======
plt.close()

CausalFeatures  = []

for i in All_CCM_dfs:
    if (len(i[2]) > 0):
        try:
            if (i[1]["x1_mean"][-30:].mean() >= 0.1) and (i[-1] == True):
                
            #if (i[-2] == True) and (i[-1] == True):
                i[1]["x1_mean"].plot()
                print(i[2][0][2] + ' ' + i[2][0][3])
                CausalFeatures.append([i[2][0][2], i[2][0][3],  i[1]["x1_mean"][-30:].mean()])
        except:
                xx=1

df_CausalFeatures = pd.DataFrame(data=CausalFeatures, columns=['x1', 'x2', 'Score'])


Features = list(df_CausalFeatures['x1'].unique()) + list(df_CausalFeatures['x2'].unique())
Features = list(set(Features))
#all causal variables vs themselvs




df_CausalFeatures.to_csv(BaseFolder+'CCM_SIM_CausalFeatures_results.csv')



#Here feed ECCM with ffn embeddings
with open(BaseFolder + 'All_All_ccm1_SIM_results.pickle', 'rb') as handle:
    All_causal_CCM_dfs = pickle.load(handle)


#ECCM ###############################################
# =============================================================================
# 
# df_CausalFeatures2 = pd.read_csv(BaseFolder+'CCM_SIM_CausalFeatures_results.csv')
# 
# df_CausalFeatures2 = df_CausalFeatures2[df_CausalFeatures2["Score"] >= 0.05]
# 
# 
# x1x2s = df_CausalFeatures2[['x1', 'x2']].values.tolist()
# x1x2s = [(i[0], i[1]) for i in x1x2s]
# 
# with open(BaseFolder+'All_ccm1_SIM_dataset.pickle', 'wb') as handle:
#     pickle.dump(df_upsampled_normalized, handle, protocol=pickle.HIGHEST_PROTOCOL)      
#    
# with open(BaseFolder+'All_ccm1_SIM_edges.pickle', 'wb') as handle:
#     pickle.dump(x1x2s, handle, protocol=pickle.HIGHEST_PROTOCOL)   
# 
# os.system('python '+BaseFolder+'eccm_multiproc_fnn.py ' + BaseFolder + ' All_ccm1_SIM_'  )
# 
# 
# =============================================================================


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam


time_series_df = df_upsampled_normalized.copy()

window_length = 5  # Define the length of your window
n_features = 1  # Number of features (in this case, assuming univariate time series)

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(time_series_df[['y2']])

# Reshape data for LSTM input (samples, time steps, features)
def create_sequences(data, window_length):
    sequences = []
    for i in range(len(data) - window_length + 1):
        sequence = data[i:i + window_length]
        sequences.append(sequence)
    return np.array(sequences)

sequences = create_sequences(scaled_data, window_length)


# Define an optimizer with a custom learning rate
custom_learning_rate = 0.001  # Change this to your desired learning rate
custom_optimizer = Adam(learning_rate=custom_learning_rate)


# Define and train an LSTM model on the entire dataset
model = Sequential([
    LSTM(32, activation='relu', input_shape=(window_length, n_features)),
    Dense(16, activation='relu'),
    Dense(3)  # Adjust output dimension based on your embedding size
])
model.compile(optimizer=custom_optimizer, loss='mse')

# Train the model on the entire dataset
model.fit(sequences, sequences[:, -1], epochs=10, verbose=0)

# Generate embeddings for the entire dataset
embeddings_list = model.predict(sequences)


embeddings_list = [list(i) for i in embeddings_list]

# Extract x, y, z coordinates from the list of points
x_coords = [point[0] for point in embeddings_list]
y_coords = [point[1] for point in embeddings_list]
z_coords = [point[2] for point in embeddings_list]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points
ax.scatter(x_coords, y_coords, z_coords, s=3)

# Set labels for axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()

































