#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 12:12:57 2023

@author: ofir
"""
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/media/ofir/StorageDevice/Dropbox/Projects/Succession_CCM_LSTM_Embedding/fnn')

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







zoo_cols = ['Rotifera',
             'Total_zoo',
             'Cops_eggs',
             'Clads_eggs',
             'Cladocera',
             'Copepodes']

phytoCols_0_10_names = ['4-Chlorophyte, sphere',
 '8-Trachelomonas spp',
 '4-Nephrocytium',
 '6-Rhodomonas spp',
 '2-Cylindrospermopsis',
 '3-Synedra ulna',
 '2-Microcystis flos-aquae',
 '4-Dictyosphaerium ehrenbergianum',
 '4-Chlamydomonas',
 '2-Chroococcus minutus',
 '2-Aphanizomenon flos aqua',
 '2-Lyngbya',
 '4-Franceia radians',
 '4-Oocystis spp',
 '4-Scenedesmus obliquus',
 '4-Coelastrum cambricum',
 '2-Cylindrospermopsis spiral',
 '3-Aulacoseira granulata',
 '3-Cymbella',
 '5-Entzia acuta',
 '4-Treubaria setigera',
 '4-Koliella',
 '4-Tetraedron new autospore',
 '4-Scenedesmus ecornis',
 '5-Peridiniopsis cunningtonii',
 '3-Rhoicosphenia',
 '2-Merismopedia',
 '4-Cosmarium S',
 '3-Anomoeoneis',
 '2-Anabaena sp',
 '4-Oocystis lacustris',
 '4-Quadrigula',
 '5-Peridiniopsis penardiforme',
 '5-Ceratium hirundinella',
 '3-Discostella',
 '5-Gymnodinium spp',
 '5-Peridiniopsis oculatum',
 '3-Navicula',
 '4-Kirchneriella microscopica',
 '2-Anabaena spiroides',
 '2-Microcystis aeruginosa',
 '3-Synedra L',
 '3-Nitzschia',
 '2-unknown filamentous cyano',
 '3-Synedra rumpens',
 '2-Cylindrospermopsis heterocyst',
 '4-Tetraedron caudatum',
 '4-Crucigenia fenestrata',
 '2-Microcystis wesenbergii',
 '4-Gloeocystis',
 '4-Coelastrum proboscideum',
 '2-Merismopedia minima',
 '4-Scenedesmus bijuga',
 '4-Spondylosium moniliforme',
 '4-Scenedesmus quadricauda',
 '0-Uroglena',
 '4-Scenedesmus spinosus',
 '5-Ceratium hirundinella cyst',
 '3-Synedra spp',
 '2-Coelosphaerium',
 '4-Scenedesmus denticulatus',
 '4-Kirchneriella obesa',
 '4-Coelastrum microporum',
 '4-Pediastrum simplex',
 '4-Scenedesmus spp',
 '4-Pediastrum boryanum',
 '4-Chodatella longisetta',
 '4-Micractinium pusillum',
 '4-Oocystis autospore',
 '1-Acronema',
 '4-Collodyction',
 '4-Nephrochlamys',
 '4-Oocystis novae-semliae',
 '4-Scenedesmus bicellularis',
 '4-Eudorina elegans',
 '3-Synedra M',
 '2-Chroococus turgidus',
 '4-Lagerheimia genevensis',
 '5-Peridiniopsis Hula',
 '4-Chodatella quadrisetta',
 '2-Aphanocapsa pulchra',
 '4-Crucigeniella rectangularis',
 '4-Selenastrum bibrianum',
 '3-Gomphonema',
 '4-Pediastrum sturmii',
 '4-Tetraedron regulare',
 '4-Chlorella',
 '3-Cyclotella meneghiniana',
 '4-Crucigenia tetrapedia',
 '1-Tetrachloris',
 '4-Franceia ovalis',
 '3-Cyclotella polymorpha',
 '5-Glenodinium oculatum',
 '4-Golenkinia radiata',
 '4-Tetraedron quadratum',
 '2-Anabaena nplanktonica',
 '2-Cylindrospermopsis raciborskyi',
 '4-Pediastrum clathratum',
 '0-Ochromonas',
 '4-Tetraedron triangulare',
 '4-Monoraphidium thin',
 '2-anabaena bergii',
 '4-Choricystis',
 '4-Chodatella citriformis autospore',
 '4-Closterium sp',
 '4-Closterium acerosum',
 '4-Pediastrum duplex',
 '4-Staurastrum gracile',
 '2-Aphanizomenon heterocyst',
 '3-Fragilaria',
 '4-Monoraphidium contortum',
 '5-Peridinium inconspicuum',
 '4-Kirchneriella lunaris',
 '2-Aphanocapsa elachista',
 '4-Tetraedron new',
 '4-Closterium acutum',
 '4-Crucigenia triangularis',
 '4-Actinastrum hantzschii',
 '2-Aphanizomenon oval',
 '2-Cyanophyte-sphere',
 '6-Cryptomonas spp',
 '4-Tetraedron minimum',
 '7-Prasynophyte',
 '4-Euastrum denticulatus',
 '3-Synedra affinis',
 '4-Cosmarium L',
 '4-Sphaerocystis',
 '4-Scenedesmus acuminatus',
 '7-Carteria cordiformis',
 '5-Peridiniopsis berolinse',
 '4-Staurastrum contortum',
 '4-Gloeococcus',
 '4-Ankistrodesmus nannoselene',
 '4-Kirchneriella elongata',
 '2-Merismopedia glauca',
 '2-Cyanodictyon imperfectum',
 '2-Microcystis pulvera',
 '4-Oocystis submarina',
 '4-Closterium aciculare',
 '4-Chodatella citriformis',
 '4-Monoraphidium arcuatum',
 '2-Radiocystis geminata',
 '3-Pleurosigma',
 '2-Aphanizomenon akinete',
 '3-Synedra acus',
 '3-Cyclotella-kuetzinginiana',
 '2-Cylindrospermopsis akinete',
 '4-Cosmarium laeve',
 '1-Planktomyces',
 '5-Glenodinium, colourless',
 '4-Pediastrum tetras',
 '5-Peridinium gatunense cyst',
 '2-Oscillatoria thick',
 '2-Chroococcus limneticus',
 '2-Raphidiopsis medit',
 '4-Coelastrum reticulatum',
 '4-Elakatothrix gelatinosa',
 '4-Tetrastrum triangulare',
 '2-Phormidium',
 '4-Selenastrum minutum',
 '2-Oscillatoria thin',
 '5-Dinoflagellate',
 '5-Peridiniopsis borgei',
 '3-Diatoma',
 '4-Chodatella ciliata',
 '0-Monas',
 '5-Peridiniopsis polonicum',
 '4-Cosmarium sphagnicolum',
 '4-Staurastrum manfeldti',
 '4-Dictyosphaerium pullchelum',
 '9-Erkenia subaequiciliata',
 '4-Chlorophyte, unknown',
 '4-Scenedesmus acutiformis',
 '4-Pandorina morum',
 '0-Malomonas',
 '8-Euglena',
 '4-Botryococcus braunii',
 '4-Scenedesmus armatus',
 '2-Romeria-like',
 '2-Limnothrix',
 '2-Microcystis botrys',
 '4-Tetraedron triangulare autospore',
 '4-Coelastrum scabrum',
 '5-Peridiniopsis elpatiewskyi',
 '4-Tetraedron sp',
 '4-Cocomyxa',
 '4-Ankistrodesmus falcutus',
 '2-Aphanocapsa delicatissima',
 '4-Mougeotia',
 '3-Epithemia',
 '5-Peridinium spp',
 '4-Staurastrum tetracerum',
 '4-Tetrastrum  apiculatum',
 '5-Peridinium gatunense',
 '4-Staurastrum spp',
 '5-Peridiniopsis Protoplast']


ChemCols_0_10_names = ['Nitrit',
 'Nitrate',
 'NH4',
 'Oxygen',
 'Norg_par',
 'Norg',
 'Cl',
 'So4',
 'H2S',
 'TSS',
 'PTD',
 'Norg_dis',
 'Port',
 'Turbidity',
 'PH',
 'Ntot',
 'Ptot']

ChemCols_30_40_names = [i+"_30_40" for i in ChemCols_0_10_names]

confounders = ['Temperature', 'Inflow']




rt_list = ['RT lake (m)',
                 'farction 2y',
                 'farction 5y',
                 'farction 10y',
                 'farction 15y',
                 'farction 20y',
                 'RT epi (m)',
                 'farction epi 2y',
                 'farction epi 5y',
                 'farction epi 10y',
                 'farction epi 15y',
                 'farction epi 20y',
                 'RT hypo (m)',
                 'farction hypo 2y',
                 'farction hypo 5y',
                 'farction hypo 10y',
                 'farction hypo 15y',
                 'farction hypo 20y']

# Define the base folder path
BaseFolder = "/media/ofir/StorageDevice/Dropbox/Projects/Succession_CCM_LSTM_Embedding/"
#BaseFolder = "/home/ofir/Dropbox/Projects/Gideon/"

df_timeseries = pd.read_csv(os.path.join(BaseFolder, "dataset.csv"))
try:
    df_timeseries = df_timeseries.set_index('index')
except:
    df_timeseries = df_timeseries.set_index('Date')



df_timeseries = df_timeseries.loc['2000-01-01':'2020-02-01']

#Normalize 0-1
df_upsampled_normalized = pd.DataFrame(index = df_timeseries.index)
#df_upsampled_normalized = df_concated_smoothed.copy()
AllScalersDict = {}
for i in df_timeseries.columns:
    scaler = MinMaxScaler((0,1))
    scaled_data = scaler.fit_transform(df_timeseries[i].values.reshape(-1, 1))
    df_upsampled_normalized[i] = [j[0] for j in scaled_data]
    AllScalersDict[i] = scaler

df_concated_fixed_outlayers = df_upsampled_normalized.copy()

#fix outlayers
for i in df_concated_fixed_outlayers.columns:
    mask = (np.abs(stats.zscore(df_concated_fixed_outlayers[i])) > 2.5)
    df_concated_fixed_outlayers[i] = df_concated_fixed_outlayers[i].mask(mask).interpolate()

try:
    df_concated_fixed_outlayers['Date'] = pd.DatetimeIndex(df_concated_fixed_outlayers.reset_index()['index'])
except:
    df_concated_fixed_outlayers['Date'] = pd.DatetimeIndex(df_concated_fixed_outlayers.reset_index()['Date'])

df_concated_fixed_outlayers = df_concated_fixed_outlayers.set_index('Date')


#df_concated_fixed_outlayers = df_concated_fixed_outlayers.diff().dropna()
#df_concated_fixed_outlayers = df_concated_fixed_outlayers.resample('7D').interpolate(method='linear') 
df_concated_fixed_outlayers = df_concated_fixed_outlayers.resample('14D').agg('mean') 












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
    marker_sizes = 500 
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
    
    ax.scatter(X,Y,Z, c=colors, s=marker_sizes[:num_points])
    ax.plot(X,Y,Z)
    
    
    plt.show()






import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam


FullCols = ChemCols_0_10_names
Dict_embeddings = {}
time_series_df = df_upsampled_normalized.copy()



window_length = 5  # Define the length of your window
n_features = 1  # Number of features (in this case, assuming univariate time series)


# Reshape data for LSTM input (samples, time steps, features)
def create_sequences(data, window_length):
    sequences = []
    for i in range(len(data) - window_length + 1):
        sequence = data[i:i + window_length]
        sequences.append(sequence)
    return np.array(sequences)


for col in FullCols:
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(time_series_df[[col]])

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
    Dict_embeddings[col] = embeddings_list



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
    ax.set_title(col)
    # Show the plot
    plt.show()
    
    
        





#Save embeddings as pickle
import pickle
#save amplified df as pickle to be read by the external process
with open(BaseFolder+'All_ccm1_embeddings.pickle', 'wb') as handle:
    pickle.dump(Dict_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)   




with open(BaseFolder + 'All_ccm1_embeddings.pickle', 'rb') as handle:
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


amplified_dfs = amplifyData(df_concated_fixed_outlayers, subSetLength=len(df_concated_fixed_outlayers)-10, jumpN=1)
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
DictCols = build_colsDict(df_concated_fixed_outlayers)

for i, vali in enumerate(amplified_dfs):
    vali.columns = [DictCols[i] for i in vali.columns]
    amplified_dfs[i] = vali



#save amplified df as pickle to be read by the external process
with open(BaseFolder+'ccm1_amplified_dfs.pickle', 'wb') as handle:
    pickle.dump(amplified_dfs, handle, protocol=pickle.HIGHEST_PROTOCOL)   

with open(BaseFolder+'ccm1_DictCols.pickle', 'wb') as handle:
    pickle.dump(DictCols, handle, protocol=pickle.HIGHEST_PROTOCOL)  

with open(BaseFolder+'ccm1_x1_x2_columns.pickle', 'wb') as handle:
    pickle.dump([FullCols, phytoCols_0_10_names], handle, protocol=pickle.HIGHEST_PROTOCOL)     


os.system('python '+BaseFolder+'ccm_multiproc_fnn.py '+ BaseFolder + ' ccm1_' )

with open(BaseFolder + 'All_' + 'ccm1_' + 'results.pickle', 'rb') as handle:
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




df_CausalFeatures.to_csv(BaseFolder+'CCM_CausalFeatures_results.csv')















