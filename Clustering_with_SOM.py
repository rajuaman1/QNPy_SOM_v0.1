import pandas as pd
import numpy as np
from minisom import MiniSom
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
import pickle
import math
import matplotlib.pyplot as plt
from tslearn.barycenters import dtw_barycenter_averaging
import os
import matplotlib
import plotly.graph_objects as go #Needed for Starburst plot

def load_padded_light_curves(path = './Padded_lc'):
    #Get the Padded Light Curves from the given path
    #Returns the ids and light curves
    quasar_ids = []
    padded_light_curves = []
    Folder = Path(path)
    #Getting the light curves from the path
    for file in tqdm(Folder.rglob('*.csv'),desc = 'Loading Padded Curves'):
        quasar_ids.append(file.name[:-4])
        light_curve = pd.read_csv(file)
        padded_light_curves.append(light_curve)
    return quasar_ids,padded_light_curves

def scale_curves(light_curves,what_scaler = 'standard'):
    #Scaling the curves from the choice of minmax, standard and robust
    scaler_dictionary = {'standard':StandardScaler(),'minmax':MinMaxScaler(),'robust':RobustScaler()}
    scaled_curves = []
    #Scaling each light curve
    for i in tqdm(range(len(light_curves)),desc = 'Scaling Curves'):
        mags_to_scale = pd.DataFrame(light_curves[i]['mag'])
        scaler = scaler_dictionary[what_scaler]
        scaled_curves.append(scaler.fit_transform(mags_to_scale))
        scaled_curves[i] = scaled_curves[i].reshape(len(mags_to_scale))
    return scaled_curves

def Train_SOM(scaled_curves,som_x = None,som_y = None,learning_rate = 0.1,sigma = 1.0,topology = 'rectangular',\
                neighborhood_function='gaussian',epochs = 50000,save_som = True,model_save_path = './',random_seed = 21,pca_init = True):
    #Initialize and train a SOM on the given data
    default_som_grid_length = math.ceil(math.sqrt(math.sqrt(len(scaled_curves))))
    if som_x is None and som_y is None:
        som_x = som_y = default_som_grid_length
    elif som_x is None or som_y is None:
        print('Please Provide both som_x and som_y or neither, going with the default values of the sqrt')
        som_x = som_y = default_som_grid_length
    som_model = MiniSom(som_x,som_y,len(scaled_curves[0]),learning_rate = learning_rate,sigma = sigma,\
                       topology = topology, neighborhood_function = neighborhood_function,random_seed=random_seed)
    if pca_init is True:
        som_model.pca_weights_init(scaled_curves)
    for i in tqdm(range(1),desc = 'Model Training'): #Stupid progress bar but fixes the verbose taking the whole screen
        som_model.train_random(scaled_curves,epochs)
    print('quantization error: {}'.format(som_model.quantization_error(scaled_curves)))
    if save_som:
        with open(model_save_path+'som_model.p', 'wb') as outfile:
            pickle.dump(som_model, outfile)
        print('Model Saved')
    return som_model

def Train_SOM_with_stats(scaled_curves,som_x = None,som_y = None,learning_rate = 0.1,sigma = 1.0,topology = 'rectangular',\
                neighborhood_function='gaussian',epochs = 50000,save_som = True,model_save_path = './',random_seed = 21,stat = 'q',\
                        pca_init = True,plot_frequency = 100):
    #Initialize and train a SOM on the given data
    default_som_grid_length = math.ceil(math.sqrt(math.sqrt(len(scaled_curves))))
    if som_x is None and som_y is None:
        som_x = som_y = default_som_grid_length
    elif som_x is None or som_y is None:
        print('Please Provide both som_x and som_y or neither, going with the default values of the sqrt')
        som_x = som_y = default_som_grid_length
    som_model = MiniSom(som_x,som_y,len(scaled_curves[0]),learning_rate = learning_rate,sigma = sigma,\
                       topology = topology, neighborhood_function = neighborhood_function,random_seed=random_seed)
    if pca_init is True:
        som_model.pca_weights_init(scaled_curves)
    max_iter = epochs
    q_error = []
    t_error = []
    indices_to_plot = []
    if stat == 'both':
        stat = 'qt'
    for i in tqdm(range(max_iter),desc = 'Evaluating SOM'):
        rand_i = np.random.randint(len(scaled_curves))
        som_model.update(scaled_curves[rand_i], som_model.winner(scaled_curves[rand_i]), i, max_iter)
        if (i % plot_frequency == 0 or i == len(scaled_curves)-1) and plot_training:
            indices_to_plot.append(i)
            if 'q' in stat:
                q_error.append(som_model.quantization_error(scaled_curves))
            if 't' in stat:
                t_error.append(som_model.topographic_error(scaled_curves))
    if save_som:
        with open(model_save_path+'som_model.p', 'wb') as outfile:
            pickle.dump(som_model, outfile)
        print('Model Saved')
    
    return som_model, q_error,t_error, indices_to_plot

def plot_training(training_metric_results,metric,plotting_frequency,indices_to_plot,figsize = (10,10),save_figs = True,fig_save_path = './'):
    #Plots the metric given (quantization error or topographic error) 
    plt.figure(figsize = figsize)
    plt.plot(indices_to_plot,training_metric_results)
    plt.ylabel(metric)
    plt.xlabel('iteration index')
    if save_figs:
        if 'Plots' not in os.listdir():
            os.makedirs(fig_save_path+'Plots')
        plt.savefig(fig_save_path+'Model_Training_'+metric+'.png')  
        
def Plot_SOM_Scaled_Average(som_model,scaled_curves,dba = True,figsize = (10,10),save_figs = False,fig_save_path = './'):
    #Plot the scaled light curves for each cluster as well as the averaged light curve of each cluster
    #Dynamic Barymetric Time Averaging (Cite the papers listed on GitHub Repo (https://github.com/fpetitjean/DBA))
    som_x,som_y = som_model.distance_map().shape
    win_map = som_model.win_map(scaled_curves)
    total = len(win_map)
    cols = int(np.sqrt(len(win_map)))
    rows = total//cols
    if total % cols != 0:
        rows += 1
    fig, axs = plt.subplots(rows,cols,figsize = figsize,layout="constrained")
    fig.suptitle('Clusters')
    count = 0
    for x in tqdm(range(som_x),desc = 'Creating Plots'):
        for y in range(som_y):
            cluster = (x,y)
            if cluster in win_map.keys():
                no_obj_in_cluster = 0
                for series in win_map[cluster]:
                    axs.flat[count].plot(series,c="gray",alpha=0.5)
                    no_obj_in_cluster += 1
            if dba is True:
                axs.flat[count].plot(dtw_barycenter_averaging(np.vstack(win_map[cluster])),c="blue")
            else:
                axs.flat[count].plot(np.mean(np.vstack(win_map[cluster]),axis=0),c="blue")
            axs.flat[count].set_title(f"Cluster {x*som_y+y+1}: {no_obj_in_cluster} curves")
            count += 1
    if save_figs:
        if 'Plots' not in os.listdir():
            os.makedirs(figs_save_path+'Plots')
        plt.savefig(figs_save_path+'Plots/Scaled_Averaged_Clusters.png')
    plt.show()
    
def SOM_Nodes_Map(som_model,figsize = (5,5),cmap = 'YlOrRd',save_figs = False,figs_save_path = './'):
    #Creates a heatmap of the SOM nodes
    plt.figure(figsize = figsize)
    plt.pcolor(som_model.distance_map().T, cmap=cmap,edgecolors='k')
    cbar = plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    if save_figs:
        if 'Plots' not in os.listdir():
            os.makedirs(figs_save_path+'Plots')
        plt.savefig(figs_save_path+'Plots/SOM_Nodes_Map.png')
    plt.show()
    
def Assign_Cluster_Labels(som_model,scaled_curves,ids):
#Creates a dataframe that has each cluster number for each id
    cluster_map = []
    som_y = som_model.distance_map().shape[1]
    for idx in tqdm(range(len(scaled_curves)),desc = 'Creating Dataframe'):
        winner_node = som_model.winner(scaled_curves[idx])
        cluster_map.append((ids[idx],winner_node[0]*som_y+winner_node[1]+1))
    clusters_df=pd.DataFrame(cluster_map,columns=["ID","Cluster"])
    return clusters_df

def SOM_Clusters_Histogram(cluster_map,color,save_figs = True,figs_save_path = './'):
    #Creates a histogram of the clusters
    cluster_map.value_counts('Cluster').plot(kind = 'bar',color = color)
    plt.ylabel('No of quasars')
    if save_figs:
        if 'Plots' not in os.listdir():
            os.makedirs(figs_save_path+'Plots')
        plt.savefig(figs_save_path+'Plots/Clusters_Histogram.png')
        
def findMin(x, y, umat):
    #Finds minimum node
    newxmin=max(0,x-1)
    newxmax=min(umat.shape[0],x+2)
    newymin=max(0,y-1)
    newymax=min(umat.shape[1],y+2)
    minx, miny = np.where(umat[newxmin:newxmax,newymin:newymax] == umat[newxmin:newxmax,newymin:newymax].min())
    return newxmin+minx[0], newymin+miny[0]

def findInternalNode(x, y, umat):
    #Finds node with internal minimum
    minx, miny = findMin(x,y,umat)
    if (minx == x and miny == y):
        cx = minx
        cy = miny
    else:
        cx,cy = findInternalNode(minx,miny,umat)
    return cx, cy
        
def Get_Gradient_Cluster(som):
    #Get the SOM Gradient Clusters
    cluster_centers = []
    cluster_pos  = []
    for row in np.arange(som.distance_map().shape[0]):
        for col in np.arange(som.distance_map().shape[1]):
            cx,cy = findInternalNode(row, col, som.distance_map().T)
            cluster_centers.append(np.array([cx,cy]))
            cluster_pos.append(np.array([row,col]))
    return np.array(cluster_centers),np.array(cluster_pos)

def Gradient_Cluster_Map(som,scaled_curves,ids):
    #Gets the clusters when created with gradients
    cluster_centers,cluster_pos = Get_Gradient_Cluster(som)
    cluster_numbers = np.arange(len(np.unique(cluster_centers,axis = 0)))
    unique_cluster_centers = np.unique(cluster_centers,axis = 0)
    cluster_numbers_map = []
    for i in range(len(scaled_curves)):
        winner_node = som.winner(scaled_curves[i])
        winner_node = np.array(winner_node)
        #Gets the central node where the winning cluster is in
        central_cluster = cluster_centers[np.where(np.isclose(cluster_pos,winner_node).sum(axis = 1) == 2)][0]
        cluster_number = cluster_numbers[np.where(np.isclose(unique_cluster_centers,central_cluster).sum(axis = 1) == 2)]
        cluster_numbers_map.append(cluster_number[0]+1)
    return pd.DataFrame({'ID':ids,'Cluster':cluster_numbers_map})

def matplotlib_cmap_to_plotly(cmap, entries):
    #Used for creating interactive plot
    h = 1.0/(entries-1)
    colorscale = []

    for k in range(entries):
        C = (np.array(cmap(k*h)[:3])*255)
        colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return colorscale

def plotStarburstMap(som):
    #Plots the Starburst Gradient Visualization of the clusters
    boner_rgb = []
    norm = matplotlib.colors.Normalize(vmin=0, vmax=255)
    bone_r_cmap = matplotlib.colormaps.get_cmap('bone_r')

    bone_r = matplotlib_cmap_to_plotly(bone_r_cmap, 255)

    layout = go.Layout(title='starburstMap')
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Heatmap(z=som.distance_map().T, colorscale=bone_r))
    shapes=[]
    for row in np.arange(som.distance_map().shape[0]):
        for col in np.arange(som.distance_map().shape[1]):
            cx,cy = findInternalNode(row, col, som.distance_map().T)
            shape=go.layout.Shape(
                    type="line",
                    x0=row,
                    y0=col,
                    x1=cx,
                    y1=cy,
                    line=dict(
                        color="Black",
                        width=1
                    )
                )
            shapes=np.append(shapes, shape)

    fig.update_layout(shapes=shapes.tolist(), 
        width=500,
        height=500) 
    fig.show()

def outliers_detection(clusters_df,som,scaled_curves,ids,outlier_percentage = 0.2):
    #Detects outliers that aren't quantized well as a percentage of the clusters
    quantization_errors = np.linalg.norm(som.quantization(scaled_curves) - scaled_curves, axis=1)
    error_treshold = np.percentile(quantization_errors, 
                               100*(1-outliers_percentage)+5)
    outlier_ids = np.array(ids)[quantization_errors>error_treshold]
    outlier_cluster = []
    for i in range(len(clus.ID)):
        if str(clus.ID[i]) in outlier_ids:
            outlier_cluster.append(clus.Cluster[i])
    #Plot the number of outliers per cluster
    plt.figure()
    plt.hist(clus['Cluster'],bins = len(np.unique(clus.Cluster))-1,alpha = 0.35,label = 'Total number of clusters',edgecolor = 'k')
    plt.hist(outlier_cluster,bins = len(np.unique(clus.Cluster))-1,alpha = 0.35,label = 'outliers',edgecolor = 'k')
    plt.xlabel('Cluster')
    plt.ylabel('No of Quasars')
    plt.legend()
    #Plot the treshold for quantization error
    plt.figure()
    plt.hist(quantization_errors,edgecolor = 'k',label = f'Threshold = {outlier_percentage}')
    plt.axvline(error_treshold, color='k', linestyle='--')
    plt.legend()
    plt.xlabel('Quantization Error')
    plt.ylabel('No of Quasars')