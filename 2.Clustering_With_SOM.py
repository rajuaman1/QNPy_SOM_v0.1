# Variables to Set (Customize)
padded_lcs_path = './Padded_lc' #Where the padded light curves are stored
what_scaler = 'standard' #Scaler to use : 'minmax','standard','robust'
som_x = None #x dimension of SOM, if set to None, then please set som_y to None also
som_y = None #Refer to above, if both are None, then the value becomes sqrt(sqrt(len(padded_lcs)))
learning_rate = 0.1 #Learning Rate of SOM
sigma = 1.0 #Influence on Neighboring Nodes of SOM
topology = 'rectangular' #Topology of SOM (can use hexagonal also)
neighborhood_function='gaussian' #Neighborhood function
epochs = 50000 #Number of epochs to train SOM
save_som = True #Whether to save the SOM
model_save_path = './' #Where to save the SOM
visualize_SOM = False #Whether to plot metrics of the SOM (for now, heat map, average clusters and number of clusters)
dba = True #Whether to use dynamic barymetric average to visualize the average
figsize = (10,10) #Figure size of the plots
clusters_save_path = './' #Where to save the clusters map
random_seed = 21 #Random seed
cmap = 'YlOrRd' #Color map for heatmap
save_figs = True #Whether to save the plots
fig_save_path = './' #Where to save the plots
clusters_color = 'tab:blue' #The color of the clusters histogram
have_latex_installed = False #Will use science style for plot if true (Need scienceplots package also)


# Import 
from Clustering_with_SOM import load_padded_light_curves,scale_curves,Train_SOM,Assign_Cluster_Labels,\
                                Plot_SOM_Scaled_Average,SOM_Nodes_Map,SOM_Clusters_Histogram
import matplotlib.pyplot as plt
if have_latex_installed:
    import scienceplots
    plt.style.use('science')

# Run To Train SOM

ids,padded_lcs = load_padded_light_curves(padded_lcs_path)
scaled_curves = scale_curves(padded_lcs,what_scaler)
som_model = Train_SOM(scaled_curves,som_x,som_y,learning_rate,sigma,topology,neighborhood_function,epochs,save_som,\
                      model_save_path,random_seed)
clusters_df = Assign_Cluster_Labels(som_model,scaled_curves,ids)
clusters_df.to_csv(f'{clusters_save_path}Clusters_Map.csv',index = False)
print('Clusters Saved')
if visualize_SOM:
    Plot_SOM_Scaled_Average(som_model,scaled_curves,dba,figsize,save_figs,fig_save_path)
    SOM_Nodes_Map(som_model,figsize,cmap,save_figs,fig_save_path)
    SOM_Clusters_Histogram(clusters_df,clusters_color,save_figs,fig_save_path)

