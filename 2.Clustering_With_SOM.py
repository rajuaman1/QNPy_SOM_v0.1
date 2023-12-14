# Variables to Set (Customize)
padded_lcs_path = './Padded_lc' #Where the padded light curves are stored
what_scaler = 'standard' #Scaler to use : 'minmax','standard','robust'
som_x = None #x dimension of SOM, if set to None, then please set som_y to None also
som_y = None #Refer to above, if both are None, then the value becomes sqrt(sqrt(len(padded_lcs)))
learning_rate = 0.1 #Learning Rate of SOM
sigma = 1 #Influence on Neighboring Nodes of SOM
topology = 'rectangular' #Topology of SOM (can use hexagonal also)
neighborhood_function='gaussian' #Neighborhood function
epochs = 50000 #Number of epochs to train SOM
save_som = False #Whether to save the SOM
model_save_path = './' #Where to save the SOM
visualize_SOM = True #Whether to plot metrics of the SOM (for now, heat map, average clusters and number of clusters)
dba = True #Whether to use dynamic barymetric average to visualize the average
figsize = (10,10) #Figure size of the plots
clusters_save_path = './' #Where to save the clusters map
random_seed = 42 #Random seed
cmap = 'YlOrRd' #Color map for heatmap
save_figs = True #Whether to save the plots
fig_save_path = './' ##Where to save the plots (Note that the program creates a folder called Plots in the directory)
clusters_color = 'tab:blue' #The color of the clusters histogram
have_science_plots = True #Will use science style for plot if true (Need scienceplots package also)
pca_init = True #Initialized weights by pca
use_gradient_cluster = True #Whether to use the gradient clustering algorithm on the SOM
plot_frequency = 100 #How frequent to sample errors for model training
stat = 'both' #You can set it to just quantization error or topographical error (with just q or t)


from Clustering_with_SOM import load_padded_light_curves,scale_curves,Train_SOM,Assign_Cluster_Labels,\
                                Plot_SOM_Scaled_Average,SOM_Nodes_Map,SOM_Clusters_Histogram,Train_SOM_with_stats,\
                                Gradient_Cluster_Map,plotStarburstMap,outliers_detection,plot_training
if have_science_plots:
    import scienceplots
    plt.style.use(['science','no-latex'])

# Run To Train SOM

ids,padded_lcs = load_padded_light_curves(padded_lcs_path)
scaled_curves = scale_curves(padded_lcs,what_scaler)
som_model, q_error,t_error, indices_to_plot = Train_SOM_with_stats(scaled_curves,som_x,som_y,learning_rate,sigma,topology,neighborhood_function,epochs,save_som,\
                      model_save_path,random_seed,stat,pca_init,plot_frequency)
if use_gradient_cluster:
    clusters_df = Gradient_Cluster_Map(som_model,scaled_curves,ids)
else:
    clusters_df = Assign_Cluster_Labels(som_model,scaled_curves,ids)
clusters_df.to_csv(f'{clusters_save_path}Clusters_Map.csv',index = False)
print('Clusters Saved')
if visualize_SOM:
    plot_training(q_error,'Quantization Error',plot_frequency,indices_to_plot,figsize,save_figs,fig_save_path)
    plot_training(t_error,'Topographic Error',plot_frequency,indices_to_plot,figsize,save_figs,fig_save_path)
    Plot_SOM_Scaled_Average(som_model,scaled_curves,dba,figsize,save_figs,fig_save_path)
    SOM_Nodes_Map(som_model,figsize,cmap,save_figs,fig_save_path)
    SOM_Clusters_Histogram(clusters_df,clusters_color,save_figs,fig_save_path)

