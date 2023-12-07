import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path 
from tqdm import tqdm
from matplotlib.colors import Normalize
from tslearn.barycenters import dtw_barycenter_averaging
from astropy.cosmology import FlatLambdaCDM
import seaborn as sns
from itertools import combinations
from scipy import stats
import os
import matplotlib

def load_cluster_map_and_unpadded_lcs(cluster_map_load_path = './',unpadded_lcs_load_path = './Light_Curves/',\
                                      have_redshifts = True,redshifts_load_path = './'):
    #Loads the cluster maps and unpadded lcs
    print('Loading Light Curves')
    cluster_map = pd.read_csv(cluster_map_load_path+'Clusters_Map.csv')
    #Loading the light curves
    quasar_ids = []
    light_curves = []
    Folder = Path(unpadded_lcs_load_path)
    #Getting the light curves from the path
    for file in tqdm(Folder.rglob('*.csv'),desc = 'Loading Curves'):
        quasar_ids.append(file.name[:-4])
        light_curve = pd.read_csv(file)
        light_curves.append(light_curve)
    #Getting Redshifts if they exist
    if have_redshifts:
        print('Loading Redshifts')
        redshifts = pd.read_csv(redshifts_load_path+'Redshift_Map.csv')
    else:
        print('No Redshifts')
        redshifts = []
    return cluster_map,quasar_ids,light_curves,redshifts

def create_dir_save_plot(path,plot_name):
    #Create a folder if it doesn't see it 
    if 'Plots' not in os.listdir():
            os.makedirs(path+'Plots')
    plt.savefig(path+'Plots/'+plot_name+'.png')

def tolerant_mean(arrs):
    #Tolerant means of arrays
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)

def get_best_grid(number_of_objects):
    cols = int(np.sqrt(number_of_objects))
    rows = number_of_objects//cols
    if number_of_objects % cols != 0:
        rows += 1
    return rows,cols

def Averaging_Clusters(chosen_cluster,cluster_map,lcs,plot = True,dba = True):
  #Either plots each light curve with its errors and the average or just returns the average
    x = []
    x_arrs = []
    y = []
    y_arrs = []
    err = []
    ids = []
    all_curves = []
    #Getting the data to plot for the cluster
    for quasar_id,cluster,i in zip(cluster_map.ID,cluster_map.Cluster,range(len(cluster_map))):
        if cluster == chosen_cluster:
            light_curve = lcs[i]
            light_curve = light_curve.dropna(subset = ['mag'])
            light_curve = light_curve.sort_values(by = 'mjd')
            all_curves.append(light_curve)
            times = light_curve.mjd - light_curve.mjd.min()
            x.append(times)
            x_arrs.append(times.to_numpy()) 
            y.append(light_curve.mag)
            y_arrs.append(light_curve.mag.to_numpy())
            err.append(light_curve.magerr)
    cmap = plt.cm.prism
    norm = Normalize(vmin=1, vmax=len(x))
    if dba: #Need to build support for mean (even though the light curves are not same length)
        average_x = dtw_barycenter_averaging(x)
        average_y = dtw_barycenter_averaging(y)
    else:
        average_x = tolerant_mean(x_arrs)[0]
        average_y = tolerant_mean(y_arrs)[0]
    #Plotting a scatter plot and line plot
    if plot is True:
        fig,(ax1,ax2) = plt.subplots(1,2,sharey= True,figsize = (6,7))
        for i in range(len(x)):
            ax1.errorbar(x[i], y[i]*(i+1), err[i], fmt='.',color = cmap(int(i)),alpha = 0.1)
            ax2.plot(x[i],y[i]*(i+1),alpha = 0.5,c = cmap(int(i)))
        ax1.invert_yaxis()
        print('Length of Cluster: '+str(len(x)))
        plt.figure()
        for i in range(len(x)):
            plt.plot(x[i],y[i],alpha = 0.5,c = 'grey')
        plt.plot(average_x,average_y,label = 'Averaged Curves')
        plt.gca().invert_yaxis()
    else:
        return average_x,average_y,x,y,len(x)

def Plot_All_Clusters(cluster_map,lcs,color = 'tab:blue',dba = True,figsize = (10,10),save_figs = True,figs_save_path = './'):
  #Plots all the clusters on a magnitude plot with the average
  #Getting the shape of the subplots
    clusters = cluster_map.value_counts('Cluster').index.to_numpy()
    total = len(clusters)
    cols = int(np.sqrt(len(clusters)))
    rows = total//cols
    if total % cols != 0:
        rows += 1
    fig, axs = plt.subplots(rows,cols,figsize=figsize,layout="constrained",sharey = True)
    fig.suptitle('Clusters')
    #Getting the values to plot
    x_axis = []
    y_axis = []
    for i in tqdm(range(len(clusters)),desc = 'Plotting Averaged Clusters'):
        x,y,back_x,back_y,no = Averaging_Clusters(clusters[i],cluster_map,lcs,plot = False,dba = dba)
        for j in range(no):
            axs.flat[i].plot(back_x[j],back_y[j],color = 'gray',alpha = 0.5)
            axs.flat[i].plot(x,y,color = color)
            axs.flat[i].set_title(f'Cluster {clusters[i]}, {no} curves')
            axs.flat[i].set_xlabel('Days')
            axs.flat[i].set_ylabel('Magnitude')
            axs.flat[i].invert_yaxis()
    axs[0,0].invert_yaxis()
    if save_figs:
        create_dir_save_plot(figs_save_path,'SOM_Nodes_Map')
    plt.show()

def get_redshifts(redshifts_map):
  #Getting all the redshifts of the selected quasars
    redshifts = redshifts_map.z.to_list()
    return redshifts

def get_fvars(lcs):
  #Calculation of the F_var
    fvars = []
    for quasar in lcs:
        N = len(quasar)
        meanmag = quasar['mag'].mean()
        s2 = (1/(N-1))*np.sum((quasar['mag']-meanmag)**2)
        erm = np.mean(quasar['magerr']**2)
        f_var = np.sqrt((s2-erm)/(np.mean(quasar["mag"])**2))
        if np.isnan(f_var):
            f_var = np.nan
        fvars.append(f_var)
    return fvars

def get_luminosities_and_masses(lcs, redshifts_map, H0 = 67.4, Om0 = 0.315):
    #Gets the luminosity and absolute magnitudes of the given quasars

    #Calculating luminosities
    c = 299792.458
    Tcmb0 = 2.725
    #Cosmological Model Chosen
    cosmo = FlatLambdaCDM(H0,Om0,Tcmb0)
    redshifts = get_redshifts(redshifts_map)

    distances=[]
    #Getting the distances from the redshifts
    for i in range(len(redshifts)):
        distances.append(cosmo.comoving_distance(redshifts[i]).value)
    #Converting distance to luminosities and correcting the magnitudes
    F0=3.75079e-09
    lambeff=3608.4
    Mpc_to_cm = 3.086e+24
    luminosity=[]
    absolmag=[]
    for i,quasar in enumerate(lcs):
        meanmag= quasar['mag'].mean()
        const=4*np.pi*((distances[i]*Mpc_to_cm)**2)*lambeff*F0
        luminosity.append(const*np.power(10, -meanmag/2.5))
        Kcorr=-2.5*(1-0.5)*np.log(1+redshifts[i])
        absolmag.append(meanmag-5*np.log10(distances[i])-25-Kcorr)
    #Using the corrected magnitudes to calculate black hole masses
    mu=2.0-0.27*np.asarray(absolmag)
    sigma=np.abs(0.580+0.11*np.asarray(absolmag))
    #Random Sampling the Masses from Distribution
    Log_Mass=np.zeros(len(mu))
    for i in range(len(mu)):
        k1=np.random.normal(mu[i],sigma[i],10).mean()
        Log_Mass[i] = k1

    return np.log10(luminosity),Log_Mass

def Cluster_Properties(cluster_map,selected_cluster,lcs,redshifts_map = None,plot = True,return_values = False,\
                       the_property = 'all',save_figs = True,figs_save_path = './'):
    if redshifts_map is None and the_property != 'Fvar':
        print('Need Redshifts to plot selected property')
        return
    all_quasar_ids = cluster_map.ID
    if selected_cluster == 'all':
        selected_quasar_ids = all_quasar_ids.to_list()
    else:
        selected_quasar_ids = cluster_map.ID[cluster_map.Cluster == selected_cluster].to_list()
    quasar_light_curves = []
    for i in range(len(lcs)):
        if all_quasar_ids[i] in selected_quasar_ids:
              quasar_light_curves.append(lcs[i])
    return_list = [[np.nan]*len(selected_quasar_ids)]*4
    if the_property == 'all':
        the_property = 'zFvarLumMass'
    if 'z' in the_property:
        redshifts = get_redshifts(redshifts_map[redshifts_map.ID.isin(selected_quasar_ids)])
        return_list[0] = redshifts
    if 'Fvar' in the_property:
        fvars = get_fvars(quasar_light_curves)
        return_list[1] = fvars
    if 'Lum' in the_property or 'Mass' in the_property:
        log_luminosities,log_masses = get_luminosities_and_masses(quasar_light_curves,redshifts_map[redshifts_map.ID.isin(selected_quasar_ids)])
        return_list[2] = log_luminosities
        return_list[3] = log_masses
    if plot is True:
        new_dataframe = pd.DataFrame({'z':redshifts,r'$F_{var}$':fvars,r'$log_{10}{L[erg s^{-1}]}$':log_luminosities,r'$log_{10}{M[M_{\odot}]}$':log_masses})
        plt.figure()
        sns.pairplot(new_dataframe,corner = True)
        plt.minorticks_off()
        if save_figs:
            create_dir_save_plot(figs_save_path,f'Cluster_{selected_cluster}_Properties')
    if return_values is True:
        return return_list

def Cluster_Properties_Comparison(cluster_map,lcs,redshifts_map,the_property = 'Fvar',show_names = False,color = '#1f77b4',\
                                  figsize = (15,15),save_figs = True,figs_save_path = './'):
  #Plots a histogram for one of the 4 different properties that we see in the clusters
    if show_names is True: #If you forget the keywords for the different properties
        print('Choose from Fvar, Lum, Mass, or z')
    property_to_num_dict = {'z':0,'Fvar':1,'Lum':2,'Mass':3}
    property_to_label_dict = {'z':'z','Fvar':'$F_{var}$','Lum':r'$log_{10}{L[erg s^{-1}]}$','Mass':r'$log_{10}{M[M_{\odot}]}$'}
    rows,cols = get_best_grid(len(cluster_map.value_counts('Cluster')))
    fig,axs = plt.subplots(rows,cols,figsize = figsize,layout = 'constrained')
    count = 0
    #Setting the x scale for all the clusters
    properties = Cluster_Properties(cluster_map,'all',lcs,redshifts_map,plot = False,return_values = True,the_property = the_property)[property_to_num_dict[the_property]]
    min_x = min(properties)
    max_x = max(properties)
    bins = np.linspace(min_x,max_x,10)
    plt.setp(axs, xlim=(min_x,max_x))
  #Plotting the different subclusters
    for i in tqdm(range(rows),desc = 'Plotting '+the_property+' Distribution'):
        for j in range(cols):
            plotting_values = np.array(Cluster_Properties(cluster_map,count+1,lcs,redshifts_map,the_property = the_property,plot = False,return_values = True)[property_to_num_dict[the_property]])
            plotting_values = plotting_values[np.isfinite(plotting_values)]
            #Splitting it so we can plot the log dist of the luminosities
            counts,bins = np.histogram(plotting_values,bins = bins)
            axs[i][j].hist(plotting_values,bins = bins,color = color,edgecolor='black',linewidth=1.2)
            axs[i][j].set_title(f'Cluster {count+1}, {len(plotting_values)} curves')
            axs[i][j].set_ylabel('Number of Curves')
            axs[i][j].set_xlabel(property_to_label_dict[the_property])
            count += 1
    if save_figs:
        create_dir_save_plot(figs_save_path,the_property+'_Distribution_Plot')

def SFplus(magnitudes):
    #Calculate the S+ Function
    combs=combinations(magnitudes,2)
    sf_vals=[]
    for x,y in combs:
        if x-y>0:
            sf_vals.append((x-y)**2)
    #sfplus=np.sqrt(np.mean(sf_vals))
    sfplus = np.mean(sf_vals)
    return sfplus

def SFminus(magnitudes):
  #Calculate the S- Function
    combs=combinations(magnitudes,2)
    sf_vals=[]
    for x,y in combs:
        if x-y<0:
            sf_vals.append((x-y)**2)
    #sfmin=np.sqrt(np.mean(sf_vals))
    sfmin = np.mean(sf_vals)
    return sfmin

def SF(magnitudes):
  #Calculate the S- Function
    combs=combinations(magnitudes,2)
    sf_vals=[]
    for x,y in combs:
        if x-y>0:
            sf_vals.append((x-y)**2)
    #sf=np.sqrt(np.mean(sf_vals))
    sf = np.mean(sf_vals)
    return sf

def Structure_Function(cluster_map,selected_cluster,lcs,bins,save_figs = True,figs_save_path = './'):
    #Create the structure function for a given cluster
    #Need to include bootstrapping 
    all_quasar_ids = cluster_map.ID
    if selected_cluster == 'all':
        selected_quasar_ids = all_quasar_ids.to_list()
    else:
        selected_quasar_ids = cluster_map.ID[cluster_map.Cluster == selected_cluster].to_list()
    quasar_light_curves = []
    for i in range(len(all_quasar_ids)):
        if all_quasar_ids[i] in selected_quasar_ids:
            quasar_light_curves.append(lcs[i])
  #Putting all the magnitudes and times together
    mag_composite=[]
    time_composite=[]
    for light_curve in quasar_light_curves:
        mag_composite=mag_composite+light_curve["mag"].to_list()
        time_composite=time_composite+ light_curve["mjd"].to_list()
    time_composite_zeroed = time_composite - np.min(time_composite)
    #Calculating the S+, S- and S for all bins
    betaplus, timeplus,xx=stats.binned_statistic(time_composite_zeroed, mag_composite, statistic=SFplus, bins=bins, range=None)
    betamin, timemin,xx=stats.binned_statistic(time_composite_zeroed, mag_composite, statistic=SFminus, bins=bins, range=None)
    beta, time,xx=stats.binned_statistic(time_composite_zeroed, mag_composite, statistic=SF, bins=bins, range=None)

    #Calculating the normalizing function
    bbeta=((betaplus)-(betamin))/(beta+0.01)

    #Plotting S+ and S-
    plt.figure()
    plt.scatter(timeplus[:-1],betaplus,marker = 'v',label = r'$S_+$ observed QSO',c = 'orange')
    plt.plot(timeplus[:-1],betaplus,c = 'orange',linestyle = '--')
    plt.scatter(timemin[:-1],betamin,marker = '^',label = r'$S_-$ observed QSO',c = 'b')
    plt.plot(timemin[:-1],betamin,c = 'b',linestyle = '--')
    plt.xlabel(r'$\tau[day]$')
    plt.ylabel(r'$S_+\cup S_-$')
    if save_figs:
        create_dir_save_plot(figs_save_path,f'Cluster_{selected_cluster}_S+&S-_Plot')
    plt.legend()
    #Plotting the normalized difference
    plt.figure()
    plt.scatter(time[:-1],bbeta,label = r'$\beta$ Observed QSO',c = 'black',s = 4)
    plt.plot(time[:-1],bbeta,linestyle = '--')
    plt.xlabel(r'$\tau[day]$')
    plt.ylabel(r'$\beta = \frac{S_+ - S_-}{S}$')
    if save_figs:
        create_dir_save_plot(figs_save_path,f'Cluster_{selected_cluster}_S+S-_Difference_Plot')
    #plt.legend()
    #Plotting the structure function evolution
    plt.figure()
    plt.scatter(time[:-1],beta,label = r'SF Observed QSO',c = 'black',s = 4)
    plt.plot(time[:-1],beta,linestyle = '--')
    plt.xlabel(r'$\tau[day]$')
    plt.ylabel(r'Structure Function')
    if save_figs:
        create_dir_save_plot(figs_save_path,f'Cluster_{selected_cluster}_Structure_Function_Plot')
