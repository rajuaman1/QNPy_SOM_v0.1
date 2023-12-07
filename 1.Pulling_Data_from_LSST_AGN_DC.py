#Variables to use (EDIT THIS TO CUSTOMIZE)
save_path = './' #Save your files here (Into two folders that are called Light_Curves and Padded_lcs) 
delete_prior_files = True #Delete previous files called Light_Curves and Padded_lcs
cadence = 100 #Minimum cadence quasars to use 
Filter = 'u' #Which filter do you want to use (ugriz)


#Import Statements

#Use the LSST_AGN_DC_Pull Script
from LSST_AGN_DC_Pull import get_all_data_quasars,Select_Cadence_and_Features,ReShape_Light_Curves,Pad_Light_Curves
import pandas as pd
import numpy as np
import os
from shutil import rmtree
from tqdm import tqdm
from time import time


# ## Loading the Data from the Site
#Load the forced_source table with the light curves and the object table with the attributes
#This is necessary because it contains the quasar classification and the redshift data that is important for later
forced_source_url = 'https://zenodo.org/records/6878414/files/ForcedSourceTable.parquet'
object_url = 'https://zenodo.org/records/6878414/files/ObjectTable.parquet'

#Now loading
start_time = time()
print('Loading LSST AGN Data from Site...')
object_df = pd.read_parquet(object_url)
fs_df = pd.read_parquet(forced_source_url)
print('Data Loaded in {}s'.format(time()-start_time))


# Processing Input Tables

#Saving all the quasar and their attributes
quasars = object_df[object_df['class'] == 'Qso']

# groupby forcedsource table by objectid
fs_gp = fs_df.groupby('objectId')

#Dropping Objects that don't have periodic data
lc_cols = [col for col in object_df.columns if 'Periodic' in col]
td_objects = object_df.dropna(subset=lc_cols, how='all').copy()

#Get all the quasar data
all_quasars_light_curves = get_all_data_quasars(quasars,td_objects,fs_gp,Filter)

# Properties of Selected Cadence Quasars

#Selecting the ones with 100 cadences and getting their magnitudes, errors, and observation times
selected_quasar_light_curves = Select_Cadence_and_Features(all_quasars_light_curves,cadence)

#Getting the ids associated with each quasar
selected_quasar_ids = []
for i in selected_quasar_light_curves:
    selected_quasar_ids.append(i.objectId.iloc[0])

#Getting the redshifts of these quasars
redshifts = []
for quasar_id in selected_quasar_ids:
    z = quasars[quasars.index == str(quasar_id)].z[0]
    redshifts.append(z)
redshifts_map = pd.DataFrame({'ID':selected_quasar_ids,'z':redshifts})
redshifts_map.to_csv(save_path+'Redshift_Map.csv',index = False)


# Processing the labels and shape of data and Creating Padding to Homogenize Length

#Reshaping and Homogenzing the light curves
reshaped_curves = ReShape_Light_Curves(selected_quasar_light_curves)
processed_curves = Pad_Light_Curves(reshaped_curves)


#Saving Files

#Create the necessary folders and deleting old files if they exist
#WARNING: Deletes and creates a new folder
for folder in ['Light_Curves','Padded_lc']:
    if delete_prior_files:
        if folder in os.listdir():
            rmtree(save_path+folder)
    if folder not in os.listdir():
        os.makedirs(save_path+folder)

#Save the formatted light curves    
for i in tqdm(range(len(processed_curves)),desc = 'Saving Light Curves'):
    reshaped_curves[i].to_csv(f'{save_path}Light_Curves/{str(selected_quasar_ids[i])}.csv',index = False)
    processed_curves[i].to_csv(f'{save_path}Padded_lc/{str(selected_quasar_ids[i])}.csv',index = False)
print('Done..')
