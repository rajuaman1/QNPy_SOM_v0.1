#Import modules
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm

def Get_Light_Curve(object_id,td_objects,fs_gp,turn_off_warning = True):
    #Gets the Light Curve if it exists
    if object_id in td_objects.index:
        demo_lc = fs_gp.get_group(object_id)
        demo_lc_bands = {}
        for i, b in enumerate('ugriz'):
            demo_lc_bands[b] = demo_lc[demo_lc['filter'] == i]
        return demo_lc_bands,1
    else:
        if turn_off_warning is False:
            print('No Light Curve Data')
        return 0,0

def get_all_data_quasars(quasars,td_objects,fs_gp):
    #Getting the agns that actually have light curve data
    data_count = 0
    no_data_count = 0

    all_quasar_light_curves = []

    for quasar_id in tqdm(quasars.index,desc = 'Selecting Quasar Light Curves'):
        light_curve,exists = Get_Light_Curve(quasar_id,td_objects,fs_gp)
        if exists == 1:
            all_quasar_light_curves.append(light_curve)
    return all_quasar_light_curves

def Select_Cadence_and_Features(light_curves,cadence = 100,features = ['objectId','mjd','psMag','psMagErr'],\
                                selected_filters = 'ugriz'):
    #Selecting certain features from all light curves with cadences greater than a certain amount (can also give 'all')
    selected_light_curves = []
    for light_curve in tqdm(light_curves,desc = f'Filtering for Cadence and Features',leave = True,position = 0):
        flags = 0
        for filter_no,Filter in enumerate(selected_filters):
            if 'psMag' in features:
                    light_curve[Filter] = light_curve[Filter].dropna(subset = ['psMag'])
            if len(light_curve[Filter]) >= cadence:
                flags += 1
                light_curve[Filter]['objectId'] = pd.to_numeric(light_curve[Filter]['objectId'])
        if flags == len(selected_filters):
            selected_light_curves.append(light_curve)
    return selected_light_curves

def ReShape_Light_Curves(light_curves,selected_filters = 'ugriz'):
    #Reshape the light curves so that they only have the columns and shape we want(mjd, psMag and psMagErr)
    light_curves_copy = copy.deepcopy(light_curves)
    for i,light_curve in tqdm(enumerate(light_curves_copy),desc = 'Formatting Output',position = 0):
        for Filter in selected_filters:
            df = light_curve[Filter][['mjd','psMag','psMagErr']]
            df.columns = ['mjd','mag','magerr']
            df = df.drop_duplicates(subset = ['mjd'])
            df = df.sort_values('mjd')
            df = df.reset_index(drop = True)
            light_curves_copy[i][Filter] = df
    return light_curves_copy

def Pad_Light_Curves(light_curves,selected_filters = 'ugriz',minimum_length = 100):
    #Pads the light curve from the end to the length of the longest one using the mean value
    light_curves_copy = copy.deepcopy(light_curves)
    #Getting the longest light curve
    longest = minimum_length
    for light_curve in light_curves_copy:
        for Filter in selected_filters:
            if len(light_curve[Filter])>longest:
                longest = len(light_curve[Filter])
                longest_series = light_curve[Filter]
    #Reindexing the curves less than the longest one
    for i,light_curve in tqdm(enumerate(light_curves_copy),desc = 'Padding Light Curves'):
        for Filter in selected_filters:
            if len(light_curve[Filter]) != longest:
                fill_number = longest - len(light_curve[Filter])
                new_rows = pd.DataFrame({'mjd':list(np.linspace(light_curve[Filter]['mjd'].iloc[-1]+0.2,light_curve[Filter]['mjd'].iloc[-1]+0.2*(fill_number+1),fill_number)),
                'mag':[light_curve[Filter]['mag'].mean()]*fill_number,
                'magerr':[light_curve[Filter]['magerr'].mean()]*fill_number})
                new_rows = pd.DataFrame(new_rows)
                light_curves_copy[i][Filter] = pd.concat((light_curve[Filter],new_rows))
    return light_curves_copy