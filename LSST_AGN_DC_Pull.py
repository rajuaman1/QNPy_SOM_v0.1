#Import modules
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm

def Get_Light_Curve(object_id,td_objects,fs_gp,Filter = 'u',turn_off_warning = True):
    #Gets the Light Curve if it exists
    if object_id in td_objects.index:
        demo_lc = fs_gp.get_group(object_id)
        demo_lc_bands = {}
        for i, b in enumerate('ugriz'):
            demo_lc_bands[b] = demo_lc[demo_lc['filter'] == i]
        return demo_lc_bands[Filter],1
    else:
        if turn_off_warning is False:
            print('No Light Curve Data')
        return 0,0

def get_all_data_quasars(quasars,td_objects,fs_gp,Filter = 'u'):
    #Getting the agns that actually have light curve data
    data_count = 0
    no_data_count = 0

    all_quasar_light_curves = []

    for quasar_id in tqdm(quasars.index,desc = 'Selecting Quasar Light Curves'):
        light_curve,exists = Get_Light_Curve(quasar_id,td_objects,fs_gp,Filter)
        if exists == 1:
            all_quasar_light_curves.append(light_curve)
    return all_quasar_light_curves

def Select_Cadence_and_Features(light_curves,cadence = 100,features = ['objectId','mjd','psMag','psMagErr']):
    #Selecting certain features from all light curves with cadences greater than a certain amount (can also give 'all')
    selected_light_curves = []
    for light_curve in tqdm(light_curves,desc = 'Filtering for Cadence and Features'):
        if len(light_curve) >= cadence:
            light_curve['objectId'] = pd.to_numeric(light_curve['objectId'])
            if 'psMag' in features:
                light_curve = light_curve.dropna(subset = ['psMag'])
            if len(light_curve)>= cadence:
                if features == 'all':
                    selected_light_curves.append(light_curve[features])
                else:
                    selected_light_curves.append(light_curve[features])
    return selected_light_curves

def ReShape_Light_Curves(light_curves):
    #Reshape the light curves so that they only have the columns and shape we want(mjd, psMag and psMagErr)
    light_curves_copy = copy.deepcopy(light_curves)
    new_curves = []
    for i in tqdm(range(len(light_curves_copy)),desc = 'Formatting Output'):
        y1=np.vstack([light_curves_copy[i]["mjd"], light_curves_copy[i]["psMag"], light_curves_copy[i]["psMagErr"]])
        df = pd.DataFrame(y1.T, columns = ['mjd','mag','magerr'])
        df = df.drop_duplicates(subset = ['mjd'])
        df = df.sort_values('mjd')
        df = df.reset_index(drop = True)
        new_curves.append(df)
    return new_curves

def Pad_Light_Curves(light_curves):
    #Pads the light curve from the end to the length of the longest one using the mean value
    light_curves_copy = copy.deepcopy(light_curves)
    longest = 100
    #Getting the longest light curve
    for light_curve in light_curves_copy:
        if len(light_curve)>longest:
            longest = len(light_curve)
            longest_series = light_curve
    #Reindexing the curves less than the longest one
    problem_index = []
    for i,light_curve in tqdm(enumerate(light_curves_copy),desc = 'Padding Light Curves'):
        #light_curves_copy[i].index = longest_series.index[:len(light_curve)] #Making sure that the indexes are the same
        if len(light_curve) != longest:
            problem_index.append(i)
            fill_number = longest - len(light_curve)
            new_rows = pd.DataFrame({'mjd':list(np.linspace(light_curve['mjd'].iloc[-1]+0.2,light_curve['mjd'].iloc[-1]+0.2*(fill_number+1),fill_number)),
            'mag':[light_curve['mag'].mean()]*fill_number,
            'magerr':[light_curve['magerr'].mean()]*fill_number})
            new_rows = pd.DataFrame(new_rows)
            light_curves_copy[i] = pd.concat((light_curve,new_rows))
    return light_curves_copy