# QNPy_SOM
Implementation of Self Organizing Maps for use in QNPy

Modules that can be used in QNPy for clustering of light curves

The workflow is through 3 different notebooks (or if you prefer, you can use the corresponding Python Script with the same name). 

1. ['Pulling Data from LSST AGN DC'](Pulling_Data_from_LSST_AGN_DC.ipynb)

Pulls and formats data from the LSST AGN Data Challenge in order to match the formatting needed for QNPy (and the clustering)

Utilizes functions from module ['LSST_AGN_DC_Pull.py'](LSST_AGN_DC_Pull.py)


2. ['Clustering With SOM'](Clustering_with_SOM.ipynb)
   
Scales data and clusters it using Self Organizing Maps (from MiniSOM) and provides visualizations of the SOM

Utilizes functions from module ['Clustering_with_SOM.py'](Clustering_with_SOM.py)


3. ['Visualizing Cluster Features'](Visualizing_Cluster_Features.ipynb)
   
Provides visualization of the clustered light curves and some astronomical properties associated with each of them

Utilizes functions from module ['Visualizing_Clusters.py'](Visualizing_Clusters.py)

