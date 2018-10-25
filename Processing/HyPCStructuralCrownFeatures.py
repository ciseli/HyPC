"""
HyPC structural crown features module

Description:
    Computes geometric based features for each crown segment

Required inputs:
    -HyPC point cloud file

Created by: Christopher Iseli
Last Modified: 20/10/2018 (Christopher Iseli)
"""

import pickle, math, time
import numpy as np
import HyPC
from HyPC import hypc

np.set_printoptions(suppress=True)

## ===== Settings ====##

inputFname = "pointCloud_sample.hypc"
outputFname = "pointCloud_sample1.hypc"

##--------------------##


with open(inputFname, "rb") as f:
    pointCloud = pickle.load(f)

num_segments = pointCloud.df['segment_id'].nunique()

try:
    pointCloud.segments
except AttributeError:
    pointCloud.segments = {}

def layerDensity(z_bins):
    densities = np.empty(20)
    for i in range(20):
        idxs = np.argwhere(z_bins==i)
        densities[i]=len(idxs)
    return densities

def layer_diameter(z_bins):
    diameters = np.empty(20)
    for i in range(20):
        idxs = np.argwhere(bins_z==i)
        if len(idxs)>1:
            x_diameter = np.max(crown_points[idxs, 0])-np.min(crown_points[idxs, 0])
            y_diameter = np.max(crown_points[idxs, 1])-np.min(crown_points[idxs, 1])
            diameters[i]=np.mean([x_diameter,y_diameter])
        else:
            diameters[i] = 0
    return diameters

def top_layer_spectra(z_bins):

    idxs = np.argwhere(z_bins==19).T[0]
    spectra = crown_attributes.iloc[idxs, :21]
    mean_spectra = np.mean(spectra, axis=0)/2**16
    return mean_spectra

for i in range(num_segments-1):
    print("segment: ",i+1)
    segment_idxs = pointCloud.df.loc[pointCloud.df['segment_id'] == i+1].index.values
    crown_points =  pointCloud.kdTree.data[segment_idxs]
    crown_attributes =  pointCloud.df.iloc[segment_idxs]
    #==== Split crown into 5% height layers ====#
    # Get max hight of crown
    max_elevation = np.max(crown_points[:,2])
    min_elevation = np.min(crown_points[:,2])
    crown_height = max_elevation-min_elevation
    # compute 5% layer splits
    layer_height = crown_height/20
    layer_mins = np.linspace(min_elevation,max_elevation,num=21)[1:]
    # assign points to layers based on height
    bins_z = np.searchsorted(layer_mins, crown_points[:, 2])

    layer_densities = layerDensity(bins_z)
    layer_diameters = layer_diameter(bins_z)
    top_layer_mean =top_layer_spectra(bins_z)

    if  str(i+1) not in pointCloud.segments:
        pointCloud.segments[str(i + 1)] = {}
    pointCloud.segments[str(i + 1)]['densities'] = layer_densities
    pointCloud.segments[str(i + 1)]['diameters'] = layer_diameters
    pointCloud.segments[str(i + 1)]['top_spectra'] = top_layer_mean

HyPC.saveHyPC(pointCloud,outputFname)

print("Complete")