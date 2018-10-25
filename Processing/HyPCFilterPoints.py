"""
HyPC point filtering module

Description:
    Filter HyPC point cloud based on attribute values

Required inputs:
    -HyPC point cloud file(s)
    -Attribute name and value to filter by

Created by: Christopher Iseli
Last Modified: 20/10/2018 (Christopher Iseli)
"""

import pickle, math, time
import numpy as np
import HyPC
from HyPC import hypc
from scipy.spatial import cKDTree
import time

## ===== Settings ====##
inputFname = "pointCloud_sample.hypc"
outputFname = "pointCloud_sample1.hypc"

attributeName = '1'         #   Attribute name to filter
attributeValue = False      #   Attribute value to filter by
##--------------------##

with open(inputFname, "rb") as f:
    pointCloud = pickle.load(f)

print("Filtering out points where '",attributeName,"' = ",attributeValue)

idxs = np.asarray(pointCloud.df.loc[pointCloud.df[attributeName] ==attributeValue].index)

newTree = cKDTree(pointCloud.kdTree.data[idxs], compact_nodes=True, balanced_tree=False)
pointCloud.kdTree = newTree
pointCloud.df = pointCloud.df.iloc[idxs]
pointCloud.df = pointCloud.df.reset_index(drop=True)

HyPC.updateHyPC(pointCloud,outputFname)

print("Complete")