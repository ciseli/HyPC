"""
HyPC polygon segmentation module

Description:
    Segment points based on a polygon shapefile. Each polygon in the specified shapefile creates a new segment.
    All polygons within a single shapefile are assigned to a single class

Required inputs:
    - HyPC point cloud file
    - polygon shapefile

Created by: Christopher Iseli
Last Modified: 20/10/2018 (Christopher Iseli)
"""

import pickle, math, time
import numpy as np
import HyPC
from HyPC import hypc
from shapely.geometry import Polygon, Point, MultiPolygon, box
import shapefile
import time
np.set_printoptions(suppress=True)

## ===== Settings ====##
inputFname = "pointCloud_sample.hypc"
inputShapefile = "D:/Honours Data/Dungrove/Crowns/species_2_crop_fixed.shp"
outputFname = "pointCloud_sample1.hypc"

class_value = 2
##--------------------##


with open(inputFname, "rb") as f:
    pointCloud = pickle.load(f)

segment_col = pointCloud.df.columns.get_loc("segment_id")
max_current_segment_id = sorted(pointCloud.df['segment_id'].unique())[-1]
segment_id= max_current_segment_id+1

class_col = pointCloud.df.columns.get_loc("class")

shpfile = shapefile.Reader(inputShapefile)

points = []
for l, feature in enumerate(shpfile.iterShapes()):
    print("Processing segment ",int(segment_id))
    points_found = False
    n_points = 0
    for i,segment in enumerate(pointCloud.segment_bounds):
        segment = box(segment[0], segment[1], segment[2], segment[3], ccw=True)
        first = feature.__geo_interface__
        poly = Polygon(first['coordinates'][0])
        if segment.intersects(poly):
            ins = []
            in_idx=[]
            for n,point in enumerate(pointCloud.kdTree.data[pointCloud.segment_indices[i]]):

                if poly.contains(Point(point[:2])):
                    ins.append([point[0],point[1]])
                    n_points+=1
                    in_idx.append(pointCloud.segment_indices[i][n])
            if len(in_idx)>0:
                pointCloud.df.iloc[in_idx,segment_col]=segment_id
                pointCloud.df.iloc[in_idx, class_col] = class_value
                points_found=True
    if points_found:
        segment_id+=1
        points.append(n_points)

HyPC.saveHyPC(pointCloud,outputFname)

print("Complete")
