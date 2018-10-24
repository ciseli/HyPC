import pickle, math, time
import numpy as np
import HyPC
from HyPC import hypc
import gdal

## ===== Settings ====##
inputHyPCFname = "pointCloud_sample.hypc"
inputDTMFname = "pointCloud_sample1.tif"
outputFname = "pointCloud_sample1.hypc"
##--------------------##

with open(inputHyPCFname, "rb") as f:
    pointCloud = pickle.load(f)

print("Processing ",inputHyPCFname)

ds = gdal.Open(inputDTMFname)
band = ds.GetRasterBand(1)
raster = band.ReadAsArray()

origin, pixel_size = [ds.GetGeoTransform()[0],ds.GetGeoTransform()[3]],ds.GetGeoTransform()[1]

## Segment point cloud to match DEM pixels
min_x,max_y = origin
max_x, min_y = min_x + pixel_size*raster.shape[1], max_y - pixel_size*raster.shape[0]+pixel_size

x_segment_bounds,xstep = np.linspace(min_x, max_x, num=raster.shape[1]-1,retstep = True)
x_segment_bounds[0]=x_segment_bounds[0]+xstep/2
y_segment_bounds,ystep = np.linspace(min_y, max_y, num=raster.shape[0]-1,retstep=True)
y_segment_bounds[0]=y_segment_bounds[0]+ystep/2

# Create bins
bins_x = np.searchsorted(x_segment_bounds, pointCloud.kdTree.data[:, 0])
bins_y = np.searchsorted(y_segment_bounds, pointCloud.kdTree.data[:, 1])

pointCloud.df["bins_x"] = bins_x.astype(np.uint16)
pointCloud.df["bins_y"] = bins_y.astype(np.uint16)
segments = pointCloud.df.groupby(['bins_y', 'bins_x']).apply(lambda x: [x.name,x.index.tolist()])

norm_heights = np.zeros(len(pointCloud.df))
for i,segment in enumerate(segments):
    cell_loc = tuple(segment[0])
    elevation = raster[cell_loc]
    heights = pointCloud.kdTree.data[segment[1]][:,2]
    if np.isnan(elevation):
        normalised_heights = 0.00003
    else:
        normalised_heights = heights - elevation

    norm_heights[segment[1]] = normalised_heights

pointCloud.df["Normalised_Z"] = norm_heights

HyPC.updateHyPC(pointCloud,outputFname)

print("Complete")