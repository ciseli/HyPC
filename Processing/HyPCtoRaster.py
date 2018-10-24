import pickle, math, time
import numpy as np
import gdal
import osr
from HyPC import hypc
import matplotlib.pyplot as plt
import pykrige.ok as ok

## ===== Settings ====##
inputFname = "pointCloud_sample.hypc"
outputRasterFname = "pointCloud_sample1.tif"

pixel_size = 1 #size in (m)
FILTER = False
attribute = "ground"
attr_value = True
INTERPOLATE = False
##--------------------##

with open(inputFname, "rb") as f:
    pointCloud = pickle.load(f)

print("Processing ",inputFname)

min_x,max_x = pointCloud.extents[0]
min_y,max_y = pointCloud.extents[1]

range_x = max_x-min_x
range_y = max_y-min_y
ratio = (range_x)/(range_y)

x_res = math.ceil(range_x/pixel_size)
y_res = math.ceil(range_y/pixel_size)

raster_max_x = min_x+(x_res*pixel_size)
raster_max_y = min_y+(y_res*pixel_size)

x_segment_bounds,xstep = np.linspace(min_x, raster_max_x, num=x_res,retstep = True)
x_segment_bounds[0]=x_segment_bounds[0]+xstep/2
y_segment_bounds,ystep = np.linspace(min_y, raster_max_y, num=y_res,retstep=True)
y_segment_bounds[0]=y_segment_bounds[0]+ystep/2

## == Filter point cloud based on attributes
if FILTER:
    filtered_idxs = np.asarray(pointCloud.df.loc[pointCloud.df[attribute] ==attr_value].index)
    pointCloudData = pointCloud.kdTree.data[filtered_idxs]
    pointCloud.df = pointCloud.df.iloc[filtered_idxs]
    pointCloud.df = pointCloud.df.reset_index(drop=True)
else:
    pointCloudData = pointCloud.kdTree.data


bins_x = np.searchsorted(x_segment_bounds, pointCloudData[:, 0])
bins_y = np.searchsorted(y_segment_bounds, pointCloudData[:, 1])

pointCloud.df["bins_x"] = bins_x.astype(np.uint16)
pointCloud.df["bins_y"] = bins_y.astype(np.uint16)
segments = pointCloud.df.groupby(['bins_y', 'bins_x']).apply(lambda x: [x.name,x.index.tolist()])

## == Create Raster
raster = np.empty((y_res,x_res),dtype=np.float32)
raster[:] = np.nan

for i,segment in enumerate(segments):
    cell_loc = tuple(segment[0])
    values = pointCloudData[segment[1]]
    mean = np.mean(values[:,2])
    raster[cell_loc]=mean

from scipy.ndimage.filters import median_filter
raster = median_filter(raster,3)
raster = np.flipud(raster)

## == Apply kriging to create continuos surface

if INTERPOLATE:
    gridx = np.arange(0.0, raster.shape[0], 1.0)
    gridy = np.arange(0.0, raster.shape[1], 1.0)
    good_idx = np.transpose(np.argwhere(np.isfinite(raster)))
    nan_idx = np.transpose(np.argwhere(np.isnan(raster)))
    good_coords = np.asarray(list(zip(good_idx[0], good_idx[1])))
    nan_coords = list(zip(nan_idx[0],nan_idx[1]))
    z = raster[good_idx[0],good_idx[1]]
    OK = ok.OrdinaryKriging(good_coords[:, 0], good_coords[:, 1], z, variogram_model='linear',verbose=False, enable_plotting=False)
    mask=np.full(raster.shape,True)
    for coord in nan_coords:
        mask[coord]=False
    mask=np.transpose(mask)

    time_grab=time.clock()
    z, ss = OK.execute('masked', gridx, gridy,mask,backend='C',n_closest_points=100)
    print("Masked: ",time.clock()-time_grab)
    for coord in nan_coords:
        raster[coord]=np.transpose(z)[coord]

plt.imshow(raster)
plt.show(block=True)

geotransform = ([pointCloud.extents[0][0], pixel_size, 0, pointCloud.extents[1][1], 0, pixel_size ])
srs = osr.SpatialReference()
srs.ImportFromEPSG(28355)
driver = gdal.GetDriverByName("GTiff")
outdata = driver.Create(outputRasterFname, x_res, y_res, 1, gdal.GDT_Float32)
outdata.SetGeoTransform(geotransform)##sets same geotransform as input
outdata.SetProjection(srs.ExportToWkt())##sets same projection as input
outdata.GetRasterBand(1).WriteArray(raster)
outdata.GetRasterBand(1).SetNoDataValue(10000)##if you want these values transparent
outdata.FlushCache() ##saves to disk!!
outdata = None
band=None
ds=None

print("Raster saved to: ",outputRasterFname)