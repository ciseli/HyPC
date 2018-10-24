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

for i in range(num_segments-1):

    segment_idxs = pointCloud.df.loc[pointCloud.df['segment_id'] == i+1].index.values
    print(len(segment_idxs))
    # get mean spectra
    spectra = pointCloud.df.iloc[segment_idxs,:21].values
    mean_spectra = np.mean(spectra,axis=0)
    median_spectra = np.median(spectra,axis=0)
    if len(segment_idxs)>0:
        max_spectra = np.max(spectra,axis=0)

    spectra_brightness = np.sum(spectra,axis=1)
    if len(spectra_brightness)>9:
        ind = np.argpartition(spectra_brightness, -10)[-10:]

        brightest_mean = np.mean(spectra[ind],axis=0)
    else:
        brightest_mean = mean_spectra

    if  str(i+1) not in pointCloud.segments:
        pointCloud.segments[str(i + 1)] = {}

    pointCloud.segments[str(i+1)]['bright_mean']=brightest_mean
    pointCloud.segments[str(i+1)]['mean_spectra']=mean_spectra

HyPC.saveHyPC(pointCloud,outputFname)

print("Complete")