"""
HyPC ground filter module

Description:
    Utilises the cloth simuldation filter (CSF) algorithm (reference) to identify ground points
        (W. Zhang, J. Qi, P. Wan, H. Wang, D. Xie, X. Wang, and G. Yan. “An easy-to-useairborne  LiDAR  data  filtering  method  based  on  cloth  simulation”)

Required inputs:
    -HyPC point cloud file

Created by: Christopher Iseli
Last Modified: 20/10/2018 (Christopher Iseli)
"""


import pickle, math, time
import HyPC
from HyPC import hypc
import CSF

## ===== Settings ====##
inputFname = "pointCloud_sample.hypc"
outputFname = "pointCloud_sample1.hypc"

rigidness = 2                   # Details on parameters are available at: http://ramm.bnu.edu.cn/projects/CSF/document/
cloth_resolution = 0.5
threshold = 0.5
iterations = 500
slope_smoothing = False
##--------------------##

print("Preparing CSF simulation..")
with open(inputFname, "rb") as f:
    pointCloud = pickle.load(f)

pointCloud.df["ground"] = False
grnd_col = pointCloud.df.columns.get_loc("ground")

csf = CSF.CSF()

# prameter settings
csf.params.bSloopSmooth = slope_smoothing
csf.params.rigidness = rigidness
csf.params.cloth_resolution = cloth_resolution
csf.params.class_threshold = threshold
csf.params.interations = iterations
csf.params.time_step = 0.65
# more details about parameter: http://ramm.bnu.edu.cn/projects/CSF/download/

csf.setPointCloud(pointCloud.kdTree.data)
ground = CSF.VecInt()  # a list to indicate the index of ground points after calculation
non_ground = CSF.VecInt() # a list to indicate the index of non-ground points after calculation
csf.do_filtering(ground, non_ground) # do actual filtering.

ground = list(ground)
pointCloud.df.iloc[ground, grnd_col] = True

HyPC.saveHyPC(pointCloud,outputFname)

