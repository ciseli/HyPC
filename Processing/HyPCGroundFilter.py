import pickle, math, time
import HyPC
from HyPC import hypc
import CSF

## ===== Settings ====##
inputFname = "pointCloud_sample.hypc"
outputFname = "pointCloud_sample1.hypc"

rigidness = 2
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

