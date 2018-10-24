import numpy as np
import pandas as pd
import time, math
from scipy.spatial import cKDTree
import pickle

## ===== Settings ====##

inputFname = "pointCloud_sample.txt"
outputFname = "pointCloud_sample1.hypc"

##--------------------##

class hypc:
    def __init__(self, asciiFname, max_segment = 200000):
        start_time = time.clock()
        print("Importing:", inputFname)
        self.df = pd.read_csv(asciiFname, sep=" ", names=column_names)

        print("Point cloud loaded in: ",time.clock()-start_time)
        self.df.loc[:, 'x':'z'] = self.df.loc[:, 'x':'z'].astype(dtype=np.float64)
        self.df.loc[:, '1':'21'] = self.df.loc[:, '1':'21'].astype(dtype=np.uint16)
        self.df['class'] = np.zeros(self.df.shape[0])
        self.df['segment_id'] = np.empty(self.df.shape[0])
        self.df = self.df.sort_values('x')
        self.df = self.df.reset_index(drop=True)

        # self.segment

        print("Creating K-DTree structure")
        self.kdTree = cKDTree(self.df.loc[:, ['x', 'y', 'z']], compact_nodes=True, balanced_tree=False)
        mins,maxes = self.kdTree.mins,self.kdTree.maxes
        self.extents = [[mins[0],maxes[0]],[mins[1],maxes[1]],[mins[2],maxes[2]]]
        self.medians = [np.median(self.kdTree.data[:,0]),np.median(self.kdTree.data[:,1]),np.median(self.kdTree.data[:,2])]
        self.df = self.df.drop(['x','y','z'],axis=1)
        self.segment_indices,self.segment_bounds = segmentPoints(self)
        print("\nNew HyPC point cloud created..", time.clock()-start_time)
        saveHyPC(self,outputFname)
        print("Time for completion: ", time.clock() - start_time)
        # self.segmentPoints()
        self.stats()

    def stats(self):
        shape = self.df.shape
        print("\nImported ASCII file with:")
        print("\tNum. Points: " +str(shape[0]))
        print("\tAttributes: " + str(shape[1]))
        print("\nExtents: ")
        print("\t X: "+str(self.extents[0]))
        print("\t Y: " + str(self.extents[1]))
        print("\t Z: " + str(self.extents[2]))

def updateHyPC(pointCloud,outFname):

    mins, maxes = pointCloud.kdTree.mins, pointCloud.kdTree.maxes
    pointCloud.extents = [[mins[0], maxes[0]], [mins[1], maxes[1]], [mins[2], maxes[2]]]
    pointCloud.medians = [np.median(pointCloud.kdTree.data[:, 0]), np.median(pointCloud.kdTree.data[:, 1]),np.median(pointCloud.kdTree.data[:, 2])]

    pointCloud.segment_indices, pointCloud.segment_bounds = segmentPoints(pointCloud)
    indices = []
    for idxs in pointCloud.segment_indices:
        indices.append(np.max(idxs))

    saveHyPC(pointCloud, outFname)

def segmentPoints(pointCloud,max_points=200000, num_segments = None):
    if pointCloud.df.shape[0]<200000:
        # segment_indices = [pointCloud.df.index.values.astype(np.uint32)]
        # segment_bounds =  [pointCloud.extents[0][0],pointCloud.extents[1][0],pointCloud.extents[0][1],pointCloud.extents[1][1]]
        num_segments=2
    else:
        num_segments = int(math.sqrt(pointCloud.df.shape[0]/max_points))
    start = time.clock()
    min_x,max_x = pointCloud.extents[0]
    min_y,max_y = pointCloud.extents[1]
    # x_size = int(np.floor((max_y - min_y) / num_segments))
    # y_size = int(np.floor((max_x - min_x) / num_segments))
    x_size = (max_x - min_x) / num_segments
    y_size = (max_y - min_y) / num_segments
    x_segment_bounds,xstep = np.linspace(min_x+x_size, max_x, num= num_segments,retstep = True)
    y_segment_bounds,ystep = np.linspace(min_y+y_size, max_y, num=num_segments,retstep=True)

    # Create bins
    bins_x = np.searchsorted(x_segment_bounds, pointCloud.kdTree.data[:, 0])
    bins_y = np.searchsorted(y_segment_bounds, pointCloud.kdTree.data[:, 1])

    x_segment_bounds =np.insert(x_segment_bounds,0,min_x)
    y_segment_bounds = np.insert(y_segment_bounds, 0, min_y)

    pointCloud.df["bins_x"] = bins_x.astype(np.uint16)
    pointCloud.df["bins_y"] = bins_y.astype(np.uint16)

    segments = pointCloud.df.groupby(['bins_x', 'bins_y'])
    # print("Time to create segments: ", time.clock()-start)

    segment_indices = []
    segment_groups = []
    for i,segment in enumerate(segments):
        segment_indices.append(segment[1].index.values.astype(np.uint32))
        segment_groups.append(list(segment[0]))
    segment_indices = np.array(segment_indices)
    segment_bounds = []

    for i, group in enumerate(segment_groups):
        x_start, x_end = x_segment_bounds[group[0]],x_segment_bounds[group[0]]+xstep
        y_start, y_end = y_segment_bounds[group[1]], y_segment_bounds[group[1]]+ystep
        rect = [x_start,y_start,x_end,y_end]
        segment_bounds.append(rect)

    return segment_indices,segment_bounds

def saveHyPC(hypc_obj,fname):
    print("Saving point cloud to :", fname)
    with open(fname, "wb") as f:
        pickle.dump(hypc_obj, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    column_names = ['x','y','z','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21']
    cloud = hypc(inputFname)


