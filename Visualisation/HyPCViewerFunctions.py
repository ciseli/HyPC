import numpy as np
import shapefile
import random

def getPointsFromShapefile(fname):
    shp = shapefile.Reader(fname)
    all_shapes = shp.shapes()  # get all the polygons
    refPoints=[]
    for shape in all_shapes:
        for point in shape.points:
            print(point)
            refPoints.append(point)

    base_size = 0.5
    num_points = len(refPoints)
    pos = np.zeros((num_points,1,3))
    size = np.zeros((num_points,1,3))
    for i in range(num_points):
        pos[i][0] = [refPoints[i][0]-base_size/2,refPoints[i][1]-base_size/2,-5]
        size[i][0] = [base_size,base_size,10]

    return pos,size


def constructPointColours(attributes,attribute,continuos):
    colours = np.zeros((attributes.shape[0], 4))
    if continuos:
        x = attributes.loc[:,attribute].values.astype(np.float32)
        min,max = np.min(x), np.max(x)
        range = max - min
        normalized = (x[:] - np.min(x)) / range
        val1,val2 = np.full((len(normalized)),0.5),np.full((len(normalized)),1.0)
        a = np.transpose(np.array([normalized[:],normalized[:],normalized[:],val2[:]]))
        colours[:] = a[:]
        return colours
    else:
        x = attributes.loc[:, attribute].values.astype(np.int8)  # returns a numpy array
        classes = np.unique(x)
        for class_val in classes:
            idxs = np.where(x==class_val)
            colours[idxs] = np.array([random.random(),random.random(),random.random(),1.0])

    return colours
