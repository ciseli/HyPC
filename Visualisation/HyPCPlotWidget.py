"""
HyPC Plot widget

Description:
    Called from the HyPCViewer script

    Contains all point cloud plotting functions and features

Created by: Christopher Iseli
Last Modified: 20/10/2018 (Christopher Iseli)
"""

## NOTE:   Modified line 330 in GLViewWidget.py

import sys,random,math,time
import numpy as np
from PyQt5 import QtGui, uic,QtCore
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import pickle
import random
import HyPCViewerFunctions as hvf
from itertools import chain

MAX_DISP_SEGMENTS=100

SELECTED_POINT_IDX = 0
SELECTED_POINT_SEGMENT = 0

class hypcPlotWidget(gl.GLViewWidget):
    def __init__(self, MainWindow):
        super(hypcPlotWidget,self).__init__(parent=MainWindow)

        self.main = MainWindow
        self.setBackgroundColor(0.25)
        self.brushSize = self.main.spbx_brushSize.value()
        self.pointSize = 0.05
        self.main.spbx_pointSize.setValue(self.pointSize)
        self.plot = None
        self.setCameraPosition([0,0,0],100,90,0)
        self.selecting = False
        self.selected_points = []
        self.currentSubsetFactor = 1
        self.region_selected = False
        self.scatterplotItems = []
        self.fov = 1
        self.opts['fov'] = self.fov

    def displayNewCloud(self,pointCloud,attribute):
        continuos = True
        if attribute=="class":
            continuos=False
        self.baseColours = hvf.constructPointColours(pointCloud.df,attribute,continuos)
        self.colours = np.copy(self.baseColours)
        self.updatePointCloud(pointCloud,True)

    def updatePointCloud(self,pointCloud,resetCamera):
        self.time_grab=time.clock()
        for item in self.scatterplotItems:# Clear all current points
            self.removeItem(item)
            item._setView(None)
        self.update()
        self.scatterplotItems = []
        if self.region_selected == False:   # Check if a region has been selected
            self.scatterplotItems = []
            subset_factor = math.ceil(pointCloud.df.shape[0] / self.main.max_display_points)
            self.currentSubsetFactor=subset_factor
            num_points = self.plotSegments(pointCloud,subset_factor)
            if resetCamera:
                self.resetCamera([0,0],pointCloud.extents)
        else:
            x_bounds = sorted([self.selectedBounds[0][0],self.selectedBounds[1][0]]) # detemine segment id's which fall within selected bounds
            y_bounds = sorted([self.selectedBounds[0][1],self.selectedBounds[1][1]])
            in_idxs = []
            for i,bound in enumerate(pointCloud.segment_bounds):
                segment_x = np.sort([bound[0],bound[2]])-pointCloud.medians[0]
                segment_y = np.sort([bound[1],bound[3]])-pointCloud.medians[1]

                x0,x1 = np.max([x_bounds[0],segment_x[0]]),np.min([x_bounds[1],segment_x[1]])
                y0,y1 = np.max([y_bounds[0],segment_y[0]]),np.min([y_bounds[1],segment_y[1]])
                if x0<=x1:
                    if y0<=y1:
                        in_idxs.append(i)
            points_in_region = 0
            for idx in pointCloud.segment_indices[in_idxs]:
                points_in_region+=len(idx)
            subset_factor = math.ceil(points_in_region / self.main.max_display_points)
            self.currentSubsetFactor=subset_factor
            num_points = self.plotSegments(pointCloud,subset_factor,in_idxs)
            selected_centre = [(x_bounds[0]+x_bounds[1])/2,(y_bounds[0]+y_bounds[1])/2]
            self.main.actZoomtoRegion.setChecked(False)
            self.main.mode = 1
            if resetCamera:
                self.resetCamera(selected_centre,self.selectedBounds)
        self.main.statusbar.showMessage(str(num_points)+" of "+str(pointCloud.df.shape[0])+" points currently displayed..")

    def plotSegments(self,pointCloud,subset_factor,in_idxs = None):
        num_points_displayed = 0
        # print("Number to display: ",len(pointCloud.segment_indices))

        if in_idxs == None:
            for i, idx in enumerate(pointCloud.segment_indices[:]):
                if SELECTED_POINT_SEGMENT==i:
                    if SELECTED_POINT_IDX in idx[::subset_factor]:
                        # print("POINT AT IDX " +str(SELECTED_POINT_IDX) +" IS DISPLAYED")
                        # print("COLOUR: ", self.colours[SELECTED_POINT_IDX] )
                        continue
                    else:
                        # print("POINT NOT FOUND")
                        scatterPlot = gl.GLScatterPlotItem()
                        scatterPlot.setData(idx=i,pos=pointCloud.kdTree.data[SELECTED_POINT_IDX,:3]-pointCloud.medians, color=self.colours[SELECTED_POINT_IDX], size=self.pointSize, pxMode=True)
                        # scatterPlot.setData(idx=i, pos=pointCloud.kdTree.data[idx[::subset_factor], :3] - pointCloud.medians,color=(random.random(),random.random(),random.random(),1.0),size=0.25,pxMode=False)
                        scatterPlot.setGLOptions('opaque')
                        self.addItem(scatterPlot)
                        self.scatterplotItems.append(scatterPlot)
                        num_points_displayed += 1


                scatterPlot = gl.GLScatterPlotItem()
                scatterPlot.setData(idx=i,pos=pointCloud.kdTree.data[idx[::subset_factor],:3]-pointCloud.medians, color=self.colours[idx[::subset_factor]], size=self.pointSize, pxMode=False)
                # scatterPlot.setData(idx=i, pos=pointCloud.kdTree.data[idx[::subset_factor], :3] - pointCloud.medians,color=(random.random(),random.random(),random.random(),1.0),size=0.25,pxMode=False)
                scatterPlot.setGLOptions('opaque')
                self.addItem(scatterPlot)
                self.scatterplotItems.append(scatterPlot)
                num_points_displayed += len(pointCloud.kdTree.data[idx[::subset_factor]])
        else:
            if len(pointCloud.segment_indices[in_idxs])>MAX_DISP_SEGMENTS: # If number of segments is above limit, merge segments for display
                excess = len(pointCloud.segment_indices[in_idxs]) - MAX_DISP_SEGMENTS
                ## FINISH THIS, NEED TO SOLVE FOR IDS AS WELL

            for i, idx in enumerate(pointCloud.segment_indices[in_idxs]):
                print(idx)
                if SELECTED_POINT_IDX in idx:
                    # print("POINT IS DISPLAYED")
                    continue
                scatterPlot = gl.GLScatterPlotItem()
                scatterPlot.setData(idx=in_idxs[i], pos=pointCloud.kdTree.data[idx[::subset_factor], :3] - pointCloud.medians,color=self.colours[idx[::subset_factor]], size=self.pointSize, pxMode=False)
                scatterPlot.setGLOptions('opaque')
                self.addItem(scatterPlot)
                self.scatterplotItems.append(scatterPlot)
                num_points_displayed += len(pointCloud.kdTree.data[idx[::subset_factor]])
        return num_points_displayed

    def reset(self):
        self.region_selected=False
        self.colours = self.baseColours
        self.updatePointCloud(self.main.pointCloud,True)

    def resetCamera(self,centre,bounds):
        bound_x = abs(np.diff(bounds[0]))
        distance_x = (bound_x/2) / math.tan(math.radians(self.fov/2))
        bound_y = abs(np.diff(bounds[1]))
        distance_y = (bound_y/2) / math.tan(math.radians(self.fov/2))
        self.setCameraPosition(pos=centre,distance=np.max([distance_x,distance_y]),elevation=90,azimuth=-90)
        self.opts['fov'] = self.fov

    def plotRefPoints(self,pos,size):
        pos[...,0] -= self.main.pointCloud.medians[0]
        pos[...,1] -= self.main.pointCloud.medians[1]
        self.refBarItem = gl.GLBarGraphItem(pos, size)
        self.refBarItem.setGLOptions('additive')
        self.refBarItem.setColor([1,0,0,0.75])
        self.addItem(self.refBarItem)

    def mousePressEvent(self, ev):
        super(hypcPlotWidget, self).mousePressEvent(ev)
        if self.main.mode ==2 and ev.button()==2:
            self.mPosition(ev)
        self._downpos = self.mousePos

    def mouseReleaseEvent(self, ev):
        super(hypcPlotWidget, self).mouseReleaseEvent(ev)
        if self.main.mode ==1:
            if ev.button()==2:
                self.mPosition(ev)
        elif self.main.mode == 2:
            if ev.button()==2:
                self.mPosition(ev)
                if self.selecting == True:
                    self.selectedBounds = [self.point1,self.point2]
                    self.selecting = False
                    self.region_selected = True
                    self.removeItem(self.m1)
                    self.updatePointCloud(self.main.pointCloud,True)
        elif self.main.mode == 3:
            if ev.button() == 2 :
                self.mPosition(ev)

        self._prev_zoom_pos = None
        self._prev_pan_pos = None


    def mouseMoveEvent(self, ev):
        shift = ev.modifiers() & QtCore.Qt.ShiftModifier
        ctrl = ev.modifiers() & QtCore.Qt.ControlModifier
        if self.selecting and self.main.mode==2:
            self.mPosition(ev)
        if shift:
            y = ev.pos().y()
            if not hasattr(self, '_prev_zoom_pos') or not self._prev_zoom_pos:
                self._prev_zoom_pos = y
                return
            dy = y - self._prev_zoom_pos
            def delta():
                return -dy * 5
            ev.angleDelta = delta
            self._prev_zoom_pos = y
            self.wheelEvent(ev)
        elif ctrl:
            pos = ev.pos().x(), ev.pos().y()
            if not hasattr(self, '_prev_pan_pos') or not self._prev_pan_pos:
                self._prev_pan_pos = pos
                return
            dx = pos[0] - self._prev_pan_pos[0]
            dy = pos[1] - self._prev_pan_pos[1]
            self.pan(dx, dy, 0, relative=True)
            self._prev_pan_pos = pos
        else:
            super(hypcPlotWidget, self).mouseMoveEvent(ev)

    def mPosition(self,ev):         ## This function is based on the method outlined at: https://groups.google.com/forum/?nomobile=true#!msg/pyqtgraph/mZiiLO8hS70/nSkTmYPtIiYJ
        #This function is called by a mouse event
        ## Get mouse coordinates saved when the mouse is clicked( incase dragging)
        mx = ev.pos().x()
        my = ev.pos().y()
        if self.main.mode != 2:
            objs =self.itemsAt((mx-25,my-25,50,50))
            if isinstance(objs, list):
                idxs = []
                for obj in objs:
                    if obj is gl.GLBarGraphItem:
                        continue
                    idxs.append(obj.idx)
            else:
                idxs = objs[0].idx
            self.pickedPoints = [] #Initiate a list for storing indices of picked points
        #Get height and width of 2D Viewport space
        view_w = self.width()
        view_h = self.height()
        #Convert pixel values to normalized coordinates
        x = 2.0 * mx / view_w - 1.0
        y = 1.0 - (2.0 * my / view_h)
        # Convert projection and view matrix to np types and inverse them
        PMi = self.projectionMatrix().inverted()[0]
        VMi = self.viewMatrix().inverted()[0]
        ray_clip = QtGui.QVector4D(x, y, -1.0, 1.0) # get transpose for matrix multiplication
        ray_eye = PMi * ray_clip
        ray_eye.setZ(-1)
        ray_eye.setW(0)
        #Convert to world coordinates
        ray_world = VMi * ray_eye
        ray_world = QtGui.QVector3D(ray_world.x(), ray_world.y(), ray_world.z()) # get transpose for matrix multiplication
        ray_world.normalize()


        if self.main.mode ==2:
            ray_pos = np.array(self.cameraPosition())
            ray_dir = np.array([ray_world.x(), ray_world.y(), ray_world.z()])
            self.groundPosition(ray_pos,ray_dir)
        else:
            self.time_grab2=time.clock()
            try:
                O = np.matrix(self.cameraPosition())  # camera position should be starting point of the ray
                ray_world = np.matrix([ray_world.x(), ray_world.y(), ray_world.z()])
                pickedInds = []
                for n,obj_idx in enumerate(idxs):
                    localPickedCoords = []
                    localPickedInds = []
                    matches = []
                    for i, C in enumerate(self.main.pointCloud.kdTree.data[self.main.pointCloud.segment_indices[obj_idx][::self.currentSubsetFactor],:3]-self.main.pointCloud.medians): # Iterate over all points currently in view
                        OC = O - C
                        b = np.inner(ray_world, OC)
                        b = b.item(0)
                        c = np.inner(OC, OC)
                        c = c.item(0) - (0.2/2)**2   #np.square((self.Sizes[i]))
                        bsqr = np.square(b)
                        if (bsqr - c) >= 0: # True means intersection
                            localPickedCoords.append(C)
                            matches.append(bsqr-c)
                            localPickedInds.append(self.main.pointCloud.segment_indices[obj_idx][i])
                    nodes = np.asarray(localPickedCoords)
                    if len(nodes>1):
                        dist_2 = np.sum((nodes - np.asarray(O)) ** 2, axis=1)
                        min = np.argmin(dist_2)
                        sel = np.argmax(matches)
                        # closest_idx =localPickedInds[np.argmin(dist_2)]
                        closest_idx = localPickedInds[sel]
                        pickedInds.append([obj_idx,n,closest_idx,min,localPickedCoords[sel]])
                    elif len(nodes==1):
                        closest_idx=localPickedInds[0]
                        pickedInds.append([obj_idx, n, closest_idx, min,localPickedCoords])
                minIdx = np.argmin([np.linalg.norm(O-row[4]) for row in pickedInds])
                objIdx, i,idx= pickedInds[minIdx][0],pickedInds[minIdx][1],pickedInds[minIdx][2]
                coordinate = pickedInds[minIdx][4]
                if self.main.mode ==1:

                    global SELECTED_POINT_IDX, SELECTED_POINT_SEGMENT
                    SELECTED_POINT_IDX= idx
                    SELECTED_POINT_SEGMENT = objIdx
                    self.resetSelection()
                    self.colours[idx] = (1, 0, 0, 1)
                    self.selected_points.append(idx)
                    self.updatePointCloud(self.main.pointCloud,False)
                    attributes = [self.main.pointCloud.df.columns.values.tolist(),self.main.pointCloud.df.iloc[idx].tolist()]
                    if self.main.spectralProfileWin:
                        self.updateSpectraPlot(attributes)
                    if self.main.attributeInfoWin:
                        self.updateAttributeTable(attributes)
                elif self.main.mode ==3:
                    # select points
                    kdTreeIdxs = self.main.pointCloud.kdTree.query_ball_point(coordinate+self.main.pointCloud.medians,self.brushSize)
                    self.colours[kdTreeIdxs] = (1, 0, 0, 1)
                    self.selected_points.append(kdTreeIdxs)
                    self.updatePointCloud(self.main.pointCloud,False)
            except Exception as e:
                print(e)
                print("Point not found..")

    def groundPosition(self,ray_pos,ray_dir):
        epsilon = 0.00001
        plane_pos = np.array([0.0, 0.0, 0.0])
        plane_normal = np.array([0.0, 0.0, 1.0])
        ndotu = plane_normal.dot(ray_dir)
        if abs(ndotu) < epsilon:
            print ("no intersection or line is within plane")
        w = ray_pos - plane_pos
        si = -plane_normal.dot(w) / ndotu
        groundPosition = w + si * ray_dir + plane_pos
        if self.selecting==False:
            self.point1 = groundPosition
            self.createPolygon(groundPosition,groundPosition)
        else:
            self.point2 = groundPosition
            self.updatePolygon(self.point1,self.point2)

    def createPolygon(self,point1,point2):
        verts = np.array([
            [point1[0], point1[1], point1[2]],
            [point2[0], point1[1], point1[2]],
            [point2[0], point2[1], point1[2]],
            [point1[0], point2[1], point1[2]]])
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3]])
        self.m1 = gl.GLMeshItem(vertexes=verts, faces=faces, drawEdges = False, drawFaces=True, smooth=False)
        self.m1.setColor((1, 0, 0, 0.5))
        self.m1.setGLOptions('additive')
        self.addItem(self.m1)
        self.selecting = True

    def updatePolygon(self,point1,point2):
        verts = np.array([
            [point1[0], point1[1], point1[2]],
            [point2[0], point1[1], point1[2]],
            [point2[0], point2[1], point1[2]],
            [point1[0], point2[1], point1[2]]])
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3]])
        self.m1.setMeshData(vertexes=verts, faces=faces, smooth=False)
        self.m1.meshDataChanged()

    def updateSpectraPlot(self,attributes):
        vb_wavelengths = [604.7045083728286, 612.8041542313671, 620.8230789206758, 643.3324081266743, 651.0988772421822, 660.0734450434167, 668.0925438846642, 676.7273262198124, 684.8737923944651, 712.7357033049248, 737.2410640980314, 751.1524148865271, 763.2107974182375, 776.5020500674016, 789.0445588804852, 801.6473532381983, 813.3302513629362, 825.946684337831, 842.7454629199874, 854.2552423729062, 864.2665102928495]
        self.main.spectralProfileWin.plot.setData(np.asarray(vb_wavelengths),np.asarray(attributes[1][:21])/2**16)
        # if self.plot is None:
        #     self.plot = pg.plot(attributes,clear=True)
        #     self.plot.setYRange(0, 65000, padding=0)
        # else:
        #     self.plot.window().close()
        #     self.plot = pg.plot(attributes, clear=True)
        #     self.plot.setYRange(0, 65000, padding=0)

    def updateAttributeTable(self,attributes):
        tableWidget = self.main.attributeInfoWin.tableWidget
        tableWidget.setRowCount(len(attributes[0]))
        tableWidget.setColumnCount(2)
        for i in range(len(attributes[0])):
            # tableWidget.insertRow(i)
            tableWidget.setItem(i, 0, QtGui.QTableWidgetItem(attributes[0][i]))
            tableWidget.setItem(i, 1, QtGui.QTableWidgetItem(str(attributes[1][i])))


    def assignPointsToClass(self,pointCloud,class_num):
        point_idxs = list(chain.from_iterable(self.selected_points))
        pointCloud.df.loc[pointCloud.df.index[point_idxs], 'class'] = class_num
        print("Points assigned to class :", class_num)

    def assignPointsToSegment(self,pointCloud):
        #Get Next available segement ID
        seg_max = pointCloud.df['segment_id'].max()
        point_idxs = list(chain.from_iterable(self.selected_points))
        pointCloud.df.loc[pointCloud.df.index[point_idxs], 'segment_id'] = int(seg_max)+1
        print("Points assigned to segment :", int(seg_max)+1)

    def resetSelection(self):
        self.selected_points = []
        self.colours = np.copy(self.baseColours)
        self.updatePointCloud(self.main.pointCloud,False)

    def undoSelection(self):
        del self.selected_points[-1]
        self.colours = np.copy(self.baseColours)
        for idxs in self.selected_points:
            self.colours[idxs] = (1, 0, 0, 1)
        self.updatePointCloud(self.main.pointCloud,False)

    def markForDeletion(self,pointCloud):
        marked_idxs = []
        for idxs in self.selected_points:
            marked_idxs.extend(idxs)

        if "to_delete" not in pointCloud.df.columns:
            pointCloud.df["to_delete"] = False
        delete_col = pointCloud.df.columns.get_loc("to_delete")
        for idx in marked_idxs:
            pointCloud.df.iat[idx, delete_col] = True

        self.main.updateAttributes()