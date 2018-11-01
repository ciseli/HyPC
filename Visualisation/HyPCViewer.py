"""
HyPC Viewer

Description:
    Base script for running the HyPC visualiser

Required inputs:
    -HyPC point cloud file

Created by: Christopher Iseli
Last Modified: 20/10/2018 (Christopher Iseli)


NOTES:

Need to add the following to Pyqtgraph's GLViewWidget.py setCameraPosition() function

    if pos is not None:
        self.opts['center'] = Vector(pos)


"""

import sys,random,math,time
from PyQt5 import QtGui, uic,QtCore
import pyqtgraph as pg
import pickle
import HyPCPlotWidget as hpw
import HyPCViewerFunctions as hvf
from HyPCViewerFunctions import *
from HyPCSpectralProfile import *
from HyPCAttributeInfo import *
from HyPC import hypc

pg.setConfigOption('background', 'k')
pg.setConfigOption('foreground', 0.75)

class hypcViewer(QtGui.QMainWindow):
    def __init__(self):
        super(hypcViewer, self).__init__()
        uic.loadUi('HyPCViewer.ui', self)

        self.max_display_points = 500000
        self.statusbar.showMessage("Ready...")
        self.plotWidget = hpw.hypcPlotWidget(self)
        self.classValue = 0
        self.mode = 1
        self.spectralProfileWin = None
        self.attributeInfoWin = None
        self.pyqtgraphLayout.addWidget(self.plotWidget)
        self.show()

    def openFile(self):

        fileName = str(QtGui.QFileDialog.getOpenFileName(self, 'Select Point Cloud file..','', "HyPC point cloud (*.hypc)")[0])
        with open(fileName, "rb") as f:
            self.pointCloud = pickle.load(f)

        self.plotWidget.displayNewCloud(self.pointCloud,'20')
        self.updateAttributes()
        index = self.attributeDropdown.findText('20', QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.attributeDropdown.setCurrentIndex(index)

    def updateAttributes(self):
        attributeLabels = (list(self.pointCloud.df))
        self.attributeDropdown.clear()
        self.attributeDropdown.addItems(attributeLabels)


    def changeAttribute(self,label):
        self.plotWidget.displayNewCloud(self.pointCloud,label)

    def changePointSize(self,size):
        self.plotWidget.pointSize = self.spbx_pointSize.value()
        self.plotWidget.updatePointCloud(self.pointCloud,False)

    def resetCamera(self):
        # QtWidgets.QMessageBox.about(self, "Message", "Camera reset")
        self.plotWidget.reset()

    def updateBrushSize(self):
        self.plotWidget.brushSize=self.spbx_brushSize.value()

    def changeClassValue(self):
        self.classValue = self.spbx_class.value()

    def assignPointsToClass(self):
        self.plotWidget.assignPointsToClass(self.pointCloud,self.classValue)

    def assignPointsToSegment(self):
        self.plotWidget.assignPointsToSegment(self.pointCloud)

    def resetSelection(self):
        self.plotWidget.resetSelection()

    def showSpectralProfile(self):
        self.spectralProfileWin = hypcSpectralProfile(self)
        self.spectralProfileWin.show()

    def showAttributeInfo(self):
        self.attributeInfoWin = hypcAttributeInfo(self)
        self.attributeInfoWin.show()

    def undoSelection(self):
        self.plotWidget.undoSelection()

    def saveChanges(self):
        print("Saving changes to file..")
        with open(inputFname, "wb") as f:
            pickle.dump(self.pointCloud, f,pickle.HIGHEST_PROTOCOL)
        print("Save complete..")

    def markForDeletion(self):
        self.plotWidget.markForDeletion(self.pointCloud)

    def zoomToRegion(self,checked):
        if checked:
            self.mode = 2
        else:
            self.mode = 1

    def selectPoints(self,checked):
        if checked:
            self.mode=3
        else:
            self.mode=1

    def loadRefPoints(self):
        fileName = QtGui.QFileDialog.getOpenFileName(self, 'Select reference shapefile..','', "Shapefile (*.shp)")[0]
        pos,size = hvf.getPointsFromShapefile(fileName)
        self.plotWidget.plotRefPoints(pos,size)

    def showRefPoints(self, checked):
        if checked:
            self.plotWidget.addItem(self.plotWidget.refBarItem)
        else:
            self.plotWidget.removeItem(self.plotWidget.refBarItem)

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    app.setStyle('Fusion')
    window = hypcViewer()
    sys.exit(app.exec_())
