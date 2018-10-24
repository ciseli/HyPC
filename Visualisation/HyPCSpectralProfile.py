# import sys,random,math
# import numpy as np
from PyQt5 import QtGui, uic
# import pyqtgraph.opengl as gl
import pyqtgraph as pg


class hypcSpectralProfile(QtGui.QDialog):
    def __init__(self,MainWindow):
        super(hypcSpectralProfile, self).__init__(parent=MainWindow)
        uic.loadUi('SpectralProfile.ui', self)

        self.plot = pg.PlotCurveItem()
        self.plotWidget.addItem(self.plot)
        self.plotWidget.setYRange(0,1.2,padding=0)



