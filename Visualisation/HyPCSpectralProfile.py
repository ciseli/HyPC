"""
HyPC Spectral Profile

Description:
    Opens window for viewing spectral profile of points

Created by: Christopher Iseli
Last Modified: 20/10/2018 (Christopher Iseli)
"""


from PyQt5 import QtGui, uic
import pyqtgraph as pg


class hypcSpectralProfile(QtGui.QDialog):
    def __init__(self,MainWindow):
        super(hypcSpectralProfile, self).__init__(parent=MainWindow)
        uic.loadUi('SpectralProfile.ui', self)

        self.plot = pg.PlotCurveItem()
        self.plotWidget.addItem(self.plot)
        self.plotWidget.setYRange(0,1.2,padding=0)



