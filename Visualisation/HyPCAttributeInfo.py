
from PyQt5 import QtGui, uic,QtCore,QtWidgets

import pyqtgraph as pg

pg.setConfigOption('background', 0.25)
pg.setConfigOption('foreground', 1)

class hypcAttributeInfo(QtGui.QDialog):
    def __init__(self,MainWindow):
        super(hypcAttributeInfo, self).__init__(parent=MainWindow)
        uic.loadUi('AttributeInfo.ui', self)

        pass


