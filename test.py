import sys
from PyQt5 import QtCore, QtGui, QtWidgets


class TestW(QDialog):
    def __init__(self):
        super(TestW,self).__init__()
        loadUi('/home/deos/e.hrustic/PycharmProjects/test.ui',  self)
app = QApplication(sys.argv)
widget=TestW()
widget.show()
sys.exit(app.exec_())

