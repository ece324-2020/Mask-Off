# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ProjectWindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainProjectWidget(object):
    def setupUi(self, MainProjectWidget):
        MainProjectWidget.setObjectName("MainProjectWidget")
        MainProjectWidget.resize(1119, 767)
        self.gridLayout_2 = QtWidgets.QGridLayout(MainProjectWidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.mainProject_grid_layout = QtWidgets.QGridLayout()
        self.mainProject_grid_layout.setObjectName("mainProject_grid_layout")
        self.gridLayout_2.addLayout(self.mainProject_grid_layout, 0, 0, 1, 1)

        self.retranslateUi(MainProjectWidget)
        QtCore.QMetaObject.connectSlotsByName(MainProjectWidget)

    def retranslateUi(self, MainProjectWidget):
        _translate = QtCore.QCoreApplication.translate
        MainProjectWidget.setWindowTitle(_translate("MainProjectWidget", "Form"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainProjectWidget = QtWidgets.QWidget()
    ui = Ui_MainProjectWidget()
    ui.setupUi(MainProjectWidget)
    MainProjectWidget.show()
    sys.exit(app.exec_())

