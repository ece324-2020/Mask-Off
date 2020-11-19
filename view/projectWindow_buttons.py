# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ProjectWindow_buttons.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(False)
        MainWindow.resize(1076, 90)
        self.ewqewrq = QtWidgets.QWidget(MainWindow)
        self.ewqewrq.setEnabled(False)
        self.ewqewrq.setMinimumSize(QtCore.QSize(800, 0))
        self.ewqewrq.setObjectName("ewqewrq")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.ewqewrq)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.setting_button = QtWidgets.QPushButton(self.ewqewrq)
        self.setting_button.setObjectName("setting_button")
        self.gridLayout.addWidget(self.setting_button, 0, 0, 1, 1)
        self.start_buttons = QtWidgets.QPushButton(self.ewqewrq)
        self.start_buttons.setObjectName("start_buttons")
        self.gridLayout.addWidget(self.start_buttons, 0, 2, 1, 1)
        self.load_button = QtWidgets.QPushButton(self.ewqewrq)
        self.load_button.setObjectName("load_button")
        self.gridLayout.addWidget(self.load_button, 0, 1, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.ewqewrq)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.setting_button.setText(_translate("MainWindow", "Setting"))
        self.start_buttons.setText(_translate("MainWindow", "Start"))
        self.load_button.setText(_translate("MainWindow", "Load"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

