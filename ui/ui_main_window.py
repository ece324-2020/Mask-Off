# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1119, 767)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_3.setObjectName("gridLayout_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1119, 28))
        self.menuBar.setObjectName("menuBar")
        self.menuLog_Information = QtWidgets.QMenu(self.menuBar)
        self.menuLog_Information.setObjectName("menuLog_Information")
        self.menuSetting_Info = QtWidgets.QMenu(self.menuBar)
        self.menuSetting_Info.setObjectName("menuSetting_Info")
        self.menuChange_Data_Type = QtWidgets.QMenu(self.menuBar)
        self.menuChange_Data_Type.setObjectName("menuChange_Data_Type")
        MainWindow.setMenuBar(self.menuBar)
        self.menuLog_Information.addSeparator()
        self.menuBar.addAction(self.menuLog_Information.menuAction())
        self.menuBar.addAction(self.menuSetting_Info.menuAction())
        self.menuBar.addAction(self.menuChange_Data_Type.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.menuLog_Information.setTitle(_translate("MainWindow", "Log Information"))
        self.menuSetting_Info.setTitle(_translate("MainWindow", "Setting Info"))
        self.menuChange_Data_Type.setTitle(_translate("MainWindow", "Change Data Type"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

