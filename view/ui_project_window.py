# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ProjectWindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_project_content(object):
    def setupUi(self, project_content):
        project_content.setObjectName("project_content")
        project_content.resize(1119, 767)
        self.centralwidget = QtWidgets.QWidget(project_content)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_3.setObjectName("gridLayout_3")
        project_content.setCentralWidget(self.centralwidget)

        self.retranslateUi(project_content)
        QtCore.QMetaObject.connectSlotsByName(project_content)

    def retranslateUi(self, project_content):
        _translate = QtCore.QCoreApplication.translate
        project_content.setWindowTitle(_translate("project_content", "MainWindow"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    project_content = QtWidgets.QMainWindow()
    ui = Ui_project_content()
    ui.setupUi(project_content)
    project_content.show()
    sys.exit(app.exec_())

