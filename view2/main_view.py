import sys, os
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime, QMetaObject, QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QEvent)
from PyQt5.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont, QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter, QPixmap, QRadialGradient)
from PyQt5.QtWidgets import *


SPLASH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/ui/splash_screen.ui"
FORMCLASS_SPLASH = uic.loadUiType(SPLASH_DIR)[0]

# SPLASH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/ui/splash_screen.ui"
# FORMCLASS_SPLASH = uic.loadUiType(SPLASH_DIR)[0]

counter = 0


# class MainWindow(QMainWindow, MAIN_DIR):
#     def __init__(self):
#         super(MainWindow, self).__init__()
#         self.setupUi(self)
#         pass

class SplashScreen(QtWidgets.QMainWindow, FORMCLASS_SPLASH):
    def __init__(self):
        super(SplashScreen, self).__init__()
        self.setupUi(self)

        self.setWindowFlag(QtCore.Qt.SplashScreen)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.progress)

        self.timer.start(35)

        self.label_description.setText("<strong>WELCOME</strong> TO MY APPLICATION")
        QtCore.QTimer.singleShot(1500, lambda: self.label_description.setText("<strong>LOADING</strong> DATABASE"))
        QtCore.QTimer.singleShot(3000, lambda: self.label_description.setText("<strong>LOADING</strong> USER INTERFACE"))

        self.show()


    def progress(self):

        global counter
        self.progressBar.setValue(counter)
        if counter > 100:
            self.timer.stop()
            self.main = MainWindow()
            self.main.show()

            self.close()

        counter += 1



