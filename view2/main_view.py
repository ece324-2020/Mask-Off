import sys, os
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QPushButton, QSizePolicy
from PyQt5.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime, QMetaObject, QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QEvent)
from PyQt5.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont, QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter, QPixmap, QRadialGradient)
from PyQt5.QtWidgets import *
import torch
from yolov5_pipeline.detect import VideoRendering
SPLASH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/ui/splash_screen.ui"
FORMCLASS_SPLASH = uic.loadUiType(SPLASH_DIR)[0]

GUI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/ui/GUI.ui"
FORMCLASS_GUI = uic.loadUiType(GUI_DIR)[0]
print(FORMCLASS_GUI)
counter = 0
count = 1 
GLOBAL_STATE = 0 
GLOBAL_TITLE_BAR = True
YOLO_ON = False
OWN_ON = False
class MainWindow(QtWidgets.QMainWindow, FORMCLASS_GUI):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        # self.setWindowFlag(QtCore.Qt.WA_MouseTracking)

        self.GLOBAL_STATE = 0
        self.GLOBAL_TITLE_BAR = True
        self.cur_path = os.path.dirname(os.path.abspath(__file__))
        self.main_folder_path = os.path.dirname(self.cur_path)
        self.label.setPixmap(
                     QtGui.QPixmap(
                        os.path.join(
                            self.main_folder_path, 'assets/mask.png')).scaled(700,700,Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.label.setAlignment(Qt.AlignHCenter)

        self.min_label = QtWidgets.QLabel("")
        self.max_label = QtWidgets.QLabel("")
        self.quit_label = QtWidgets.QLabel("")
    
        
        close_pic = QtGui.QPixmap(os.path.join(self.main_folder_path, 'assets/icons/cli-x.png')).scaled(700,700,Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.btn_close.setIcon(
            QtGui.QIcon(close_pic))
        self.btn_close.setIconSize(close_pic.size())
        # self.btn_close.setAlignment(Qt.AlignHCenter)

        
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.btn_toggle_menu.clicked.connect(lambda: self.toggleMenu(220, True))
        self.uiDefinitions()
        self.stackedWidget.setMinimumWidth(20)

        self.addNewMenu("HOME", "btn_home", os.path.join(self.main_folder_path, 'assets/icons/cli-home.png'), True)
        self.addNewMenu("Yolo_v5", "btn_yolov5", "url(:/16x16/icons/16x16/cil-user-follow.png)", True)
        self.addNewMenu("Different Object Detection Method", "btn_own", "", True)
        self.addNewMenu("Setting", "btn_widgets", "url(:/16x16/icons/16x16/cil-equalizer.png)", False)
        self.selectStandardMenu("btn_home")
        self.stackedWidget.setCurrentWidget(self.page_home)
        self.userIcon("WM", "", True)
        self.tableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.frame_label_top_btns.mouseMoveEvent = self.moveWindow

    def returStatus(self):
        return self.GLOBAL_STATE

    def setStatus(self, status):
        global GLOBAL_STATE
        GLOBAL_STATE = status

    def enableMaximumSize(self, width, height):
        if width != '' and height != '':
            self.setMaximumSize(QSize(width, height))
            self.ui.frame_size_grip.hide()
            self.ui.btn_maximize_restore.hide()

    def maximize_restore(self):
        global GLOBAL_STATE
        status = GLOBAL_STATE
        if status == 0:
            self.showMaximized()
            GLOBAL_STATE = 1
            self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
            self.btn_maximize_restore.setToolTip("Restore")  
            self.btn_maximize_restore.setIcon(QtGui.QIcon(u"../assets/icons/cil-window-restore.png"))
            self.frame_top_btns.setStyleSheet("background-color: rgb(87, 88, 81)")
            self.frame_size_grip.hide()
        else:
            GLOBAL_STATE = 0
            self.showNormal()
            self.resize(self.width()+1, self.height()+1)
            self.horizontalLayout.setContentsMargins(10, 10, 10, 10)
            self.btn_maximize_restore.setToolTip("Maximize")  
            self.btn_maximize_restore.setIcon(QtGui.QIcon(u"../assets/icons/cil-window-maximize.png"))
            self.frame_top_btns.setStyleSheet("background-color: rgb(87, 88, 81)")
            self.frame_size_grip.show()

    def moveWindow(self, event):
            # IF MAXIMIZED CHANGE TO NORMAL
            if self.returStatus() == 1:
                self.maximize_restore()

            # MOVE WINDOW
            if event.buttons() == Qt.LeftButton:
                self.move(self.pos() + event.globalPos() - self.dragPos)
                self.dragPos = event.globalPos()
                event.accept()

    def labelTitle(self, text):
        self.ui.label_title_bar_top.setText(text)

    # LABEL DESCRIPTION
    def labelDescription(self, text):
        self.ui.label_top_info_1.setText(text)

    def mousePressEvent(self, event):
        self.dragPos = event.globalPos()
        if event.buttons() == Qt.LeftButton:
            print('Mouse click: LEFT CLICK')
        if event.buttons() == Qt.RightButton:
            print('Mouse click: RIGHT CLICK')
        if event.buttons() == Qt.MidButton:
            print('Mouse click: MIDDLE BUTTON')

    def removeTitleBar(self, status):
        
        self.GLOBAL_TITLE_BAR = status

    def uiDefinitions(self):
        def dobleClickMaximizeRestore(event):
            # IF DOUBLE CLICK CHANGE STATUS
            if event.type() == QtCore.QEvent.MouseButtonDblClick:
                QtCore.QTimer.singleShot(250, lambda: UIFunctions.maximize_restore(self))

        ## REMOVE ==> STANDARD TITLE BAR
        if self.GLOBAL_TITLE_BAR:
            self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
            self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
            self.frame_label_top_btns.mouseDoubleClickEvent = dobleClickMaximizeRestore
        else:
            self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
            self.frame_label_top_btns.setContentsMargins(8, 0, 0, 5)
            self.frame_label_top_btns.setMinimumHeight(42)
            self.frame_icon_top_bar.hide()
            self.frame_btns_right.hide()
            self.frame_size_grip.hide()


        ## SHOW ==> DROP SHADOW
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(17)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QColor(0, 0, 0, 150))
        self.frame_main.setGraphicsEffect(self.shadow)

        ## ==> RESIZE WINDOW
        self.sizegrip = QSizeGrip(self.frame_size_grip)
        self.sizegrip.setStyleSheet("width: 20px; height: 20px; margin 0px; padding: 0px;")

        ### ==> MINIMIZE
        self.btn_minimize.clicked.connect(lambda: self.showMinimized())

        ## ==> MAXIMIZE/RESTORE
        self.btn_maximize_restore.clicked.connect(lambda: self.maximize_restore())

        ## SHOW ==> CLOSE APPLICATION
        self.btn_close.clicked.connect(lambda: self.close())

        # WIDGET TO MOVE
    def userIcon(self, initialsTooltip, icon, showHide):
        if showHide:
            # SET TEXT
            # self.label_user_icon.setText(initialsTooltip)

            # SET ICON
            if icon:
                style = self.label_user_icon.styleSheet()
                setIcon = "QLabel { background-image: " + icon + "; }"
                self.ui.label_user_icon.setStyleSheet(style + setIcon)
                self.ui.label_user_icon.setText('')
                self.ui.label_user_icon.setToolTip(initialsTooltip)
        else:
            self.ui.label_user_icon.hide()


    def selectStandardMenu(self, widget):
        for w in self.frame_left_menu.findChildren(QPushButton):
            if w.objectName() == widget:
                w.setStyleSheet(self.selectMenu(w.styleSheet()))
    
    def toggleMenu(self, maxWidth, enable):
        if enable:
            # GET WIDTH
            width = self.frame_left_menu.width()
            maxExtend = maxWidth
            standard = 70

            # SET MAX WIDTH
            if width == 70:
                widthExtended = maxExtend
            else:
                widthExtended = standard

            # ANIMATION
            self.animation = QPropertyAnimation(self.frame_left_menu, b"minimumWidth")
            self.animation.setDuration(300)
            self.animation.setStartValue(width)
            self.animation.setEndValue(widthExtended)
            self.animation.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
            self.animation.start()

    def addNewMenu(self, name, objName, icon, isTopMenu):
        font = QFont()
        font.setFamily(u"Segoe UI")
        button = QPushButton(str(count))
        button.setObjectName(objName)
        sizePolicy3 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(button.sizePolicy().hasHeightForWidth())
        button.setSizePolicy(sizePolicy3)
        button.setMinimumSize(QSize(0, 70))
        button.setLayoutDirection(Qt.LeftToRight)
        button.setFont(font)
        button.setStyleSheet(Style.style_bt_standard.replace('ICON_REPLACE', icon))
        button.setText(name)
        button.setToolTip(name)
        button.clicked.connect(self.Button)

        if isTopMenu:
            self.layout_menus.addWidget(button)
        else:
            self.layout_menu_bottom.addWidget(button)


    def Button(self):
        # GET BT CLICKED
        btnWidget = self.sender()
        
        # PAGE HOME
        if btnWidget.objectName() == "btn_home":
            self.stackedWidget.setCurrentWidget(self.page_home)
            self.resetStyle("btn_home")
            btnWidget.setStyleSheet(self.selectMenu(btnWidget.styleSheet()))

        # PAGE NEW USER
        if btnWidget.objectName() == "btn_yolov5":
            self.stackedWidget.setCurrentWidget(self.page_yolo)
            global YOLO_ON
            if YOLO_ON != True:
                YOLO_ON = False
                self._label = QtWidgets.QLabel()
                self._yolo_videoRendering = VideoRendering(self._label)
                self.yolo_layout.addWidget(self._label)
                with torch.no_grad():
                    self._yolo_videoRendering.detect()
            self.resetStyle("btn_yolov5")
            btnWidget.setStyleSheet(self.selectMenu(btnWidget.styleSheet()))

        # PAGE WIDGETS
        if btnWidget.objectName() == "btn_own":
            self.stackedWidget.setCurrentWidget(self.page_own)
            self.resetStyle("btn_own")
            btnWidget.setStyleSheet(self.selectMenu(btnWidget.styleSheet()))
        
        if btnWidget.objectName() == "btn_widgets":
            self.stackedWidget.setCurrentWidget(self.page_widgets)
            self.resetStyle("btn_widgets")
            btnWidget.setStyleSheet(self.selectMenu(btnWidget.styleSheet()))

    def selectMenu(self, getStyle):
        select = getStyle + ("QPushButton { border-right: 7px solid rgb(87, 88, 81); }")
        return select

    def resetStyle(self, widget):
        for w in self.frame_left_menu.findChildren(QPushButton):
            if w.objectName() != widget:
                w.setStyleSheet(self.deselectMenu(w.styleSheet()))

    def deselectMenu(self, getStyle):
        deselect = getStyle.replace("QPushButton { border-right: 7px solid rgb(87, 88, 81); }", "")
        return deselect

    def eventFilter(self, watched, event):
        if watched == self.le and event.type() == QtCore.QEvent.MouseButtonDblClick:
            print("pos: ", event.pos())

    def keyPressEvent(self, event):
        print('Key: ' + str(event.key()) + ' | Text Press: ' + str(event.text()))

    def resizeEvent(self, event):
        self.resizeFunction()
        return super(MainWindow, self).resizeEvent(event)

    def resizeFunction(self):
        print('Height: ' + str(self.height()) + ' | Width: ' + str(self.width()))

class Style():

    style_bt_standard = (
    """
    QPushButton {
        background-image: ICON_REPLACE;
        background-position: left center;
        background-repeat: no-repeat;
        border: none;
        border-left: 28px solid rgb(27, 29, 35);
        background-color: rgb(27, 29, 35);
        text-align: left;
        padding-left: 45px;
    }
    QPushButton[Active=true] {
        background-image: ICON_REPLACE;
        background-position: left center;
        background-repeat: no-repeat;
        border: none;
        border-left: 28px solid rgb(27, 29, 35);
        border-right: 5px solid rgb(44, 49, 60);
        background-color: rgb(27, 29, 35);
        text-align: left;
        padding-left: 45px;
    }
    QPushButton:hover {
        background-color: rgb(33, 37, 43);
        border-left: 28px solid rgb(33, 37, 43);
    }
    QPushButton:pressed {
        background-color: rgb(85, 170, 255);
        border-left: 28px solid rgb(85, 170, 255);
    }
    """
    )


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



