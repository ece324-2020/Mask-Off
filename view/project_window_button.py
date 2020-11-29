from PyQt5 import uic, QtWidgets
import os
import sys
import torch
from view.ui_main_window import Ui_MainWindow




DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/ui/ProjectWindow_buttons.ui"

# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from yolov5_pipeline.detect import VideoRendering


FORMCLASS = uic.loadUiType(DIR)[0]


class ProjectWindowButton(QtWidgets.QWidget, FORMCLASS):
    def __init__(self, main, project):
        super(ProjectWindowButton, self).__init__()
        self.setupUi(self)
        self.setting_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self.start_button.setEnabled(True)
        
        self.setting_button.clicked.connect(self.on_setting_button)
        self.load_button.clicked.connect(self.on_load_button)
        self.start_button.clicked.connect(self.on_start_button)

        self._main = main
        # self._main = Ui_MainWindow()
     
        self._project = project

    def on_setting_button(self):
        pass
    
    def on_load_button(self):
        pass

    def on_start_button(self):
        self._label = QtWidgets.QLabel()
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self._label)
        self._videoRendering = VideoRendering(self._label)
        
        # self._main._ui_main.setupUi(self._main)
        # self._main._ui_main.gridLayout_3.addWidget(self._label)
        # self._main.show()

        with torch.no_grad():
            self._videoRendering.detect()

        

    