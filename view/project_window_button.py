from PyQt5 import uic, QtWidgets
import os
import sys

from view.main_window import MainWindow

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/ui/ProjectWindow_buttons.ui"
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# from yolov5_pipeline.detect import detect


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
        self._project = project

    def on_setting_button(self):
        pass
    
    def on_load_button(self):
        pass

    def on_start_button(self):
        self._main.show()
        # self._project.close()
        
        

    