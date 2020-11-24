from PyQt5 import uic, QtWidgets, QtGui
from PyQt5.QtCore import Qt
import os
from view.project_window_button import ProjectWindowButton

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/ui/ProjectWindowContentLayout.ui"

FORMCLASS = uic.loadUiType(DIR)[0]


class ProjectWindowDetail(QtWidgets.QWidget, FORMCLASS):
    def __init__(self, main, project):
        super(ProjectWindowDetail, self).__init__()
        self.setupUi(self)
        
        self.button = ProjectWindowButton(main, project)
        self.project_button_layout.addWidget(self.button)

        self.cur_path = os.path.dirname(os.path.abspath(__file__))
        self.main_folder_path = os.path.dirname(self.cur_path)
        
        self.logo_image = QtWidgets.QLabel("")
        self.logo_image.setPixmap(
            QtGui.QPixmap(
                os.path.join(
                        self.main_folder_path, 'assets/mask.png')).scaled(700,700,Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.logo_image.setAlignment(Qt.AlignHCenter)
        self.logo_layout.addWidget(self.logo_image)