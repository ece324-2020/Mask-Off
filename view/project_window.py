from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtWidgets
from view.ui_project_window import Ui_project_content
from view.project_window_detail import ProjectWindowDetail
class ProjectWindow(QMainWindow):
    def __init__(self):
        super(ProjectWindow, self).__init__()
        self._ui_project = Ui_project_content()
        self._ui_project.setupUi(self)
        # self._ui_project.setupUi(self)
        self._ui_project_content = ProjectWindowDetail()
        self._ui_project.gridLayout_3.addWidget(self._ui_project_content)



        
        