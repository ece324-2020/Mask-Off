from PyQt5.QtWidgets import QMainWindow

from PyQt5 import QtWidgets
from view.ui_project_window import Ui_project_content
from view.project_window_detail import ProjectWindowDetail
from view.main_window import MainWindow




class MainView(QMainWindow):
    def __init__(self):
        super(MainView, self).__init__()
        self._ui_project = Ui_project_content()
        self._ui_project.setupUi(self)

        self._ui_main = MainWindow()

        self._ui_project_content = ProjectWindowDetail(self._ui_main, self._ui_project)
        self._ui_project.gridLayout_3.addWidget(self._ui_project_content)
        

        
        