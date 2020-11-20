from PyQt5.QtWidgets import QMainWindow

from PyQt5 import QtWidgets
from view.main_window import Ui_MainWindow
from view.project_window import Ui_MainProjectWidget




class MainView(QMainWindow):
    def __init__(self):
        super(MainView, self).__init__()
        # self._ui_project = Ui_MainProjectWidget()
        self._ui_main = Ui_MainWindow()
     
        self._ui_main.setupUi(self)
        # self._ui_project.setupUi(self)



        
        