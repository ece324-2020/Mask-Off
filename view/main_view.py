from PyQt5.QtWidgets import QMainWindow

from PyQt5 import QtWidgets
from view.project_window import ProjectWindow




class MainView(QMainWindow):
    def __init__(self):
        super(MainView, self).__init__()
        self._ui = ProjectWindow()
        
        


        
        