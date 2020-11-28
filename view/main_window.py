from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtWidgets
from view.ui_main_window import Ui_MainWindow
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self._ui_main = Ui_MainWindow()
        self._ui_main.setupUi(self)
        # self.setupUi(self)
        



        
        