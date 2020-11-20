from PyQt5 import uic, QtWidgets
import os


DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/ui/ProjectWindowContentLayout.ui"

FORMCLASS = uic.loadUiType(DIR)[0]


class ProjectWindowDetail(QtWidgets.QWidget, FORMCLASS):
    def __init__(self):
        super(ProjectWindowDetail, self).__init__()
        self.setupUi(self)
        s