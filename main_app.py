from PyQt5.QtWidgets import QApplication
import sys
from view.main_view import MainView

from view2.main_view import SplashScreen

class App(QApplication):
    def __init__(self, sys_argv):
        super(App, self). __init__(sys_argv) 
        self._ui = SplashScreen()
        # self._ui1 = MainView()
        # self._ui1.show()


if __name__ == "__main__":
    app = App(sys.argv)
    sys.exit(app.exec_()) 