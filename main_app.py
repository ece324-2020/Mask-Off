from PyQt5.QtWidgets import QApplication
import sys
from view.main_view import MainView

class App(QApplication):
    def __init__(self, sys_argv):
        super(App, self). __init__(sys_argv) 
        self._ui = MainView()
        self._ui.show()


if __name__ == "__main__":
    app = App(sys.argv)
    sys.exit(app.exec_()) 