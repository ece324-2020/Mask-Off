from PyQt5.QtWidgets import QApplication
import sys
from view.main_view import MainView
from torch import nn    
import torch
from view2.main_view import SplashScreen

class Baseline(nn.Module):

    def __init__(self):
        super(Baseline, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 3)
        self.pool1 = nn.MaxPool2d(2, 2, padding=1)
        self.conv2 = nn.Conv2d(10, 5, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(5, 5, 5)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(5 * 13 * 13, 100)
        self.bn = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 3)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(-1, 5 * 13 * 13)
        x = torch.relu(self.bn(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))

        return x
class App(QApplication):
    def __init__(self, sys_argv):
        super(App, self). __init__(sys_argv) 
        self._ui = SplashScreen()
        # self._ui1 = MainView()
        # self._ui1.show()


if __name__ == "__main__":
    app = App(sys.argv)
    sys.exit(app.exec_()) 