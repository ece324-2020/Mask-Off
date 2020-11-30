# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 12:02:33 2020

@author: batuh
"""

import cv2
import torch
import numpy as np

from facenet_pytorch import MTCNN
from torchvision import transforms
from torch import nn

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
        self.fc2 = nn.Linear(100, 3)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(-1, 5 * 13 * 13)

        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x

def GetDim(input_channel=0):
    cap = cv2.VideoCapture(input_channel)
    ret, frame = cap.read()
    return (frame.shape[:2])

  
def GetFaceSquare(face_boxes, i, dim, offsets):
    left, top, right, bot = face_boxes[i]

    # Increase box size and check for edge cases
    left = int(max(left - offsets[0], 0))
    top = int(max(top - offsets[1], 0))
    right = int(min(right + offsets[2], dim[1]))
    bot = int(min(bot + offsets[3], dim[0]))
    height = bot - top
    width = right - left

    # Make box (approximately) a square
    if height > width:
        adding = (height - width) // 2
        left = int(max((left - adding), 0))
        right = int(min((right + adding), dim[1]))
    elif width > height:
        adding = (width - height) // 2
        top = int(max((top - adding), 0))
        bot = int(min((bot + adding), dim[0]))

    return left, top, right, bot


def LiveMaskDetector(model_path="baseline_rtpv.pt2", input_dim=None, input_channel=0,
                     means=[0.5142, 0.4515, 0.4201], stds=[0.2757, 0.2692, 0.2875],
                     offsets=[0, 0, 0, 0]):

    if type(input_dim) == type(None):
        dim = GetDim(input_channel)
    elif len(input_dim) != 2:
        print("Input dim must be of size 2: [height,width]")
        return -1

    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((128, 128)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(means, stds)])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    video = cv2.VideoCapture(input_channel)
    face_detector = MTCNN(keep_all=True, device=device)

    labels = ['Mask', 'No Mask', 'Uncovered Nose']
    labelcolour = [(0, 255, 0), (0, 0, 255), (0, 165, 255)]

    model = torch.load(model_path)
    model.to(device)
    model.eval()

    while True:
        ret, frame = video.read()

        framec = frame.copy()
        face_boxes, _ = face_detector.detect(framec)

        # Check if its none type
        if (type(face_boxes) != type(None)):
            # For every detected face...
            for i in range(len(face_boxes)):
                left, top, right, bot = GetFaceSquare(face_boxes, i, dim, offsets)

                face = framec[top:bot, left:right]
                # cv2 extracts in BGR order, our model is trained with RGB
                face[:, :, [0, 1, 2]] = face[:, :, [2, 1, 0]]

                # Change to tensor, resize etc.
                face = transform(face)
                # Add extra dim for model input
                face = face.unsqueeze(0)
                # If cuda exists use it
                face = face.to(device)

                # Get prediction, turn to 'probability'
                output = model(face)
                exped = torch.exp(output.squeeze().detach())
                tot = exped.sum()

                prob = exped / tot

                if device.type == 'cuda':
                    prob = prob.cpu().numpy()
                else:
                    prob = prob.numpy()
                pred = np.argmax(prob)

                # Draw the bounding square on frame
                frame = cv2.rectangle(frame, (left, top), (right, bot), labelcolour[pred], 2)
                # Add classification above
                title = labels[pred] + " " + str(round(prob[pred], 2))
                titlesize = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                titlex = left + (right - left) // 2 - titlesize[0][0] // 2
                titlex = max(0, titlex)
                cv2.putText(frame, title, (int(titlex), top), cv2.FONT_HERSHEY_SIMPLEX,
                            1, labelcolour[pred], 2)


        else:
            pass
        cv2.namedWindow("Mask Off", cv2.WINDOW_NORMAL)
        cv2.imshow('Mask Off', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            video.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    live = LiveMaskDetector()