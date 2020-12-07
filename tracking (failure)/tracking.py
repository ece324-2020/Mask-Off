import cv2
import sys
import torch
from facenet_pytorch import MTCNN

"""
    Note this file doesn't actually work. For some reason the tracking box becomes really tall and narrow
    and I can't figure out why.
    It is not used anywhere else in the repo but is included as a proof of work.
"""

def get_bounding_box(frame, mtcnn, dim, offsets):
    img_ = frame.copy()
    boxes, _ = mtcnn.detect(img_)
    try:
        if len(boxes) == 1:
            # Get bounding box edges
            left, top, right, bot = boxes[0]
            left = int(max(left - offsets[0], 0))
            top = int(max(top - offsets[1], 0))
            right = int(min(right + offsets[2], dim[1]))
            bot = int(min(bot + offsets[3], dim[0]))
            fheight = bot - top
            fwidth = right - left

            # To make bounding box a square
            if fheight > fwidth:
                adding = (fheight - fwidth) // 2
                left = int(max((left - adding), 0))
                right = int(min((right + adding), dim[1]))
            elif fwidth > fheight:
                adding = (fwidth - fheight) // 2
                top = int(max((top - adding), 0))
                bot = int((min(bot + adding), dim[0]))
            return left, top, right - left, bot - top
    except TypeError:
        return 0, 0, 0, 0


if __name__ == '__main__':
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    # tracker_type = tracker_types[2]
    tracker_type = tracker_types[0]
    tracker = cv2.TrackerKCF_create()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)
    tracking = False
    offsets = (0, 0, 0, 0)

    # Read video
    video = cv2.VideoCapture(0)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # Define an initial bounding box
    bbox = (287, 23, 86, 320)

    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        dim = (frame.shape[:2])

        if tracking:
            # Start timer
            timer = cv2.getTickCount()

            # Update tracker
            ok, bbox = tracker.update(frame)

            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

            # Draw bounding box
            if ok:
                # Tracking success
                print("bbox after update")
                print(bbox)
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))/ 4

                # This is where we could check to see if the mask wearing status has changed


                cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)
            else:
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        #
        # # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display result
        cv2.imshow("Tracking", frame)

        if cv2.waitKey(33) == ord('a'):
            # For debugging purposes, this file won't detect faces until it is prompted by the user
            # In a production environment, this would happen automatically

            bbox = get_bounding_box(frame, mtcnn, dim, offsets)

            # Pass cropped image into network to classify and then store results somewhere

            print("network")
            print(bbox)
            bbox = cv2.selectROI(frame, False)
            print("roi")
            print(bbox)
            tracking = True

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break