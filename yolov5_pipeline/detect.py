import argparse
import time
from pathlib import Path

from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import threading
import sys
import os
# sys.path.append(dirname(realpath(__file__))


import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
# print(os.path.dirname(os.path.join(os.path.realpath(__file__), '/')))
# sys.path.append(os.path.dirname(os.path.join(os.path.realpath(__file__))))
# sys.path.append(os.path.dirname(__file__))
from yolov5_pipeline.models.experimental import attempt_load
from yolov5_pipeline.utils.datasets import LoadStreams, LoadImages
from yolov5_pipeline.utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from yolov5_pipeline.utils.plots import plot_one_box
from yolov5_pipeline.utils.torch_utils import select_device, load_classifier, time_synchronized


class VideoRendering():
    def __init__(self, label):
        super(VideoRendering, self).__init__()
        self.label = label

        self.weights = 'yolov5_pipeline/best.pt'
        self.source = '0'
        self.img_size = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.device = ''
        self.view_img = True
        self.save_txt = True
        self.save_conf= None
        self.classes = None
        self.agnostic_nms = None
        self.augment = None
        self.update = None
        self.project = 'yolov5_pipeline/runs/detect'
        self.name = 'exp'
        self.exist_ok = None
        self.fullscreen_ = 0
        self.isStart_ = 1
     
        
        
    def run(self, dataset, device, half, model, classify, webcam, save_dir, names, save_txt
            , view_img, save_img, colors):
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=self.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = Path(path[i]), '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = Path(path), '', im0s

                save_path = str(save_dir / p.name)
                txt_path = str(save_dir / 'labels' / p.stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))

                # Stream results
                if view_img:
                    cv2.namedWindow("YOLOV5", cv2.WINDOW_NORMAL)
                   
                    # cv2.imshow("YOLOV5", im0)
                    if(self.isStart_):
                        cv2.resizeWindow("YOLOV5", 1500, 1000)
                        cv2.moveWindow("YOLOV5", 0, 0)
                        self.isStart_ = 0
                    else:
                        pass
                    # if(self.fullscreen_):
                    #     cv2.setWindowProperty(str(p), cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    # else:
                    #     cv2.moveWindow(str(p), 0, 0)
                    #     cv2.resizeWindow(str(p), 1100,450)
                        
                    # cv2.moveWindow(str(p), 0, 0)
                    # cv2.resizeWindow(str(p), 1100,450)

             
                     
                    img = im0
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h,w,c = img.shape
                    print(h, " ", w, " ", c, " ", "아 제발", "\n")
                    # qImg = QtGui.QImage(img, w, h, w*c, cv2.QImage.Format_RGB888)
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # qimage = qimage2ndarray.array2qimage(frame)
                    qimage = QtGui.QImage(im0, im0.shape[1], im0.shape[0], QtGui.QImage.Format_RGB888)
                    pixmap = QtGui.QPixmap.fromImage(qimage).scaled(self.label.width(),self.label.height(),Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.label.setPixmap(pixmap)

                    # else:
                    #     win = QtWidgets.QWidget()
                    #     QtWidgets.QMessageBox.about(win, "Error", "Cannot read frame.")
                    #     print("cannot read frame.")
                    #     break
                    

                    
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        
                        raise StopIteration

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'images':
                        cv2.imwrite(save_path, im0)
                    else:
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fourcc = 'mp4v'  # output video codec
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                        vid_writer.write(im0)



    def detect(self, save_img=False):
        source, weights, view_img, save_txt, imgsz = self.source, self.weights, self.view_img, self.save_txt, self.img_size
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://'))

        # Directories
        save_dir = Path(increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(self.device)
        print(device)
        print("Here")
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz)
        else:
            save_img = True
            dataset = LoadImages(source, img_size=imgsz)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        # th = threading.Thread(target=self.run(dataset, device, half, model, classify, webcam, save_dir, names, save_txt
            # ,view_img, save_img, colors))
        self.run(dataset, device, half, model, classify, webcam, save_dir, names, save_txt
            ,view_img, save_img, colors)
        # th.start()


        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {save_dir}{s}")

        print('Done. (%.3fs)' % (time.time() - t0))


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
#     parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
#     parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
#     parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--view-img', action='store_true', help='display results')
#     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
#     parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
#     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--update', action='store_true', help='update all models')
#     parser.add_argument('--project', default='runs/detect', help='save results to project/name')
#     parser.add_argument('--name', default='exp', help='save results to project/name')
#     parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
#     opt = parser.parse_args()
#     print(opt)

    

    # with torch.no_grad():
    #     if opt.update:  # update all models (to fix SourceChangeWarning)
    #         for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
    #             detect()
    #             strip_optimizer(opt.weights)
    #     else:
    #         detect()
