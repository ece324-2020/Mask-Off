import torch
import cv2
from PIL import Image
from facenet_pytorch import MTCNN
from recorded_video_to_data import Find_Vid_dim, get_bounding_square, save_image
from process_labels import fill_value


def create_annotation(left, right, top, bot, class_type, frame_num, dim):
    f = open('../template.xml', 'r')
    file = class_type + "_" + str(frame_num) + '.xml'
    g = open(file, 'w')
    name = ("<name>", "</name>")
    width = ("<width>", "</width>")
    height = ("<height>", "</height>")
    xmin = ("<xmin>", "</xmin>")
    xmax = ("<xmax>", "</xmax>")
    ymin = ("<ymin>", "</ymin>")
    ymax = ("<ymax>", "</ymax>")
    filename = ("<filename>", "</filename>")
    options = [filename, xmin, xmax, ymin, ymax, name, width, height]
    fillers = [file.split('.xml')[0] + ".png", str(left), str(right), str(top), str(bot), class_type, str(dim[1]), str(dim[0])]

    for line in f:
        copy = line
        for i, option in enumerate(options):
            if option[0] in line:
                copy = fill_value(line, option, option, fillers[i])
        g.write(copy)

    f.close()
    g.close()


def main(freq=10, offsets=(0, 0, 0, 0), test_mode=True, class_type="mask_weared_incorrect", video="improperly_masked.mp4"):
    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()

    dim = frame.shape[:2]

    frame_cnt = 0
    im_cnt = 0

    video = cap

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)

    while True:
        ret, frame = video.read()
        frame_cnt = frame_cnt + 1
        try:
            if frame_cnt % freq == 0:
                img_ = frame.copy()
                left, right, top, bot = get_bounding_square(mtcnn, img_, dim, offsets, False)

                # Save image
                if test_mode is False:
                    img_[:, :, [0, 1, 2]] = img_[:, :, [2, 1, 0]]
                    image = Image.fromarray(img_)
                    image.save(class_type + "_" + str(frame_cnt) + '.jpg', "JPEG")
                    create_annotation(left, right, top, bot, class_type, frame_cnt, dim)

        # This occurs if no face is in the frame
        except TypeError:
            pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            video.release()
            cv2.destroyAllWindows()
            return 0


if __name__ == '__main__':
    pre_path = "personal_data/improperly_masked_"
    offsets = [0 for i in range(3)]

    # class_type is what will be put in the annotation
    # video is the path to the video that you want to use

    # Note that this doesn't actually produce the format that yolo needs, but it can easily converted online
    # using roboflow or something similar.
    # Additionally, this doesn't put the images and the annotations into their own file, you'll have to do that yourself

    main(freq=10, test_mode=False, class_type="without_mask", video="no_mask.mp4")
