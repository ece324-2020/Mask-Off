import torch
import cv2
from PIL import Image
from facenet_pytorch import MTCNN

def Find_Vid_dim(input_channel=0):
    cap = cv2.VideoCapture(input_channel)
    ret, frame = cap.read()
    print(frame.shape[:2])
    return frame.shape[:2]


def make_square(fheight, fwidth, left, right, top, bot, dim):
    if fheight > fwidth:
        adding = (fheight - fwidth) // 2
        left = int(max((left - adding), 0))
        right = int(min((right + adding), dim[1]))
    elif fwidth > fheight:
        adding = (fwidth - fheight) // 2
        top = int(max((top - adding), 0))
        bot = int((min(bot + adding), dim[0]))
    return left, right, top, bot


def get_bounding_square(mtcnn, img_, dim, offsets, square=True):
    boxes, _ = mtcnn.detect(img_)

    # Get bounding box edges
    left, top, right, bot = boxes[0]
    left = int(max(left - offsets[0], 0))
    top = int(max(top - offsets[1], 0))
    right = int(min(right + offsets[2], dim[1]))
    bot = int(min(bot + offsets[3], dim[0]))
    fheight = bot - top
    fwidth = right - left

    # To make bounding box a square
    if square:
        left, right, top, bot = make_square(fheight, fwidth, left, right, top, bot, dim)

    return left, right, top, bot


def save_image(img_, left, right, top, bot, im_cnt):
    face = img_[top:bot, left:right]
    face[:, :, [0, 1, 2]] = face[:, :, [2, 1, 0]]
    im_suffix = str(im_cnt) + ".jpg"
    im_name = pre_path + im_suffix
    save_image = Image.fromarray(face)
    save_image = save_image.resize((128, 128))
    save_image.save(im_name, "JPEG")
    return im_cnt + 1


def Video_to_pic(pre_path="", freq=10, offsets=(0, 0, 0, 0), test_mode=True):
    """
    To stop code from executing hit 'q'. The code will automatically make the
    bounding box around your face a (near) square.

    Parameters
    ----------
    path = str, optional
        Path to destination directory. Can include a prefix to images as well,
        if desired (Directory must already exist).
        Example : data/no_mask -> the resulting images will be in
        the 'data' directory and be named no_mask0.jpg,no_mask1.jpg, etc.
    input_channel : int, optional
        Channel from which cv2 accesses your camera. Mine happened to be 1 but
        the default is 0.
    freq : int, optional
        Frequency of image from frames. The default is 10, so one in every 10
        frames is saved as an image.
    offsets : list of length 4,
        [left,top,right,bottom] List contains information as to how many pixels
        to offset in each direction in order given above. Play around with
        the values while setting test_mode to true until the box completely
        covers your face.
        The default is [0,0,0,0].
    input_dim : list of length 2, optional
        Dimensions of input video like [height,width]. The default is None,
        if not input the code will find this out automatically each time.
    test_mode : boolean, optional
        When set to true, images will not be saved. The default is True.

    """
    # if input_dim == None:
    #     dim = Find_Vid_dim(input_channel)
    # elif len(input_dim) == 2:
    #     dim = input_dim
    # else:
    #     print("Incorrect input_dim input")
    #     return 0

    cap = cv2.VideoCapture("improperly_masked.mp4")
    ret, frame = cap.read()
    print(frame.shape[:2])

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
                print(frame_cnt, freq)
                img_ = frame.copy()
                left, right, top, bot = get_bounding_square(mtcnn, img_, dim, offsets)

                # Draw box
                # frame = cv2.rectangle(frame, (left, bot), (right, top), (0, 0, 255), 2)

                # Save image
                if test_mode is False:
                    im_cnt = save_image(img_, left, right, top, bot, im_cnt)

        # This occurs if no face is in the frame
        except TypeError:
            pass

        cv2.imshow('Mask detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video.release()
            cv2.destroyAllWindows()
            return 0


if __name__ == '__main__':
    pre_path = "personal_data/improperly_masked_"
    # [left, top, right, bot]
    offsets = [50, 80, 50, 30]

    Video_to_pic(pre_path, freq=10, offsets=offsets, test_mode=False)
