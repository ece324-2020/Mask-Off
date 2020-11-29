from PIL import Image
import os
import glob
from process_labels import get_label


def save_cropped(dict, type):
    for file, boxes in dict.items():
        img = Image.open('images/' + file)
        # have to change to RGB otherwise can't save as jpeg
        img = img.convert('RGB')
        counter = 0
        for box in boxes:
            box = [int(value) for value in box]
            cropped = img.crop(box)
            cropped = cropped.resize((128, 128))
            cropped.save('output/' + type + '/' + file.split('.png')[0] + '_' + str(counter) + '.jpg', "JPEG")
            counter += 1


def main(pre_path="", freq=10, offsets=(0, 0, 0, 0), test_mode=True):

    # dim is (width, height)
    # shape is {filename: [[left, right, top, bot], [...], ...], ...}
    uncovered = {}
    covered = {}
    nose_unc = {}

    name = ("<name>", "</name>")
    width = ("<width>", "</width>")
    xmin = ("<xmin>", "</xmin>")
    xmax = ("<xmax>", "</xmax>")
    ymin = ("<ymin>", "</ymin>")
    ymax = ("<ymax>", "</ymax>")
    object = ("<object>", "</object>")
    filename = ("<filename>", "</filename>")

    os.chdir('Data/archive')

    for image_annotation in glob.glob('annotations/*.xml'):
        f = open(image_annotation, 'r')
        actual_filename = image_annotation.split('annotations/')[1].split(".xml")[0] + '.png'
        current_class = ""
        left = 0
        right = 0
        top = 0
        bottom = 0
        for line in f:
            if name[0] in line:
                current_class = get_label(line, name)
            elif xmin[0] in line:
                left = get_label(line, xmin)
            elif xmax[0] in line:
                right = get_label(line, xmax)
            elif ymin[0] in line:
                top = get_label(line, ymin)
            elif ymax[0] in line:
                bottom = get_label(line, ymax)
            elif object[1] in line:
                # save in the right place
                if current_class == "mask_weared_incorrect":
                    try:
                        nose_unc[actual_filename] += [(left, top, right, bottom)]
                    except KeyError:
                        nose_unc[actual_filename] = [(left, top, right, bottom)]

                elif current_class == "without_mask":
                    try:
                        uncovered[actual_filename] += [(left, top, right, bottom)]
                    except KeyError:
                        uncovered[actual_filename] = [(left, top, right, bottom)]

                if current_class == "with_mask":
                    try:
                        covered[actual_filename] += [(left, top, right, bottom)]
                    except KeyError:
                        covered[actual_filename] = [(left, top, right, bottom)]

                # clean for next one
                current_class = ""
                left = right = top = bottom = 0

    type = "nose_unc"
    save_cropped(nose_unc, "unc_nose")
    save_cropped(covered, "masked")
    save_cropped(uncovered, "no_mask")


if __name__ == '__main__':
    pre_path = "personal_data/improperly_masked_"
    # [left, top, right, bot]
    offsets = [50, 80, 50, 30]

    main(pre_path, freq=10, offsets=offsets, test_mode=False)
