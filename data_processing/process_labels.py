import cv2
import glob
import os


def get_label(string, tags):
    return string.split(tags[0])[1].split(tags[1])[0]


def reflect_width(string, original_tags, target_tags, width_val):
    value = int(string.split(original_tags[0])[1].split(original_tags[1])[0])
    return fill_value(string, original_tags, target_tags, str(width_val - value))


def fill_value(string, original_tags, target_tags, filling):
    lst = string.split(original_tags[0])
    beginning = lst[0]
    lst = lst[1].split(original_tags[1])
    end = lst[1]
    return beginning + target_tags[0] + filling + target_tags[1] + end


def get_width(string):
    return int(string.split("<width>")[1].split("</width>")[0])


if __name__ == "__main__":
    # Process the labels first

    os.chdir("./Data/archive/annotations")

    improper = 0
    improper_label = "mask_weared_incorrect"
    name = ("<name>", "</name>")
    width = ("<width>", "</width>")
    xmin = ("<xmin>", "</xmin>")
    xmax = ("<xmax>", "</xmax>")
    filename = ("<filename>", "</filename>")

    labels = []
    width_val = 0
    save = False
    images_to_flip = []

    # Need to remove any files that may be there from running this previously
    print("Deleting following existing files...")
    for file in glob.glob("*flipped.xml"):
        print(file)
        os.remove(file)

    print()
    print("Remaining Files...")
    for file in glob.glob("*.xml"):
        print(file)

    for file in glob.glob("*.xml"):
        f = open(file, 'r')
        g = open(file.split('.xml')[0] + "_flipped.xml", 'w+')

        # We don't want to save by default
        save = False

        for line in f:
            copy = line
            if width[0] in line:
                width_val = get_width(line)

            elif filename[0] in line:
                copy = fill_value(line, filename, filename, file.split('.xml')[0] + "_flipped.png")

            # reflect all of the boxes
            elif xmin[0] in line:
                # Here we pass the opposite tags since it goes from max to min when flipping
                copy = reflect_width(line, xmin, xmax, width_val)
            elif xmax[0] in line:
                copy = reflect_width(line, xmax, xmin, width_val)

            elif name[0] in line:
                # Find which labels are in the dataset
                if get_label(line, name) not in labels:
                    labels.append(get_label(line, name))

                # Find how many are incorrect
                if get_label(line, name) == improper_label:
                    # Remember to save the modified file
                    save = True
                    print(file)
                    # Increment improper count
                    improper += 1
                    # Save the file name to be flipped
                    images_to_flip.append(file.split(".xml"))

            g.write(copy)

        f.close()
        g.close()
        if not save:
            os.remove(file.split('.xml')[0] + "_flipped.xml")

    test = "aeriuhgfvaedr<a>10</a> 1234"
    eval = reflect_width(test, ("<a>", "</a>"), ("<a>", "</a>"), 100)
    print(f"Replacing middle value works: {eval == 'aeriuhgfvaedr<a>90</a> 1234'}")

    print(f"Different labels used in this dataset: {labels}")
    print(f"Number of people improperly wearing mask: {improper}")

    # Flip the image horizontally
    os.chdir("../images")

    # Remove previous flip attempts
    for image in glob.glob("*_flipped.png"):
        os.remove(image)

    for image_path in glob.glob("*.png"):
        if image_path.split(".png") in images_to_flip:
            img = cv2.imread(image_path)
            # Horizontally flip
            img = cv2.flip(img, 1)
            cv2.imwrite(image_path.split(".png")[0] + "_flipped.png", img)

