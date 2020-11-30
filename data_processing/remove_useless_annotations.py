import os
import glob

# this file is to executed in the file that holds both the images and annotation directories
# it will go through the photos and delete the annotations for those that were too poor to use

os.chdir('images')
files = []
for photo in glob.glob('*.jpg'):
    files.append(photo.split('.jpg')[0])

# do stuff
os.chdir('../annotations')
for annotation in glob.glob('*.xml'):
    if annotation.split('.xml')[0] not in files:
        os.remove(annotation)