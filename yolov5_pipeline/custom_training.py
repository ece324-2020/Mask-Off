import torch
from IPython.display import Image, clear_output, display  # to display images
from IPython.core.magic import register_line_cell_magic

from yolov5_pipeline.utils.datasets import *
from yolov5_pipeline.train import train
from yolov5_pipeline.utils.general import plot_results

import glob
import yaml
import subprocess

CONFIG_PATH = "datasets/data.yaml"




# define number of classes based on YAML
def defineNumClass(config):
    with open(config, 'r') as stream:
        num_classes = str(yaml.safe_load(stream)['nc'])


# we will use  /yolov5_pipeline/models/yolov5s.yaml file for model configuration
#customize iPython writefile so we can write variables

@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))

# Here, we are able to pass a number of arguments:
# - **img:** define input image size
# - **batch:** determine batch size
# - **epochs:** define the number of training epochs. (Note: often, 3000+ are common here!)
# - **data:** set the path to our yaml file
# - **cfg:** specify our model configuration
# - **weights:** specify a custom path to weights. (Note: you can download weights from the Ultralytics Google Drive [folder](https://drive.google.com/open?id=1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J))
# - **name:** result names
# - **nosave:** only save the final checkpoint
# - **cache:** cache images for faster training
# """



if __name__ == "__main__":
    print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
    defineNumClass(CONFIG_PATH)
    
    cmd = "python3 train.py --img 416 --batch 16 --epochs 500 --data '../data.yaml' --cfg ./models/custom_yolov5s.yaml --weights '' --name yolov5s_results  --cache"
    p = subprocess.Popen(cmd, shell=True)
    
    plot_results()
    Image(filename='result png file here', width=1000) 
    
    # displace inference on all test images
    for imageName in glob.glob('/content/yolov5/inference/output/*.jpg'): #assuming JPG
        display(Image(filename=imageName))
        print("\n")



# logs save in the folder "runs"
# you could use tensorboard as follows in colab
# %load_ext tensorboard
# %tensorboard --logdir runs
