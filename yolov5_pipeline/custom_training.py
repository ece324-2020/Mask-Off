
# * Install YOLOv5 dependencies
# * Download custom YOLOv5 object detection data
# * Write our YOLOv5 Training configuration
# * Run YOLOv5 training
# * Evaluate YOLOv5 performance
# * Visualize YOLOv5 training data
# * Run YOLOv5 inference on test images
# * Export saved YOLOv5 weights for future inference




import torch
from IPython.display import Image, clear_output  # to display images
from utils.google_utils import gdrive_download  # to download models/datasets

clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))





# define number of classes based on YAML
import yaml
with open("datasets/data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])

# Commented out IPython magic to ensure Python compatibility.
# this is the model configuration we will use for our tutorial 
# %cat /content/yolov5/models/yolov5s.yaml

#customize iPython writefile so we can write variables
from IPython.core.magic import register_line_cell_magic

@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))

