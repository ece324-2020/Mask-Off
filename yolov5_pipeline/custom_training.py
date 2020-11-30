import torch
from IPython.display import Image, clear_output  # to display images

clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))





# define number of classes based on YAML
import yaml
with open("datasets/data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])


# we will use  /yolov5_pipeline/models/yolov5s.yaml file for model configuration
#customize iPython writefile so we can write variables
from IPython.core.magic import register_line_cell_magic

@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))

