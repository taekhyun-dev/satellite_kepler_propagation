# simserver/fl/model.py
import os
from torchvision.models import mobilenet_v3_small

def new_model_skeleton(num_classes: int):
    return mobilenet_v3_small(num_classes=int(os.getenv("FL_NUM_CLASSES", str(num_classes))))
