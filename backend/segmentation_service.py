import os
import sys
from PIL import Image
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_data_inference.unet import Unet


class SegmentationService:
    def __init__(self, model_path=None, num_classes=3, backbone='vgg16', use_cuda=True):
        self.model = None
        self.model_path = model_path
        self.num_classes = num_classes
        self.backbone = backbone
        self.use_cuda = use_cuda
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        self.model = Unet(model_path=model_path, num_classes=self.num_classes, backbone=self.backbone, cuda=self.use_cuda)
        self.model_path = model_path
        return True

    def set_model(self, model_path):
        return self.load_model(model_path)

    def segment_rgb(self, image_np):
        img = Image.fromarray(image_np)
        result = self.model.detect_image_ui(img)
        return np.array(result)

    def segment_mask(self, image_np):
        img = Image.fromarray(image_np)
        mask = self.model.get_segmentation_mask(img)
        return mask

