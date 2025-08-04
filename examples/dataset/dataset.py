import numpy as np
from PIL import Image

import mindspore.dataset.vision as vision
from mindspore.dataset import GeneratorDataset

class MyDataset():
    def __init__(self):
        self.input_ids = Image.open("./image.jpg").convert('RGB')
        self.to_tensor = vision.ToTensor()
        self.resize = vision.Resize((224, 224))
        mean_vec = [0.485, 0.456, 0.406]
        std_vec = [0.229, 0.224, 0.225]
        self.normalize = vision.Normalize(mean=mean_vec, std=std_vec, is_hwc=False)

    def __len__(self):
        return 10

    def __getitem__(self, index):
        img_resize = self.resize(self.input_ids)
        img_resize = self.to_tensor(img_resize)
        img_resize = self.normalize(img_resize)
        return img_resize
