import torch
import torch.utils.data
from PIL import Image
import random
import torchvision.transforms as transforms

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

import utils

class ImagePatchTranslationLoader(torch.utils.data.Dataset):
    def __init__(self, image_filenames, patches_per_image, patch_size, max_offset, log=None):
        self.image_filenames = image_filenames
        self.patch_size = patch_size
        self.max_offset = max_offset
        self.log = log
        self.image_index = 0
        self.patches_per_image = patches_per_image
        self.last_image_index = -1
        self.last_image = None

    def __getitem__(self, index):
        # if self.log:
        #     worker_id = torch.utils.data.get_worker_info().id
        #     self.log.info('worker id: {} index: {}'.format(worker_id, index))

        image_index = index // self.patches_per_image
        image = self.last_image
        if image_index != self.last_image_index:
            if self.last_image:
                self.last_image.close()
            image_filename = self.image_filenames[image_index]
            image = Image.open(image_filename) # .convert('RGB')
            self.last_image_index = index
            self.last_image = image

        # Crop two randomly selected but overlapping squares of patch_size * patch_scale.
        # Scale them down by patch_scale to give two patches of patch_size.
        # TODO: Add some noise...
        # TODO: confidence GT...
        src_size = self.patch_size
        max_offset = self.max_offset
        #half_src_size = src_size // 2
        #quarter_src_size = half_src_size // 2

        image_w, image_h = image.size

        xa = random.randint(max_offset, image_w - src_size - max_offset)
        ya = random.randint(max_offset, image_h - src_size - max_offset)
        patch_a = image.crop((xa, ya, xa + src_size, ya + src_size))

        x_offset = random.randint(-max_offset, max_offset) # inclusive.
        y_offset = random.randint(-max_offset, max_offset)
        xb = xa + x_offset
        yb = ya + y_offset
        patch_b = image.crop((xb, yb, xb + src_size, yb + src_size))

        with torch.no_grad():
            patch_a = transforms.ToTensor()(patch_a)
            patch_b = transforms.ToTensor()(patch_b)

        #if self.log:
        #   self.log.info('{}:{}\t{}:{}'.format((x_offset / self.patch_scale), round(cx,3), (y_offset / self.patch_scale), round(cy, 3)))

        return patch_a, patch_b, torch.tensor([x_offset / max_offset, y_offset / max_offset, 0.5, 0.5])

    def __len__(self):
        return len(self.image_filenames) * self.patches_per_image