import torch
import torch.utils.data
from PIL import Image
import random
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

import utils

class ImagePatchTranslationLoader(torch.utils.data.Dataset):
    def __init__(self, image_filenames, patches_per_image, patch_size=17, patch_scale=4, log=None):
        self.image_filenames = image_filenames
        self.patch_size = patch_size
        self.patch_scale = patch_scale
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
        src_size = self.patch_size * self.patch_scale
        half_src_size = src_size // 2
        quarter_src_size = half_src_size // 2

        image_w, image_h = image.size
        xa = random.randint(quarter_src_size, image_w - src_size - quarter_src_size)
        ya = random.randint(quarter_src_size, image_h - src_size - quarter_src_size)

        x_offset = random.randint(-quarter_src_size, quarter_src_size)
        y_offset = random.randint(-quarter_src_size, quarter_src_size)

        xb = xa + x_offset
        yb = ya + y_offset

        patch_a = image.crop((xa, ya, xa + src_size, ya + src_size))
        patch_b = image.crop((xb, yb, xb + src_size, yb + src_size))

        with torch.no_grad():
            patch_a = transforms.ToTensor()(patch_a)
            patch_b = transforms.ToTensor()(patch_b)

            # Fit gaussian to difference of patches.
        xy1 = quarter_src_size
        xy2 = half_src_size + quarter_src_size
        center_patch = patch_a[:, xy1:xy2, xy1:xy2]
        d = np.zeros((half_src_size, half_src_size))
        min_d = 100
        for x in range(0, half_src_size):
            for y in range(0, half_src_size):
                patch_diff = center_patch - patch_a[:, y:half_src_size + y, x:half_src_size + x]
                patch_diff += torch.randn_like(patch_diff) * 0.1
                s = patch_diff.abs().sum()
                d[y,x] = s
                if s < min_d and s != 0:
                    min_d = s.item()

        min_d /= 1.2
        d[quarter_src_size, quarter_src_size] = min_d
        d = min_d / d

        mx, my, sigma = utils.fit2dGuassian(d, quarter_src_size, quarter_src_size)

        fig = plt.figure(figsize=(24, 8), dpi=112)
        ax1 = fig.add_subplot(131)
        ax1.imshow(patch_a.squeeze(0), cmap='gray')
        ax2 = fig.add_subplot(132)
        ax2.imshow(d, cmap='gray')
        mn = multivariate_normal((mx,my), sigma)
        xx, yy = np.mgrid[0:30, 0:30]
        pos = np.dstack((xx, yy))
        ax2.contour(xx, yy, mn.pdf(pos), colors='r', levels=np.linspace(0,0.01,6))

        ax3 = fig.add_subplot(133, projection='3d')
        x = range(0, half_src_size)
        y = range(0, half_src_size)
        x, y = np.meshgrid(x,y)
        ax3.plot_surface(x,y, d)
        plt.show()

        # patch_a = patch_a.resize((self.patch_size, self.patch_size), Image.BILINEAR)
        # patch_b = patch_b.resize((self.patch_size, self.patch_size), Image.BILINEAR)

        # scale patches on GPU

        patch_a = F.interpolate(patch_a.unsqueeze(0), size=[self.patch_size, self.patch_size], mode='bilinear', align_corners=True).squeeze()
        patch_b = F.interpolate(patch_b.unsqueeze(0), size=[self.patch_size, self.patch_size], mode='bilinear', align_corners=True).squeeze()

        #if self.log:
        #   self.log.info('{}:{}\t{}:{}'.format((x_offset / self.patch_scale), round(cx,3), (y_offset / self.patch_scale), round(cy, 3)))

        return patch_a, patch_b, torch.tensor([x_offset / quarter_src_size, y_offset / quarter_src_size, 0, 0])

    def __len__(self):
        return len(self.image_filenames) * self.patches_per_image