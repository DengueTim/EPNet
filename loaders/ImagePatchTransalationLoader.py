import torch
import torch.utils.data
from PIL import Image
import random
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt

class ImagePatchTranslationLoader(torch.utils.data.Dataset):
    def __init__(self, image_filenames, patches_per_image, patch_size=17, patch_scale=8, log=None):
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

        image_w, image_h = image.size
        xa = random.randint(half_src_size, image_w - half_src_size - src_size)
        ya = random.randint(half_src_size, image_h - half_src_size - src_size)

        x_offset = random.randint(-half_src_size, half_src_size)
        y_offset = random.randint(-half_src_size, half_src_size)

        xb = xa + x_offset
        yb = ya + y_offset

        patch_a = image.crop((xa, ya, xa + src_size, ya + src_size))
        patch_b = image.crop((xb, yb, xb + src_size, yb + src_size))

        with torch.no_grad():
            patch_a = transforms.ToTensor()(patch_a)
            patch_b = transforms.ToTensor()(patch_b)

        patch_a = F.interpolate(patch_a.unsqueeze(0), size=[self.patch_size, self.patch_size], mode='bilinear', align_corners=True).squeeze()
        patch_b = F.interpolate(patch_b.unsqueeze(0), size=[self.patch_size, self.patch_size], mode='bilinear', align_corners=True).squeeze()

        # xy1 = half_src_size // 2
        # xy2 = half_src_size + xy1
        # center_patch = patch_a[xy1:xy2, xy1:xy2]
        # d = np.zeros((half_src_size, half_src_size))
        # for x in range(0, half_src_size):
        #     for y in range(0, half_src_size):
        #         patch_diff = center_patch - patch_a[y:half_src_size + y, x:half_src_size + x]
        #         d[y,x] = patch_diff.abs().sum()

        # half_patch_size = 1 + self.patch_size // 2
        # xy1 = half_patch_size - 5
        # xy2 = half_patch_size + 4
        # center_patch = patch_a[xy1:xy2, xy1:xy2]
        # d = np.zeros((half_patch_size, half_patch_size))
        # for x in range(0, half_patch_size):
        #     for y in range(0, half_patch_size):
        #         patch_diff = center_patch - patch_a[y:half_patch_size + y, x:half_patch_size + x]
        #         d[y, x] = patch_diff.abs().sum()
        # #plt.imshow(patch_a.permute(1, 2, 0))
        # plt.imshow(patch_a, cmap='gray')
        # plt.show()
        # plt.imshow(d, cmap='gray')
        # plt.show()
#        patch_a = patch_a.resize((self.patch_size, self.patch_size), Image.BILINEAR)
#        patch_b = patch_b.resize((self.patch_size, self.patch_size), Image.BILINEAR)



        # WIP: Some kind of measure of how smooth the patch is around its center
        # Like higher gradients(more texture) should lead to a more confidant prediction.
        xy1 = self.patch_size // 2 - 4
        xy2 = self.patch_size // 2 + 5
        center_patch = patch_a[xy1:xy2, xy1:xy2]
        sum_of_err_x = 0;
        sum_of_err_y = 0;
        offsets = [1, -1, 2, -2, 3, -3]
        for i in offsets:
            shifted_patch = patch_a[xy1:xy2, xy1 + i:xy2 + i]
            diff = center_patch - shifted_patch
            sum_of_err_x += diff.abs().sum().item() / len(offsets)
            shifted_patch = patch_a[xy1 + 1:xy2 + 1, xy1:xy2]
            diff = center_patch - shifted_patch
            sum_of_err_y += diff.abs().sum().item() / len(offsets)

        cx = 1 / (1 + sum_of_err_x)
        cy = 1 / (1 + sum_of_err_y)

        #if self.log:
        #   self.log.info('{}:{}\t{}:{}'.format((x_offset / self.patch_scale), round(cx,3), (y_offset / self.patch_scale), round(cy, 3)))

        return patch_a, patch_b, torch.tensor([x_offset / half_src_size, y_offset / half_src_size, cx, cy])

    def __len__(self):
        return len(self.image_filenames) * self.patches_per_image