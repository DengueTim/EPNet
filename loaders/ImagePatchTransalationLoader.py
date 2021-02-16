import torch
import torch.utils.data
from PIL import Image
import random
import torchvision.transforms as transforms

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
            image = Image.open(image_filename).convert('RGB')
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

        patch_a = patch_a.resize((self.patch_size, self.patch_size), Image.BILINEAR)
        patch_b = patch_b.resize((self.patch_size, self.patch_size), Image.BILINEAR)

        patch_a = transforms.ToTensor()(patch_a)
        patch_b = transforms.ToTensor()(patch_b)

        # WIP: Some kind of measure of how smooth the patch is around its center
        # Like higher gradients(more texture) should lead to a more confidant prediction.
        xy1 = self.patch_size // 2 - 4
        xy2 = self.patch_size // 2 + 5
        center_patch = patch_a[:, xy1:xy2, xy1:xy2]
        sum_of_err_x = 0;
        sum_of_err_y = 0;
        for i in (1,-1,2,-2,3,-3):
            shifted_patch = patch_a[:, xy1:xy2, xy1 + i:xy2 + i]
            diff = center_patch - shifted_patch
            sum_of_err_x += diff.abs().sum().item() / (9 * 9 * 3)
            shifted_patch = patch_a[:, xy1 + 1:xy2 + 1, xy1:xy2]
            diff = center_patch - shifted_patch
            sum_of_err_y += diff.abs().sum().item() / (9 * 9 * 3)

        cx = 1 - 1 / (1 + sum_of_err_x)
        cy = 1 - 1 / (1 + sum_of_err_y)

        #if self.log:
        #   self.log.info('{}:{}\t{}:{}'.format((x_offset / self.patch_scale), round(cx,3), (y_offset / self.patch_scale), round(cy, 3)))

        return patch_a, patch_b, torch.tensor([x_offset / half_src_size, y_offset / half_src_size, cx, cy])

    def __len__(self):
        return len(self.image_filenames) * self.patches_per_image