import os
import random
import torch
import torch.utils.data
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from loaders.SceneFlowIO import readFlow


class SceneFlowLoader(torch.utils.data.Dataset):
    def __init__(self, scene_flow_root, scene_flow_filenames, size_multiple_of=64, log=None):
        self.scene_flow_root = scene_flow_root
        self.filenames = scene_flow_filenames
        self.size_multiple_of = size_multiple_of
        self.log = log

    def __getitem__(self, index):
        image_filename_a, image_filename_b, flow_filename_a_to_b, flow_filename_b_to_a = self.filenames[index]

        image_filename_a = os.path.join(self.scene_flow_root, image_filename_a)
        image_filename_b = os.path.join(self.scene_flow_root, image_filename_b)
        flow_filename_a_to_b = os.path.join(self.scene_flow_root, flow_filename_a_to_b)
        flow_filename_b_to_a = os.path.join(self.scene_flow_root, flow_filename_b_to_a)

        with torch.no_grad():
            pil_image_a = Image.open(image_filename_a).convert('L') # .convert('RGB')
            pil_image_a = crop_img_size_to_multiple_of(pil_image_a, self.size_multiple_of)
            image_a = transforms.ToTensor()(pil_image_a)
            pil_image_a.close()

            pil_image_b = Image.open(image_filename_b).convert('L') # .convert('RGB')
            pil_image_b = crop_img_size_to_multiple_of(pil_image_b, self.size_multiple_of)
            image_b = transforms.ToTensor()(pil_image_b)
            pil_image_b.close()

            flow_a2b = readFlow(flow_filename_a_to_b)
            flow_a2b = np.transpose(flow_a2b, (2, 0, 1)) # pytorch like, channels first.
            flow_a2b = crop_flow_size_to_multiple_of(flow_a2b, self.size_multiple_of)
            flow_a2b = torch.from_numpy(flow_a2b.copy())

            flow_b2a = readFlow(flow_filename_b_to_a)
            flow_b2a = np.transpose(flow_b2a, (2, 0, 1))
            flow_b2a = crop_flow_size_to_multiple_of(flow_b2a, self.size_multiple_of)
            flow_b2a = torch.from_numpy(flow_b2a.copy())

            if random.getrandbits(1):
                image_a = torch.flip(image_a, [1])
                image_b = torch.flip(image_b, [1])
                flow_a2b = torch.flip(flow_a2b, [1])
                flow_a2b[1] = -flow_a2b[1]
                flow_b2a = torch.flip(flow_b2a, [1])
                flow_b2a[1] = -flow_b2a[1]

            if random.getrandbits(1):
                image_a = torch.flip(image_a, [2])
                image_b = torch.flip(image_b, [2])
                flow_a2b = torch.flip(flow_a2b, [2])
                flow_a2b[0] = -flow_a2b[0]
                flow_b2a = torch.flip(flow_b2a, [2])
                flow_b2a[0] = -flow_b2a[0]

        return image_a, image_b, flow_a2b, flow_b2a

    def __len__(self):
        return len(self.filenames)


def crop_img_size_to_multiple_of(img, size_multiple_of):
    # Crop H and W dims to be divisible by 64
    H, W = np.shape(img)

    h = (H // size_multiple_of) * size_multiple_of
    w = (W // size_multiple_of) * size_multiple_of
    oh = (H - h) // 2
    ow = (W - w) // 2

    return img.crop((ow, oh, ow + w, oh + h))


def crop_flow_size_to_multiple_of(flow, size_multiple_of):
    # Crop H and W dims to be divisible by 64
    C, H, W = np.shape(flow)

    h = (H // size_multiple_of) * size_multiple_of
    w = (W // size_multiple_of) * size_multiple_of
    oh = (H - h) // 2
    ow = (W - w) // 2

    return flow[:, oh:oh + h, ow:ow + w]
