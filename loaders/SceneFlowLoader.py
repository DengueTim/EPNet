import os
import torch
import torch.utils.data
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from loaders.SceneFlowIO import readFlow


class SceneFlowLoader(torch.utils.data.Dataset):
    def __init__(self, scene_flow_root, scene_flow_filenames, log=None):
        self.scene_flow_root = scene_flow_root
        self.filenames = scene_flow_filenames
        self.log = log

    def __getitem__(self, index):
        image_filename_a, image_filename_b, flow_filename_a_to_b, flow_filename_b_to_a = self.filenames[index]

        image_filename_a = os.path.join(self.scene_flow_root, image_filename_a)
        image_filename_b = os.path.join(self.scene_flow_root, image_filename_b)
        flow_filename_a_to_b = os.path.join(self.scene_flow_root, flow_filename_a_to_b)
        flow_filename_b_to_a = os.path.join(self.scene_flow_root, flow_filename_b_to_a)

        with torch.no_grad():
            pil_image_a = Image.open(image_filename_a).convert('L') # .convert('RGB')
            image_a = transforms.ToTensor()(pil_image_a)
            pil_image_a.close()

            pil_image_b = Image.open(image_filename_b).convert('L') # .convert('RGB')
            image_b = transforms.ToTensor()(pil_image_b)
            pil_image_b.close()

            flow_a2b = readFlow(flow_filename_a_to_b).copy()
            flow_a2b = np.transpose(flow_a2b,(2,0,1)) # pytorch like, channels first.
            flow_a2b = torch.from_numpy(flow_a2b)
            flow_b2a = readFlow(flow_filename_b_to_a).copy()
            flow_b2a = np.transpose(flow_b2a, (2, 0, 1))
            flow_b2a = torch.from_numpy(flow_b2a)

        return image_a, image_b, flow_a2b, flow_b2a

    def __len__(self):
        return len(self.filenames)