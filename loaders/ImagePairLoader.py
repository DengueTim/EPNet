import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms as transforms


class ImagePairLoader(torch.utils.data.Dataset):
    def __init__(self, image_filenames, log=None):
        self.image_filenames = image_filenames
        self.log = log

    def __getitem__(self, index):
        image_filename_a, image_filename_b = self.image_filenames[index]

        with torch.no_grad():
            pil_image_a = Image.open(image_filename_a) # .convert('RGB')
            image_a = transforms.ToTensor()(pil_image_a)
            pil_image_a.close()

            pil_image_b = Image.open(image_filename_b) # .convert('RGB')
            image_b = transforms.ToTensor()(pil_image_b)
            pil_image_b.close()

        return image_a, image_b

    def __len__(self):
        return len(self.image_filenames)