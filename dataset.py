import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import cv2

class RandomCrop(transforms.RandomResizedCrop):
    def __call__(self, imgs):
        i, j, h, w = self.get_params(imgs[0], self.scale, self.ratio)
        for imgCount in range(len(imgs)):
            imgs[imgCount] = transforms.functional.resized_crop(
                imgs[imgCount], i, j, h, w, self.size, self.interpolation
            )
        return imgs


def remove_hair(pil_img):
    """
    Simulated hair removal using OpenCV.
    Input: PIL image
    Output: PIL image with reduced hair
    """
    image = np.array(pil_img)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


    kernel = cv2.getStructuringElement(1, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)


    return Image.fromarray(inpainted)

class ISICLoader(Dataset):
    def __init__(self, images, masks, augment=True, typeData="train", size=(256, 256)):
        self.augment = augment if typeData == "train" else False
        self.images = images
        self.masks = masks
        self.size = size
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(self.size, interpolation=Image.NEAREST)


    def __len__(self):
        return len(self.images)


    def rotate(self, image, mask, degrees=(-15, 15), p=0.5):
        if torch.rand(1) < p:
            degree = np.random.uniform(*degrees)
            image = image.rotate(degree, Image.NEAREST)
            mask = mask.rotate(degree, Image.NEAREST)
        return image, mask


    def horizontal_flip(self, image, mask, p=0.5):
        if torch.rand(1) < p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image, mask


    def vertical_flip(self, image, mask, p=0.5):
        if torch.rand(1) < p:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return image, mask


    def random_resized_crop(self, image, mask, p=0.1):
        if torch.rand(1) < p:
            crop = RandomCrop(self.size, scale=(0.8, 0.95))
            image, mask = crop([image, mask])
        else:
            image = self.resize(image)
            mask = self.resize(mask)
        return image, mask


    def augment_image_mask(self, image, mask):
        image, mask = self.random_resized_crop(image, mask)
        image, mask = self.rotate(image, mask)
        image, mask = self.horizontal_flip(image, mask)
        image, mask = self.vertical_flip(image, mask)
        return image, mask


    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        mask = Image.fromarray(self.masks[idx])
        image = remove_hair(image)
        if self.augment:
            image, mask = self.augment_image_mask(image, mask)
        else:
            image = self.resize(image)
            mask = self.resize(mask)


        image = self.to_tensor(image)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64)).unsqueeze(0)
        return image, mask




