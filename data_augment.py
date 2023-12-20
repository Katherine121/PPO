import random
import imgaug.augmenters as iaa
import numpy as np
from PIL import Image
from torch import nn
# from cor import *


class ImageAugment(nn.Module):
    def __init__(self, style_idx, shift_idx):
        """
        ImageAugment
        :param style_idx: the index of style augment
        :param shift_idx: the index of position augment
        """
        super(ImageAugment, self).__init__()
        self.style_idx = style_idx
        self.shift_idx = shift_idx

        if style_idx == "cutout":
            self.style = iaa.Cutout(nb_iterations=(1, 3), size=(0.1, 0.3))
        elif style_idx == "rain":
            self.style = iaa.Rain()
        elif style_idx == "snow":
            self.style = iaa.Snowflakes()
        elif style_idx == "fog":
            self.style = iaa.Fog()
        elif style_idx == "bright":
            self.style = iaa.imgcorruptlike.Brightness()
        else:
            self.style = None

    def forward(self, image):
        """
        forward pass of ImageAugment
        :param image: the provided input image
        :return: the image after augmented
        """
        if self.style is not None:
            image = self.style(image=image)
        return image

    def forward_shift(self, cur_noise):
        """
        forward pass of position augment
        :return: the position shift after augmented
        """
        if self.shift_idx == "random":
            random_lat_shift = random.uniform(-cur_noise * 9e-6, cur_noise * 9e-6)
            random_lon_shift = random.uniform(-cur_noise * 9.8e-6, cur_noise * 9.8e-6)
            random_hei_shift = random.uniform(-cur_noise, cur_noise)
        else:
            random_lat_shift = 0
            random_lon_shift = 0
            random_hei_shift = 0
        return list([random_lat_shift, random_lon_shift, random_hei_shift])


if __name__ == '__main__':
    image_augment = ImageAugment(style_idx="bright", shift_idx="ori")
    pic = Image.open("../1.png")

    pic = np.array(pic)
    pic = image_augment(pic)
    pic = Image.fromarray(pic)
    pic.save("../2.png")
