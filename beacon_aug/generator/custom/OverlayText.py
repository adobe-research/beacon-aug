# Copyright 2021 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

'''
Adopt from albumentations
https://github.com/albumentations-team/albumentations/blob/95a048d342cbe748e5acf15cb1a28611a6303885/albumentations/augmentations/crops/transforms.py
'''
import numpy as np
import random
from imgaug.imgaug import draw_text


class OverlayText:
    """crop a random part of the input and rescale it to some size.

    Args:
        text (str): overlay text
        x (int or list): value or range of the x_coordinate of text, 
                Default:  None. (random select in range of image)
        y (int or list): value or range of the y_coordinate of text, 
                Default:  None. (random select in range of image)      
        size (int or list): value or range of the size of text, 
                Default:  25    
        color ( RGB value): value of text color
                Default:  (0,255,0)   


    e.g.

    -  text overlap in random place of image:    

        .. code-block::

            import beacon_aug as BA
            aug = BA.OverlayText(p=1, text="Text", library="custom")
            image_auged = aug(image=image)["image"]

    - a list of text overlap in limit place of image
        .. code-block::

            aug = BA.OverlayText(p=1, text=[ "Text1","Text2" ,  x= [10,100], y= [100,140], 
                                library="custom")
            image_auged = aug(image=image)["image"]

    """

    def __init__(self, text="Text", x=None, y=None, size=30, color=(255, 255, 0)):
        self.library = "custom"
        self.image_only = True  # True only applies  to image; False applies to both image and bbox

        self.x = x
        self.y = y
        self.size = size
        self.color = color  # True only applies  to image; False applies to both image and bbox
        self.text = text

    def __call__(self, img):
        # load image
        size = random.randint(self.size[0], self.size[1]) if type(
            self.size) != int and float else self.size

        # default x, y ( range to whole image shape)
        if self.x is None:
            self.x = [0, min(img.shape[0] - size, 0)]
        if self.y is None:
            self.y = [0, min(img.shape[1] - size, 0)]

        # random x, y location
        x = random.randint(self.x[0], self.x[1]) if type(self.x) != int and float else self.x
        y = random.randint(self.y[0], self.y[1]) if type(self.x) != int and float else self.x

        text = random.choice(self.text) if type(self.text) != str else self.text

        img_auged = draw_text(img, y=y, x=x, size=size, text=text)
        return img_auged
