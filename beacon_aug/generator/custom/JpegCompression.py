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
import torch
from beacon_aug.external.DiffJPEG.DiffJPEG import DiffJPEG
# import torchvision.transforms as torch
import torch


class JpegCompression:
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

    - text overlap in random place of image   

        .. code-block::

            import beacon_aug as BA
            aug = BA.OverlayText(p=1, text="Text", library="custom")
            image_auged = aug(image=image)["image"]

    - A list of text overlap in limit place of image   

        .. code-block::

            import beacon_aug as BA
            aug = BA.OverlayText(p=1, text=[ "Text1","Text2" ,  x= [10,100], y= [100,140], 
                          library="custom")
            image_auged = aug(image=image)["image"]              

    """

    def __init__(self, differentiable=True, quality=80):
        self.library = "custom"
        self.image_only = True  # True only applies  to image; False applies to both image and bbox

        self.differentiable = differentiable
        self.quality = quality

    def __call__(self, img):
        # load image
        if type(img) == torch.Tensor:
            x = img.copy()
            img_height = x.shape[2]
            img_width = x.shape[3]
        else:
            if type(img) == np.ndarray:
                img_width = img.shape[0]
                img_height = img.shape[1]
            # convert uint8 array to float tensor (batch x 3x height x width)
            x = torch.from_numpy(np.array([img]).transpose(0, 3, 1, 2))/255

        layer = DiffJPEG(height=img_height, width=img_width,
                         differentiable=self.differentiable, quality=self.quality)
        y = layer.forward(x)

        if type(img) == torch.Tensor:
            return y
        else:
            return (y[0].permute(1, 2, 0).detach().numpy()*255).astype(np.uint8)
