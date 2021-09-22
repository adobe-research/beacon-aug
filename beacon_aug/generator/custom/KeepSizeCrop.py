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
import albumentations as A
import cv2


class KeepSizeCrop:
    """crop a random part of the input and rescale it to some size.

    Args:
        scale ((float, float)): range of size of the origin size cropped
        ratio ((float, float)): range of aspect ratio of the origin aspect ratio cropped
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.

    e.g.

    -  Test the function:    

        .. code-block::

            from beacon_aug.generator.custom.KeepSizeCrop import KeepSizeCrop
            op = KeepSizeCrop(scale(0.08, 1.0))
            img_transformed = op(img )

    - Test the augmentation wrapper:    

        .. code-block::

            import beacon_aug as BA
            aug = BA.KeepSizeCrop(p=1, scale(0.08, 1.0), library="custom")
            image_auged = aug(image=image)["image"]              


    """

    def __init__(self, scale=(0.08, 1.0),  ratio=(0.75, 1.3333333333333333),
                 interpolation=cv2.INTER_LINEAR, ):
        self.image_only = True  # True only applies  to image; False applies to both image and bbox
        self.library = "custom"
        self._scale = scale
        self._ratio = ratio
        self._interpolation = interpolation

    def __call__(self, img):
        # load image
        aug = A.RandomResizedCrop(height=img.shape[0], width=img.shape[1],
                                  scale=self._scale,  ratio=self._ratio,  interpolation=self._interpolation)
        img = aug(image=img)["image"]
        return img
