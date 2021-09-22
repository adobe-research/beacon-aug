# Copyright 2021 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.


'''
`"AutoAugment: Learning Augmentation Strategies from Data" <https://arxiv.org/pdf/1805.09501.pdf>`_.
Goal: 
    automatically search for improved data augmentation policies. Designed a search space where 
    a policy consists of many subpolicies, one of which is randomly chosen for each image in each 
    mini-batch. 

The beacon_aug version is adapted from the torchvision:
    https://github.com/pytorch/vision/blob/master/torchvision/transforms/autoaugment.py
Changes to above:
1) change all operators from torchvision to Beacon_aug  (support all libraries)
2) result equivalent to calling A.Compose(...)

'''

import beacon_aug as BA
import albumentations as A
import torch
import math
from torch import Tensor
from typing import List, Tuple, Optional
import numpy as np
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode
import cv2


def _get_transforms(policy):
    '''
    A sub-policy: two operations, each operation being an image processing function (such as
                translation, rotation, or shearing) and the probabilities and magnitudes with 
                which the functions are applied   (https://arxiv.org/pdf/1805.09501.pdf)

    Default: ImageNet 
            (transfers well to other datasets, such as Oxford Flowers, Caltech-101, Oxford-IIT Pets,
            FGVC Aircraft,and Stanford Cars) 
    '''
    if policy.lower() == "imagenet":
        return [  # (op_name, p, magnitude_id)
            # magnitude_id: location of magnitude intervals(<=_BINS, see detail in def _get_magnitudes():
            (("Posterize", 0.4, 8), ("Rotate", 0.6, 9)),
            (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
            (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
            (("Posterize", 0.6, 7), ("Posterize", 0.6, 6)),
            (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
            (("Equalize", 0.4, None), ("Rotate", 0.8, 8)),
            (("Solarize", 0.6, 3), ("Equalize", 0.6, None)),
            (("Posterize", 0.8, 5), ("Equalize", 1.0, None)),
            (("Rotate", 0.2, 3), ("Solarize", 0.6, 8)),
            (("Equalize", 0.6, None), ("Posterize", 0.4, 6)),
            (("Rotate", 0.8, 8), ("Saturation", 0.4, 0)),
            (("Rotate", 0.4, 9), ("Equalize", 0.6, None)),
            (("Equalize", 0.0, None), ("Equalize", 0.8, None)),
            (("Invert", 0.6, None), ("Equalize", 1.0, None)),
            (("Saturation", 0.6, 4), ("Contrast", 1.0, 8)),
            (("Rotate", 0.8, 8), ("Saturation", 1.0, 2)),
            (("Saturation", 0.8, 8), ("Solarize", 0.8, 7)),
            (("Sharpness", 0.4, 7), ("Invert", 0.6, None)),
            (("ShearX", 0.6, 5), ("Equalize", 1.0, None)),
            (("Saturation", 0.4, 0), ("Equalize", 0.6, None)),
            (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
            (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
            (("Invert", 0.6, None), ("Equalize", 1.0, None)),
            (("Saturation", 0.6, 4), ("Contrast", 1.0, 8)),
            (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
        ]
    elif policy.lower() == "cifar10":
        return [
            (("Invert", 0.1, None), ("Contrast", 0.2, 6)),
            (("Rotate", 0.7, 2), ("TranslateX", 0.3, 9)),
            (("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)),
            (("ShearY", 0.5, 8), ("TranslateY", 0.7, 9)),
            (("AutoContrast", 0.5, None), ("Equalize", 0.9, None)),
            (("ShearY", 0.2, 7), ("Posterize", 0.3, 7)),
            (("Saturation", 0.4, 3), ("Brightness", 0.6, 7)),
            (("Sharpness", 0.3, 9), ("Brightness", 0.7, 9)),
            (("Equalize", 0.6, None), ("Equalize", 0.5, None)),
            (("Contrast", 0.6, 7), ("Sharpness", 0.6, 5)),
            (("Saturation", 0.7, 7), ("TranslateX", 0.5, 8)),
            (("Equalize", 0.3, None), ("AutoContrast", 0.4, None)),
            (("TranslateY", 0.4, 3), ("Sharpness", 0.2, 6)),
            (("Brightness", 0.9, 6), ("Saturation", 0.2, 8)),
            (("Solarize", 0.5, 2), ("Invert", 0.0, None)),
            (("Equalize", 0.2, None), ("AutoContrast", 0.6, None)),
            (("Equalize", 0.2, None), ("Equalize", 0.6, None)),
            (("Saturation", 0.9, 9), ("Equalize", 0.6, None)),
            (("AutoContrast", 0.8, None), ("Solarize", 0.2, 8)),
            (("Brightness", 0.1, 3), ("Saturation", 0.7, 0)),
            (("Solarize", 0.4, 5), ("AutoContrast", 0.9, None)),
            (("TranslateY", 0.9, 9), ("TranslateY", 0.7, 9)),
            (("AutoContrast", 0.9, None), ("Solarize", 0.8, 3)),
            (("Equalize", 0.8, None), ("Invert", 0.1, None)),
            (("TranslateY", 0.7, 9), ("AutoContrast", 0.9, None)),
        ]
    elif policy.lower() == "svhn":
        return [
            (("ShearX", 0.9, 4), ("Invert", 0.2, None)),
            (("ShearY", 0.9, 8), ("Invert", 0.7, None)),
            (("Equalize", 0.6, None), ("Solarize", 0.6, 6)),
            (("Invert", 0.9, None), ("Equalize", 0.6, None)),
            (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
            (("ShearX", 0.9, 4), ("AutoContrast", 0.8, None)),
            (("ShearY", 0.9, 8), ("Invert", 0.4, None)),
            (("ShearY", 0.9, 5), ("Solarize", 0.2, 6)),
            (("Invert", 0.9, None), ("AutoContrast", 0.8, None)),
            (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
            (("ShearX", 0.9, 4), ("Solarize", 0.3, 3)),
            (("ShearY", 0.8, 8), ("Invert", 0.7, None)),
            (("Equalize", 0.9, None), ("TranslateY", 0.6, 6)),
            (("Invert", 0.9, None), ("Equalize", 0.6, None)),
            (("Contrast", 0.3, 3), ("Rotate", 0.8, 4)),
            (("Invert", 0.8, None), ("TranslateY", 0.0, 2)),
            (("ShearY", 0.7, 6), ("Solarize", 0.4, 8)),
            (("Invert", 0.6, None), ("Rotate", 0.8, 4)),
            (("ShearY", 0.3, 7), ("TranslateX", 0.9, 3)),
            (("ShearX", 0.1, 6), ("Invert", 0.6, None)),
            (("Solarize", 0.7, 2), ("TranslateY", 0.6, 7)),
            (("ShearY", 0.8, 4), ("Invert", 0.8, None)),
            (("ShearX", 0.7, 9), ("TranslateY", 0.8, 3)),
            (("ShearY", 0.8, 5), ("AutoContrast", 0.7, None)),
            (("ShearX", 0.7, 2), ("Invert", 0.1, None)),
        ]


def _get_magnitudes():
    _BINS = 10  # number of intervals
    return {
        # name: (magnitudes, signed)
        "ShearX": (np.linspace(0.0, 0.3, _BINS), True),
        "ShearY": (np.linspace(0.0, 0.3, _BINS), True),
        "TranslateX": (np.linspace(0.0, 150.0 / 331.0, _BINS), True),
        "TranslateY": (np.linspace(0.0, 150.0 / 331.0, _BINS), True),
        "Rotate": (np.linspace(0.0, 30.0, _BINS), True),
        "Brightness": (np.linspace(0.0, 0.9, _BINS), True),
        "Saturation": (np.linspace(0.0, 0.9, _BINS), False),
        "Contrast": (np.linspace(0.0, 0.9, _BINS), True),
        "Sharpness": (np.linspace(0.0, 0.9, _BINS), True),
        "Posterize": (np.array([8, 8, 7, 7, 6, 6, 5, 5, 4, 4]), False),
        "Solarize": (np.linspace(256.0, 0.0, _BINS), False),
        "AutoContrast": (None, None),
        "Equalize": (None, None),
        "Invert": (None, None),
    }


class AutoAugment:
    """AutoAugment data augmentation method based on
    `"AutoAugment: Learning Augmentation Strategies from Data" <https://arxiv.org/pdf/1805.09501.pdf>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".
    Args:
        policy (AutoAugmentPolicy): Desired policy enum defined by
            :class:`torchvision.transforms.autoaugment.AutoAugmentPolicy`. Default is ``AutoAugmentPolicy.IMAGENET``.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        library (str):flag for library. Should be one of: 'albumentations','imgaug','torchvision','keras'.
                                Default: 'albumentations'.


    e.g.

    .. code-block::

        auto_pipeline = BA.AutoAugment(policy = "imagenet")
        image_auged = auto_pipeline(image=image)["image"]

    """

    def __init__(self, policy="imagenet", interpolation=cv2.INTER_NEAREST,
                 library="albumentations", *args, **kwargs):
        super(AutoAugment).__init__(*args, **kwargs)
        self.policy = policy
        self.interpolation = interpolation
        self.library = library

        self.transforms = _get_transforms(policy)
        if self.transforms is None:
            raise ValueError("The provided policy {} is not recognized.".format(policy))
        self._op_meta = _get_magnitudes()

    @staticmethod
    def get_params(transform_num: int) -> Tuple[int, Tensor, Tensor]:
        """Get parameters for autoaugment transformation
        Returns:
            params required by the autoaugment transformation
        """

        policy_id = np.random.random_integers(0, transform_num-1)  # 0 to transform_num)
        probs = np.random.rand(1, 2)[0]                             # [prob1,prob2]
        signs = np.random.randint(2, size=2)                        # [ 0,1] / [0,0]/...
        # print("Parameters: ", " policy_id=", policy_id, " probs=", probs,  "signs=", signs)
        return policy_id, probs, signs

    def _get_op_meta(self, name: str) -> Tuple[Optional[Tensor], Optional[bool]]:
        return self._op_meta[name]

    def __new__(cls, policy="imagenet", interpolation=cv2.INTER_NEAREST,
                library="albumentations", *args, **kwargs):
        obj = super(AutoAugment, cls).__new__(cls)
        obj.__init__(policy, interpolation,  library, *args, **kwargs)
        transform_id, probs, signs = obj.get_params(len(obj.transforms))
        compose_ls = []
        for i, (op_name, p, magnitude_id) in enumerate(obj.transforms[transform_id]):
            if probs[i] <= p:
                print("transform id=", i, op_name, p, magnitude_id)
                magnitudes, signed = obj._get_op_meta(op_name)
                magnitude = float(magnitudes[magnitude_id].item()) \
                    if magnitudes is not None and magnitude_id is not None else 0.0
                if signed is not None and signed and signs[i] == 0:
                    magnitude *= -1.0

                if op_name == "ShearX":
                    compose_ls.append(BA.Affine(p=1, rotate=0.0, translate_px=[0, 0], scale=1.0, shear=[math.degrees(magnitude), 0.0],
                                                interpolation=obj.interpolation,  library=obj.library))
                elif op_name == "ShearY":
                    compose_ls.append(BA.Affine(p=1, rotate=0.0, translate_px=[0, 0], scale=1.0,  shear=[0.0, math.degrees(magnitude)],
                                                interpolation=obj.interpolation,  library=obj.library))
                elif op_name == "TranslateX":
                    compose_ls.append(BA.Affine(p=1, rotate=0.0, translate_px=[int(F._get_image_size(img)[0] * magnitude), 0],
                                                scale=1.0, shear=[0.0, 0.0],  interpolation=obj.interpolation,  library=obj.library))
                elif op_name == "TranslateY":
                    compose_ls.append(BA.Affine(p=1, rotate=0.0, translate_px=[0, int(F._get_image_size(img)[1] * magnitude)],
                                                scale=1.0, shear=[0.0, 0.0], interpolation=obj.interpolation,  library=obj.library))
                elif op_name == "Rotate":
                    compose_ls.append(BA.Rotate(p=1, limit=magnitude,
                                                interpolation=obj.interpolation,  library=obj.library))
                elif op_name == "Brightness":
                    compose_ls.append(BA.Brightness(p=1, limit=magnitude, library=obj.library))
                elif op_name == "Saturation":
                    compose_ls.append(BA.Saturation(p=1, saturation=magnitude, library=obj.library))
                elif op_name == "Contrast":
                    compose_ls.append(BA.Contrast(p=1, limit=magnitude, library=obj.library))
                elif op_name == "Sharpness":
                    compose_ls.append(BA.Sharpen(p=1, alpha=magnitude, library=obj.library))
                elif op_name == "Posterize":
                    compose_ls.append(BA.Posterize(p=1, num_bits=int(magnitude),
                                                   library=obj.library))
                elif op_name == "Solarize":
                    compose_ls.append(BA.Solarize(p=1, threshold=magnitude, library=obj.library))
                elif op_name == "AutoContrast":  # only support  imgaug, torchvision
                    compose_ls.append(BA.Autocontrast(p=1))
                elif op_name == "Equalize":
                    compose_ls.append(BA.Equalize(p=1, library=obj.library))
                elif op_name == "Invert":
                    compose_ls.append(BA.Invert(p=1, library=obj.library))
                else:
                    raise ValueError("The provided operator {} is not recognized.".format(op_name))
        return A.Compose(compose_ls)

    def __repr__(self):
        return self.__class__.__name__ + '(policy={})'.format(self.policy)
