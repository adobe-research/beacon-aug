# Copyright 2021 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.


# import beacon_aug as BA
from .. import __init__ as BA
import cv2
import numpy as np


def Benign(in_place=True, out_place=True, compress=True,
           severity=[1, 5], p=1, in_place_n=1, *args, **kwargs):
    """ Apply run random number of operators according to hint

    Parameters
    ----------
    if in_place = True:
        in_place_n (int,[int,int]): number (or range) of operators to select in "imagenet_c" (not change pixel location)
                                    default: 1
        severity(int,[int,int]): severity (or range) of imagenet_c transformations (change pixel location)
                                default: [1,5]

    This set of distortions is a slightly simplified (and harder) set of transformations taken from this paper:
    [1] Black, Alexander, et al. "Deep Image Comparator: Learning To Visualize Editorial
    Change." Proceedings of the IEEE/CVF Conference on CVPR Workshops, 2021.

    The original paper additionally has two layers of randomization:
    - Sample a random transformation from the `out_place` transformations.
    - Sample a random set of transformations from `[in_place]`, `[out_place]`, `[in_place, out_place]`
      transformations (compression is always applied).
    Moreover, the parameters of some transformations have been slightly adjusted.

    e.g.

    .. code-block::

        auto_pipeline = BA.Benign()
        image_auged = auto_pipeline(image=image)["image"]

    """

    # Corruption and Perturbation Robustness
    in_place_ops = [BA.Resize(height=256, width=256, library="albumentations"),
                    BA.Collections(tag="imagenet-c", p=p, severity=severity, n=in_place_n)]

    # transformations that change pixel coordinates
    out_place_ops = [
        # rotation (max 15 degree),
        BA.Rotate(p=p, limit=(-15, 15), border_mode=cv2.BORDER_CONSTANT, library="albumentations"),
        # random crop (90% area), padding (max 10% each side),
        BA.CropAndPad(p=p, percent=(-0.1, 0.1), library="albumentations"),
        BA.HorizontalFlip(p=0.5),
    ]
    compress_ops = [
        # JPEG compression (40%-90%)
        BA.JpegCompression(p=p, quality_lower=40, quality_upper=90),
    ]

    op_list = (in_place_ops if in_place else []) + \
              (out_place_ops if out_place else []) + \
              (compress_ops if compress else [])

    return BA.Compose(op_list)
