# Copyright 2021 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.


import cv2
import matplotlib.pyplot as plt
import matplotlib
from skimage import data
import os
import random
import time
# import beacon_aug as BA
from . import operators as BA

import torch
import numpy as np
from typing import Callable, Iterable
import inspect
import cv2
import torchvision.transforms.functional as torch_f

kw_size_list = ["size", "crop_size", "kernel_size"]
kw_wh_list = ["weight", "heigh"]


def set_seed(seed=None, reset= False):
    '''
    Set Random seed to control the replication of transformations
    
    .. code-block::

        import beacon_aug as BA
        BA.properties.set_seed(reset=True)
  
    '''
    if reset:
        seed =  random.randint(0, 1048) 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print (seed)


def isOpDifferentiable(op=None,
                       size: Iterable[int] = (2, 3, 8, 8),  # rgb image (8x8) batch size 2,
                       device: str = "cpu"):
    """

    Check if `op` is differentiable. `source <https://git.corp.adobe.com/gist/holdgeof/22a496b76758cd9bfb690c8e88ac1c92>`_

    Contributed by Yannick Hold-Geoffroy 

    e.g.

    .. code-block::

        import beacon_aug as BA
        BA.properties.isOpDifferentiable(op=BA.RandomCrop(library="torch"))

    """
    # set default
    if op == None:
        op = BA.RandomCrop(library="torch")

    x = torch.rand(size, requires_grad=True, device=device)
    try:
        # make sure the size is smaller than the input image
        for kw in kw_size_list:
            if kw in dir(op.processor):
                setattr(op.processor, kw, (3, 3))
        for kw in kw_wh_list:
            if kw in dir(op.processor):
                setattr(op.processor, kw, 3)
        # set "as_layter = True" to support tensor input in Beacon_aug
        y = op.apply(x, as_layer=True)
        loss = y.abs().sum()
        loss.backward()
    except Exception as e:
        print(e)
        return False

    is_diff = (
        x.grad.shape == tuple(size) and
        torch.any(y != 0.) and
        torch.all(torch.isfinite(x.grad)).item()
    )

    return is_diff


def isAntiAliasing(BA_operator, library, interpolation, **kwargs):
    """
    Check if `op` is anti aliasing. `source Paper <https://github.com/GaParmar/clean-fid>`_
    Whether or not the operator is anti-alising  depends on the PSNR between original and rescaled images

    Contributed by `Richard Zhang`, 

    e.g.

    .. code-block::

        import beacon_aug as BA
        BA.properties.isAntiAliasing(BA.Resize, library= "torchvision",interpolation = "linear" )
    """

    def psnr_uint8(img0, img1):
        return -10*np.log10(np.mean((img0/255. - img1/255.)**2))

    is_anti_aliasing = False

    img_np = np.zeros((128, 128, 3), dtype='uint8')+255
    img_np = cv2.circle(img_np, (128//2, 128//2), 50, (0, 0, 0), 1)

    if "resize" in BA_operator.__name__.lower():
        # interpolation_all = ["nearest", "linear", "area", "cubic", "Lanczos", "hamming"]
        interpolation_dict = {"albumentations":
                              {"nearest": cv2.INTER_NEAREST,
                               "linear": cv2.INTER_LINEAR,
                               "area": cv2.INTER_AREA,
                               "cubic": cv2.INTER_CUBIC,
                               "Lanczos": cv2.INTER_LANCZOS4,
                               },
                              "imgaug":
                              {"nearest": "nearest",
                               "linear": "linear",
                               "cubic": "cubic",
                               "area": "area"
                               },
                              "torchvision":
                              {"nearest": torch_f.InterpolationMode.NEAREST,
                               "linear": torch_f.InterpolationMode.BILINEAR,
                               "box": torch_f.InterpolationMode.BOX,
                               "cubic": torch_f.InterpolationMode.BICUBIC,
                               "Lanczos": torch_f.InterpolationMode.LANCZOS,
                               "hamming": torch_f.InterpolationMode.HAMMING,
                               }}
        interp = interpolation_dict[library][interpolation] if library in interpolation_dict else interpolation
        # print(interp)

        op = BA.Resize(p=1, library=library, interpolation=interp,
                       height=16, width=16, **kwargs)
        op_up = BA.Resize(p=1, library=library, interpolation=interp, height=128, width=128)

        img_resized = op(image=img_np)["image"].copy()     # downscale to 16x16
        img_up = op_up(image=img_resized)["image"].copy()  # upscale to 128*128

        psnr = psnr_uint8(img_np, img_up)

        if psnr > 18:
            is_anti_aliasing = True

    return is_anti_aliasing


def library_attributes(BA_operator,  image_data=None,
                       **kwargs):
    '''    Visualize the augmentation result comparision to all available libraries

    e.g.

    .. code-block::

        import beacon_aug as BA
        attributes_result = library_attributes(BA.Add)

    '''
    if image_data == None:
        image_data = np.random.randint(255, size=(8, 8, 3))
    avail_libraries = BA_operator(**kwargs).avail_libraries

    attributes_result = {"runtime": {}, "differentiable": {}}
    for i, library in enumerate(avail_libraries):
        t_before = time.time()
        op = BA_operator(always_apply=False, p=1, library=library,
                         **kwargs)
        image_auged = op(image=image_data)["image"]
        t_after = time.time()
        runtime = t_after - t_before
        attributes_result["runtime"][library] = runtime
        attributes_result["differentiable"][library] = isOpDifferentiable(op)

    return attributes_result
