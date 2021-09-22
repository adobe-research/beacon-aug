# Copyright 2021 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

'''# Test all operators '''

from matplotlib.pyplot import get
import beacon_aug.generator.custom as custom
from beacon_aug.generator.operator_generator import DEFAULT_LIBRARIES
import yaml
import glob
import os
import pandas as pd
import beacon_aug as BA
from beacon_aug import screenshot
from skimage import io

image = io.imread("../data/example.png")

#####################

SAVE_PATH = "../docs/_images/"
FLAG = True

D = {}   # runtime table dictionary

"""Standard"""
extra_paras = {"Affine": {"shear": 30},
               "RandomCrop": {"height": 24, "width": 24},
               "CenterCrop": {"height": 24, "width": 24},
               "RandomResizedCrop": {"height": 24, "width": 24},
               "Resize": {"height": 24, "width": 24},
               "RandomSizedCrop": {"min_max_height": (64, 64)},
               "CropAndPad": {"percent": 0.1},
               "Rotate": {"limit": (30, 30)},
               "Autocontrast": {"limit": (0.5, 0.5)},
               "Brightness": {"limit": (0.5, 0.5)},
               "Saturation": {"saturation": (0.5, 0.5)},
               "Contrast": {"limit": (0.5, 0.5)},
               "Posterize": {"num_bits": 2},
               "Solarize": {"threshold": 64},
               "Sharpen": {"alpha": (0.5, 0.5)},
               }


ym_fnames = glob.glob("../beacon_aug/generator/standard/library_conversion.yaml")
ym_f = open(ym_fnames[0])
OP_para = yaml.safe_load(ym_f)

# standard libs
for op_name in OP_para:     # each operator
    print("@"*10, op_name)
    if op_name in extra_paras:
        # parse default paras
        _, D[op_name] = screenshot.screenshot_library(getattr(BA, op_name), **extra_paras[op_name],
                                                      image_data=image.copy(), individual_fig=FLAG, save_fig_path=SAVE_PATH)
    else:
        # no need to parse
        _, D[op_name] = screenshot.screenshot_library(getattr(BA, op_name),
                                                      image_data=image.copy(),  individual_fig=FLAG, save_fig_path=SAVE_PATH)


"""Customize"""
custom_doc_path = "../beacon_aug/generator/custom/"
extra_paras = {"OverlayText": {"size": 50},
               "TextFlow": {"size": 50},
               }

for op_name in custom.__all__:
    print("@"*10, op_name)
    if op_name in extra_paras:
        # parse default paras
        _, D[op_name] = screenshot.screenshot_library(getattr(BA, op_name), **extra_paras[op_name],
                                                      image_data=image.copy(), individual_fig=FLAG, save_fig_path=SAVE_PATH)
    else:
        # no need to parse
        _, D[op_name] = screenshot.screenshot_library(getattr(BA, op_name),
                                                      image_data=image.copy(),  individual_fig=FLAG, save_fig_path=SAVE_PATH)


f = open('properties.yaml', 'w+')
yaml.dump(D, f, allow_unicode=True)
