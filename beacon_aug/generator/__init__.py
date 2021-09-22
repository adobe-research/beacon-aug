# Copyright 2021 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.
import importlib
import os
import sys
import glob
import yaml
from . import custom
from . import gan_based

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_LIBRARIES = ["albumentations", "imgaug", "torchvision", "keras",
                     "custom", "gan_based"]

if importlib.util.find_spec('augly') is not None:
    DEFAULT_LIBRARIES.append("augly")

if importlib.util.find_spec('kornia') is not None:
    DEFAULT_LIBRARIES.append("kornia")

if importlib.util.find_spec('mmcv') is not None and importlib.util.find_spec('beacon_aug.external.mmcv.mmseg.transforms'):
    DEFAULT_LIBRARIES.append("mmcv")

if importlib.util.find_spec('wand') is not None and importlib.util.find_spec('beacon_aug.external.imagenet_c.imagenet_c'):
    DEFAULT_LIBRARIES.append("imagenet_c")


def load_config(op_name, standard_fnames="./standard/*.yaml"):
    '''
    Extact and load the config for the current operator
    @input:
        standard_fnames: .json or .yaml file
    '''
    config = None
    config_custom = None
    # load config from custom python file
    for custom_op_name in custom.__all__:
        # get local fname withoutextension
        # custom_submodule = os.path.splitext(os.path.basename(py_fname))[0]
        if op_name == custom_op_name:
            module = importlib.import_module("custom." + op_name)  # dynamic import module
            custom_op_function = getattr(module, op_name)
            # add "custom" library to config paras
            transform_type = "ImageOnlyBATransform" if custom_op_function().image_only else "DualBATransform"
            config_custom = {"transform": transform_type,
                             custom_op_function().library: {"function": custom_op_function}}
            config = config_custom

    config_gan_based = None
    # load config from gan_based python file
    for custom_op_name in gan_based.__all__:
        # get local fname withoutextension
        # custom_submodule = os.path.splitext(os.path.basename(py_fname))[0]
        if op_name == custom_op_name:
            module = importlib.import_module("gan_based." + op_name)  # dynamic import module
            custom_op_function = getattr(module, op_name)
            # add "custom" library to config paras
            transform_type = "ImageOnlyBATransform" if custom_op_function().image_only else "DualBATransform"
            config_gan_based = {"transform": transform_type,
                                custom_op_function().library: {"function": custom_op_function}}
            config = config_gan_based

    # load config from standard json file
    js_fnames = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       standard_fnames))
    for js_fname in js_fnames:
        js_f = open(js_fname)
        OP_para = yaml.safe_load(js_f)
        if OP_para is not None:
            if op_name in OP_para:
                config = OP_para[op_name]
                # if same class exist in standard, add custom library to standard
                if config_custom:
                    config["custom"] = config_custom["custom"]
                if config_gan_based:
                    config["gan_based"] = config_custom["gan_based"]

    return config


def avail_libraries(operator_paras):
    ''' Return the list of all supportive library in this operator '''
    r = [l for l in DEFAULT_LIBRARIES if l in operator_paras]                      # only available ones
    return r


def load_avail_libraries(op_name):
    ''' Return the list of all supportive library in this operator '''
    operator_paras = load_config(op_name)
    r = [l for l in DEFAULT_LIBRARIES if l in operator_paras]                      # only available ones
    return r
