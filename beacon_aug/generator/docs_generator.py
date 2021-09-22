# Copyright 2021 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.


"""
automatically extract the docs from original backend libraries
"""
# beacon_aug
from . import DEFAULT_LIBRARIES, load_avail_libraries
from . import custom, gan_based
import beacon_aug as BA

# libraries
from torch import from_numpy
from torch import Tensor
import keras_preprocessing.image as keras
from albumentations.augmentations.functional import MAX_VALUES_BY_DTYPE
from albumentations.core.transforms_interface import BasicTransform, DualTransform, ImageOnlyTransform, to_tuple
from albumentations.augmentations.keypoints_utils import convert_keypoints_from_albumentations, convert_keypoints_to_albumentations
from albumentations.augmentations.bbox_utils import convert_bboxes_from_albumentations, convert_bboxes_to_albumentations
import albumentations as A

# imgaug
try:
    from imgaug import augmenters as iaa
except ImportError:
    import imgaug.imgaug.augmenters as iaa
from imgaug.augmenters import geometric as iaa_geometic

from numpy.core.fromnumeric import shape
import imgaug as ia
from re import L
import imgaug
import pdb
import json
import yaml
from PIL import Image
import cv2
import torchvision.transforms as torch
import torchvision.transforms.functional as torch_f

if "augly" in DEFAULT_LIBRARIES:
    import augly.image as augly
if "kornia" in DEFAULT_LIBRARIES:
    import kornia as K
if "mmcv" in DEFAULT_LIBRARIES:
    from beacon_aug.external.mmcv.mmseg import transforms as mmcv
if "kornia" in DEFAULT_LIBRARIES:
    import kornia as K
if "imagenet_c" in DEFAULT_LIBRARIES:
    from beacon_aug.external.imagenet_c.imagenet_c import imagenet_c  # .corruptions import transforms


# others
from inspect import getfullargspec
import os
import sys
import numpy as np
import glob
import argparse
import importlib


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

_AFFINE_MODE_CV2_TO_SKIMAGE = {v: k for k,
                               v in iaa_geometic._AFFINE_MODE_SKIMAGE_TO_CV2.items()}  # reverse map
_CODE_TO_CV2_KERAS_MODE = {0: 'nearest', 1: 'wrap', 2: 'reflect', 3: 'mirror', 4: 'constant'}

_ARGS_NAMES_EXCEPT = ["self", "seed", "name", "random_state", "deterministic", "p", "deterministic",
                      "from_colorspace", "nb_iterations"]


def replace_alias(function, op_dic, lib_name):
    """Find the source function from alias"""

    if (' alias' in function.__doc__.lower() or 'alias for' in function.__doc__.lower() or "deprecated in favor of" in function.__doc__):

        # (imgaug.PIL)
        if 'Alias for :class:'.lower() in function.__doc__.lower():
            original_fct_name = op_dic[lib_name]["function"]
            new_fct_fullname = function.__doc__.split("See :class:`~")[-1].split("`")[0]
            function = eval(new_fct_fullname)
            return replace_alias(function, op_dic, lib_name)

        elif 'an alias for'.lower() in function.__doc__.lower():
            new_fct_fullname = function.__doc__.split(
                ":class:`~")[-1].split("`")[0]
            function = eval(new_fct_fullname)
            return replace_alias(function, op_dic, lib_name)

        # (torchvision)
        elif "Note: This transform is deprecated in favor of".lower() in function.__doc__.lower():
            original_fct_name = op_dic[lib_name]["function"]
            new_fct_name = function.__doc__.split(
                "deprecated in favor of ")[-1].split(".")[0]
            new_fct_fullname = original_fct_name.replace(
                original_fct_name.split('.')[-1], new_fct_name)
            function = eval(new_fct_fullname)
            return replace_alias(function, op_dic, lib_name)

    return function


def generate_doc(op_name, source_js_folders=os.path.join(os.path.dirname(os.path.abspath(__file__)), "standard/*.yaml"),
                 verbose=False):

    Final_doc = ""

    js_fnames = glob.glob(source_js_folders)
    if verbose:
        print("js_fnames=", js_fnames)
        print("op_name=", op_name)

    lib_arg_doc_towrite = ""
    avail_libraries = load_avail_libraries(op_name)
    avail_lib_text = ""
    for key in avail_libraries:
        avail_lib_text += (" ``" + str(key) + "``,")

    avail_libraries_except_alb = avail_libraries.copy()

    if verbose:
        print("avail_libraries=", avail_libraries)

    # load op_dic for standard operators
    op_dic = None
    for js_fname in js_fnames:
        js_f = open(js_fname)
        OP_para = yaml.safe_load(js_f)
        if op_name in OP_para.keys():     # each operators
            # if op_name in ["HorizontalFlip"]:     # each operators (for debug)
            # doc for library specific arguments
            op_dic = OP_para[op_name].copy()

    if "albumentations" in avail_libraries:
        # extract the original doc from albumentations definition
        A_doc = eval(str(op_dic["albumentations"]["function"])+".__doc__")

        lib_doc_towrite = f'''library (str):  flag for library. Should be one of: {avail_lib_text[:-1]}.
                        Default: ``{avail_libraries[0]}``.\n'''
        A_doc = A_doc.replace("Args:\n", "Args:\n        " +
                              lib_doc_towrite)             # insert "library" argument after "Args"
        # remove it for other libs
        avail_libraries_except_alb.remove("albumentations")
    else:
        A_doc = ""

    for lib_name in avail_libraries_except_alb:                # other lib
        if verbose == True:
            print(op_name, lib_name)

        #### Extract original doc `lib_doc`from  original library ####
        lib_doc = None
        if lib_name != "custom" and lib_name != "gan_based":
            try:
                function = eval(op_dic[lib_name]["function"])
            except TypeError:
                raise TypeError("op_dic not found:", lib_name, op_name)
        else:
            module = importlib.import_module(lib_name + "." + op_name)  # dynamic import module
            function = getattr(module, op_name)

        # get the source original doc
        if function.__doc__ is not None and op_dic is not None:
            function = replace_alias(function, op_dic, lib_name)
            if verbose == True:
                print("\t", function)
        # Compost library individual docs
        lib_doc = function.__doc__
        if verbose == True:
            print("\t", lib_doc)

        ##### edit the library doc ####

        # write if statement
        lib_arg_doc_towrite += (
            f"    if library = ``{lib_name}``:       (see: ``{function.__name__}``)")
        if lib_name == "imgaug":
            base_class = str(function).split(".")[-2]
            full_name = function.__module__ + '.' + function.__qualname__
            website = f"https://imgaug.readthedocs.io/en/latest/source/api_augmenters_{base_class}.html#{full_name}"
        elif lib_name == "torchvision":
            website = f"https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#{function.__name__}"
        elif lib_name == "custom":
            website = f"https://github.com/adobe-research/beacon-aug/tree/master/beacon_aug/generator/custom/{function.__name__}.py"
        elif lib_name == "gan_based":
            website = f"https://github.com/adobe-research/beacon-aug/tree/master/beacon_aug/generator/gan_based/{function.__name__}.py"

        elif lib_name == "mmcv":
            website = f"https://github.com/adobe-research/beacon-aug/blob/master/beacon_aug/external/mmcv/mmseg/transforms.py"
        elif lib_name == "keras":
            website = f"https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/{function.__name__ }"
        else:
            website = None
        url_doc = (f"\n") if website == None else (
            f"       `[Source] <{website}>`_ \n\n")
        lib_arg_doc_towrite += url_doc

        # write arguments
        if lib_doc is not None:
            if lib_name == "imgaug":
                arg_tags = ["Parameters", "Attributes", "Variables"]
                for tag in arg_tags:
                    if f"{tag}\n    ----------\n" in lib_doc:
                        lib_doc_paras = lib_doc.split(
                            f"{tag}\n    ----------\n")[1].split("Examples\n")[0]
                if verbose:
                    print("Error for lib_doc_paras", function)

                discrip_others = None
                for key in getfullargspec(function)[0]:
                    if key not in _ARGS_NAMES_EXCEPT:

                        discrip = lib_doc_paras.split(
                            key+" :")[1].split("\n\n")[0]          # key+" :" +
                        # replace it with the cited key
                        if "See :class:`Affine`." in discrip:
                            discrip = iaa.Affine.__doc__.split(
                                key+" :")[1].split("\n\n")[0]
                        # add \t for each line
                        discrip_splited = ["        "+_ +
                                           "\n" for _ in discrip.split("\n")[1:]]
                        discrip_others = "".join(discrip_splited)
                        arg_type = discrip.split("\n")[0]
                        if verbose == True:
                            print("*"*10, key, "\n", discrip_others)
                        full = "            " + key + \
                            "(" + arg_type + "):\n" + discrip_others
                        lib_arg_doc_towrite += (f"{full}\n\n")
            elif lib_name == "torchvision":
                if "Args:" in lib_doc:
                    # pdb.set_trace()
                    discrip = lib_doc.split("Args:")[1].split(".. _filters:")[0]
                    # add \t for each line
                    discrip_splited = ["    "+_+"\n" for _ in discrip.split("\n")]
                    full = "".join(discrip_splited)
                    lib_arg_doc_towrite += (full.split("\n    \n        ")[0])
                    lib_arg_doc_towrite += ("\n\n")
                else:
                    lib_arg_doc_towrite += ("\t\t" + lib_doc)
            elif lib_name == "custom" or lib_name == "gan_based":
                if "Args:" in lib_doc:
                    discrip = lib_doc.split("Args:")[1].split("----")[0]
                    # add \t for each line
                    discrip_splited = ["    "+_+"\n" for _ in discrip.split("\n")]
                    full = "".join(discrip_splited)
                    lib_arg_doc_towrite += (full.split("\n    \n        ")[0])
                    lib_arg_doc_towrite += ("\n\n")
                else:
                    lib_arg_doc_towrite += ("\t\t" + lib_doc)
            elif lib_name == "mmcv":
                if "Args:" in lib_doc:
                    # pdb.set_trace()
                    discrip = lib_doc.split("Args:")[1]
                    # add \t for each line
                    discrip_splited = ["    "+_+"\n" for _ in discrip.split("\n")]
                    full = "".join(discrip_splited)
                    lib_arg_doc_towrite += (full.split("\n    \n        ")[0])
                    lib_arg_doc_towrite += ("\n\n")
                else:
                    lib_arg_doc_towrite += ("\t\t" + lib_doc)

            elif lib_name == "keras":
                discrip = lib_doc.split("Arguments")[1].split("# Returns")[0]
                # add \t for each line
                discrip_split = ["    "+_+"\n" for _ in discrip.split("\n")]
                full = "".join(discrip_split)
                lib_arg_doc_towrite += (full)
                lib_arg_doc_towrite += ("\n\n")

            else:  # other libraries not customized auto doc
                lib_arg_doc_towrite += ("\t\t" + lib_doc)

    # insert lib_arg_doc_towrite before "Targets"
    if A_doc != "":
        A_doc = A_doc.replace("Targets:\n", lib_arg_doc_towrite + "    Targets:\n")
    else:
        A_doc = f'''        library (str):  flag for library. Should be one of: {avail_lib_text}.
                        Default: `{avail_libraries[0]}`.\n'''
        A_doc += "    Args:\n\t"+lib_arg_doc_towrite
    Final_doc += (A_doc)

    # get example default statements
    example_doc = ""
    if op_dic is not None:
        for lib_name in avail_libraries:
            default_statement = ""
            default_paras = None
            if lib_name not in ["custom", "gan_based"]:
                if lib_name == "albumentations" and "default_para" in op_dic:
                    default_paras = op_dic["default_para"]
                if lib_name != "albumentations" and "default_lib_paras" in op_dic[lib_name]:
                    default_paras = op_dic[lib_name]["default_lib_paras"]

            if default_paras is not None:
                for i in default_paras:
                    default_statement += str(i) + "=" + str(default_paras[i]) + ","

                example_doc += (
                    f'''        aug = BA.{op_name}(p=1, {default_statement[:-1]},library="{lib_name}")\n''')

    Final_doc += (f'''
    e.g.
    
    .. code-block::

        import beacon_aug as BA
{example_doc}
        image_auged = aug(image=image)["image"]
    ''')

    return Final_doc
