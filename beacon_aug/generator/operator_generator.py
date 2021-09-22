# Copyright 2021 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.


"""
Inspriation source:
[1] https://github.com/albumentations-team/albumentations/blob/master/albumentations/imgaug/transforms.py
[2] https://github.com/albumentations-team/albumentations/blob/master/benchmark/benchmark.py

#imgaug.augmenters.geometric.Rotate
[imgaug]  https://imgaug.readthedocs.io/en/latest/source/api_augmenters_geometric.html?highlight=Rotate
[Keras] https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image/affine_transformations.py
[torchvision] https://pytorch.org/vision/stable/transforms.html
"""
# beacon_aug
from . import DEFAULT_LIBRARIES, load_avail_libraries, load_config, avail_libraries
from . import custom, gan_based
from .docs_generator import generate_doc

# imgaug
from re import L
import imgaug as ia
from numpy.core.fromnumeric import shape

from beacon_aug import generator
try:
    from imgaug import augmenters as iaa
except ImportError:
    import imgaug.imgaug.augmenters as iaa
from imgaug.augmenters import geometric as iaa_geometic

# albumentations
import albumentations as A
from albumentations.augmentations.bbox_utils import convert_bboxes_from_albumentations, convert_bboxes_to_albumentations
from albumentations.augmentations.keypoints_utils import convert_keypoints_from_albumentations, convert_keypoints_to_albumentations
from albumentations.core.transforms_interface import BasicTransform, DualTransform, ImageOnlyTransform, to_tuple
from albumentations.augmentations.functional import MAX_VALUES_BY_DTYPE
# torchvision
import torchvision.transforms.functional as torch_f
import torchvision.transforms as torch
import keras_preprocessing.image as keras
from torch import Tensor
from torch import from_numpy

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
import importlib
import os
import cv2
from PIL import Image
import json
import yaml
import numpy as np
import warnings
import os
import inspect
import sys


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
__all__ = [
    "BasicBATransform",
    "DualBATransform",
    "ImageOnlyBATransform",
]


MAX_VALUE = MAX_VALUES_BY_DTYPE[np.dtype("uint8")]

# {v: k for k, v in iaa_geometic._AFFINE_MODE_SKIMAGE_TO_CV2.items()}  # reverse map
_AFFINE_MODE_CV2_TO_SKIMAGE = {0: 'constant', 1: 'edge', 2: 'symmetric', 4: 'reflect', 3: 'wrap'}
_CODE_TO_CV2_KERAS_MODE = {0: 'nearest', 1: 'wrap', 2: 'reflect', 3: 'mirror', 4: 'constant'}
_INTERPOLATION_MODE = {0: torch_f.InterpolationMode.NEAREST,
                       2: torch_f.InterpolationMode.BILINEAR,
                       3: torch_f.InterpolationMode.BICUBIC,
                       4: torch_f.InterpolationMode.BOX,
                       5: torch_f.InterpolationMode.HAMMING,
                       1: torch_f.InterpolationMode.LANCZOS}
_INTERPOLATION_MODE_INVERSE = {v: k for k, v in _INTERPOLATION_MODE.items()}
_ARGS_NAMES_EXCEPT = ["save_key", "replay", "params",
                      "replay_mode", "applied_in_replay", "torchvision_op"]


def get_library_fullname(library):
    # print ("library=", library)
    if library.lower() in ["albumentations", "a"]:
        library = "albumentations"
    elif library.lower() in ["imgaug", "iaa"]:
        library = "imgaug"
    elif library.lower() in ["torchvision", "torch"]:
        library = "torchvision"
    elif library.lower() == "pillow":
        library = "pillow"
    elif library.lower() == "keras":
        library = "keras"
    elif library.lower() == "augmentor":
        library = "augmentor"
    elif library.lower() == "solt":
        library = "solt"
    elif library.lower() == "augly":
        library = "augly"
    elif library.lower() == "mmcv":
        library = "mmcv"
    elif library.lower() in ["imagenet_c", "imagenet-c"]:
        library = "imagenet_c"
    elif library.lower() in ["kornia"]:
        library = "kornia"
    elif library.lower() == "custom":
        library = "custom"
    elif library.lower() == "gan_based":
        library = "gan_based"
    else:
        assert library in DEFAULT_LIBRARIES, "'library' has to be in: " + \
            "'".join(map(str, DEFAULT_LIBRARIES)) + \
            "\nCheck if the dependencies is installed"
    return library


def random_uniform(x):
    '''
    Support both list/tuple or digit input for library variable parser
    return number
    '''
    if type(x) == list or type(x) == tuple:
        return np.random.uniform(*x)
    else:
        return np.random.uniform(x)


def format_check(x):
    if type(x) == list or type(x) == tuple:
        return np.array(x)
    return x

##### Parse Parameters ######


def parse_val(val):
    # convert string to variable; list to tuple
    val = eval(val) if (type(val) == str and "." in val) else val
    val = tuple(val) if (type(val) == list) else val
    return val


def parse_default_paras(function, default_paras, kwargs):
    if default_paras:
        lib_kwargs_updated_add_default = kwargs.copy()
        for default_lib_parname in default_paras:
            # the parameter never been parsed and the name exist in this function
            if default_lib_parname not in kwargs and \
                    default_lib_parname in inspect.getfullargspec(function)[0]:
                val = parse_val(default_paras[default_lib_parname])
                lib_kwargs_updated_add_default[default_lib_parname] = val
        kwargs = lib_kwargs_updated_add_default
    return kwargs


def parameter_exchanger(function, alb2lib=None, class_kwargs=None,
                        default_paras=None, default_lib_parnames=None, verbose=None):
    """Parse the parameter from library1 to library2
    # Arguments
        alb2lib (dictionary): { <lib_parname>:  [<alb_parname> , exchange_equation(1to2)  ]   }
            e.g.  (lib1=albumentation(value), lib2=imgaug(cval))
            paras_albumentation2imgaug = { "cval": ["value", lambda x:x ]   }
            paras_albumentation2imgaug = {
                "cval": ["value", "x" ]   }              # equivalent to
        function: the library2 transformation function that to be exam and put
            all <lib_parname> belong to library.function(**kwargs)
        class_kwargs: common class kwargs
    # Returns
        updated kwargs for this function.
    """

    def parse_equation(lambda_function):
        # Parse the lambda equation function: lambda_function must has "x"
        # return:  function

        if type(lambda_function) == str:
            # (*x) then means the variable support both list and int/float, need to return list
            if "(*x)" in lambda_function:
                lambda_function = lambda_function.replace("(*x)", "x")
                # guarantee the input meet the list requirement : calling format_check
                if ("-" in lambda_function or "+" in lambda_function) and "random_uniform" not in lambda_function:
                    lambda_function = lambda_function.replace("x", "(format_check(x))")
                return lambda x: list(eval(lambda_function))
            else:
                # guarantee the input meet the list requirement : calling format_check
                if ("-" in lambda_function or "+" in lambda_function) and "random_uniform" not in lambda_function:
                    lambda_function = lambda_function.replace("x", "format_check(x)")
                return lambda x: eval(lambda_function)
        else:
            return lambda_function

    if verbose:
        print("\nIn ", str(function), ": \n\t original kwargs= ", class_kwargs, "\n")

    # 1) parse default common arguments
    if default_paras:
        kwargs_add_default = class_kwargs.copy()
        for default_para in default_paras:
            if default_para not in class_kwargs:                                                # alb para not been parsed
                for kw in class_kwargs:
                    # if corresponding lib arg also been parsed
                    if kw in alb2lib and alb2lib[kw][0] == default_para:
                        # deprecate this parse
                        default_paras[default_para] = None
                if default_paras[default_para]:
                    val = parse_val(default_paras[default_para])
                    # add default_para with kwargs
                    kwargs_add_default[default_para] = val
        class_kwargs = kwargs_add_default
    if verbose:
        print(": \n\t 1) default added kwargs= ", class_kwargs, "\n")

    # 2) Convert all alb paras to lib specifics
    if alb2lib is not None:
        for lib_parname in alb2lib.keys():
            [alb_parname, exchange_equation] = alb2lib[lib_parname]
            # parse single alb_parname to lib
            if "," not in alb_parname:
                if alb_parname not in inspect.getfullargspec(function)[0] and alb_parname in class_kwargs:
                    # add the correspond lib2 parameter
                    equation = parse_equation(exchange_equation)
                    if verbose:
                        print("  \n\t lambda quation(str) = ", exchange_equation, "\n")
                        print("  \t input variable = ", class_kwargs[alb_parname], "\n")
                        print("  \t parsed quation(function) = ",
                              inspect.getsource(equation), "\n")
                    class_kwargs[lib_parname] = equation(class_kwargs[alb_parname])
                    # clean up the lib key
                    class_kwargs.pop(alb_parname)
            # parse multiple alb_parnames to lib
            else:
                alb_parname_ls = alb_parname.split(",")
                check_all = np.prod([(i not in inspect.getfullargspec(function)[
                                    0] and i in class_kwargs) for i in alb_parname_ls])
                if check_all:
                    equation = parse_equation(exchange_equation)
                    class_kwargs[lib_parname] = equation(
                        [class_kwargs[i] for i in alb_parname_ls])
                    # clean up the lib key
                    [class_kwargs.pop(i) for i in alb_parname_ls]
        if verbose:
            print(": \n\t 2) after alb to lb , class_kwargs= ", class_kwargs, "\n")

    # 3) Remove the redundant kwargs
    if class_kwargs is not None:
        class_kwargs_temp = class_kwargs.copy()
        for kwarg in class_kwargs_temp.keys():
            val = parse_val(class_kwargs_temp[kwarg])
            class_kwargs[kwarg] = val
            if kwarg not in inspect.getfullargspec(function)[0]:
                # clean up the lib1 key
                class_kwargs.pop(kwarg)
                warnings.warn("\n%ris not used in this library. It has been omitted.\n" %
                              (kwarg), UserWarning)

    lib_kwargs_updated = class_kwargs
    # 4) Parse default lib arguments
    if default_lib_parnames:
        lib_kwargs_updated_add_default = lib_kwargs_updated.copy()
        for default_lib_parname in default_lib_parnames:
            if default_lib_parname not in lib_kwargs_updated.keys():
                val = parse_val(default_lib_parnames[default_lib_parname])
                lib_kwargs_updated_add_default[default_lib_parname] = val
        lib_kwargs_updated = lib_kwargs_updated_add_default
        if verbose:
            print("\n4) lib_kwargs_updated_add_default= ", lib_kwargs_updated_add_default, "\n")

    # 5) Parse probability
    if "p" in inspect.getfullargspec(function)[0]:
        # In torchvision.transform, default p = 0.5, but the operator possibility is
        # controlled by beacon_aug (p). So we set the library backend p to 1
        lib_kwargs_updated["p"] = 1

    if verbose:
        print(": \n\t Final: lib_kwargs_updated = ", lib_kwargs_updated, "\n")

    return lib_kwargs_updated


# Operator class definition
class BasicBATransform(A.BasicTransform):
    def __init__(self, always_apply=False, p=0.5, library="imgaug", **kwargs):
        super(BasicBATransform, self).__init__(always_apply, p)
        self.library = get_library_fullname(library)
        # will replace is later in operator class definition
        self.avail_libraries = []
        self.kwargs = kwargs

    def is_supported_by(self, library):
        # if the library operator exists, claim it is supportable
        return hasattr(self, library+"_op") or hasattr(self, library+"_pipeline")

    @property
    def processor(self):
        if self.is_supported_by(self.library) == True:
            # run library operator
            return getattr(self, self.library+"_op")

        else:
            raise TypeError("Library '" + self.library +
                            "' is not supported in this transformation")

    def update_params(self, params, **kwargs):
        params = super(BasicBATransform, self).update_params(params, **kwargs)
        # for other library, add the property to fix parameter
        if self.library == "imgaug":
            processor = self.processor.to_deterministic()
        else:
            processor = self.processor
        params["processor"] = processor
        return params

    def apply(self, img,  processor=None, as_layer=False, verbose=False, **params):
        '''
        @img: numpy arrary/ PIL image
        @param: as_layer: if run as a tensor layer (only for torch vision)
        '''
        def apply_singleAug_exceptTorch(img):
            # load image, need to convert nparray to PIL
            if self.library in ("augmentor", "pillow", "augly", "imagenet_c") and type(img) == np.ndarray:
                img = Image.fromarray(img).convert("RGB")
            # apply transformation
            if self.library == "albumentations":
                img_auged = processor.apply(img, **params)

            elif self.library == "imgaug":
                img_auged = processor.augment_image(img)

            elif self.library == "keras":
                img = processor(img, **self.keras_paras)
                img_auged = np.ascontiguousarray(img).astype(np.uint8)

            elif self.library == "augly":
                img = processor(img)
                img_auged = np.array(img, np.uint8, copy=False)

            elif self.library == "mmcv":
                img = processor({"img": img})["img"]
                img_auged = np.array(img, np.uint8, copy=False)

            elif self.library == "imagenet_c":
                img = processor(img)
                img_auged = np.array(img, np.uint8, copy=False)

            elif self.library == "custom":
                if hasattr(self, "custom_paras"):
                    img = processor(img, **self.custom_paras)
                else:
                    img = processor(img)
                img_auged = np.array(img, np.uint8, copy=False)

            elif self.library == "gan_based":
                if hasattr(self, "gan_based_paras"):
                    img = processor(img, **self.gan_based_paras)
                else:
                    img = processor(img)
                img_auged = np.array(img, np.uint8, copy=False)
            return img_auged

        processor = self.processor
        if verbose:
            print("*"*10, self.library, ":", processor)

        # '''input'''
        # if load tensor need to convert tensor to nparray
        img_batch = None
        if as_layer == True and type(img) == Tensor and self.library not in ("torchvision"):
            # cautious, it will make  requires_grad=False
            img_batch = img.detach().numpy().T
            img_batch = (img_batch*255).astype(np.uint8)

        # '''apply transformation'''
        # torchvision can directly handle image batch
        if self.library in ["torchvision", "kornia"]:
            if self.library == "torchvision":
                # input: can be PIL or {tensor}( batch x 3 x  Heightx Width)
                if type(img) == np.ndarray:
                    img = Image.fromarray(img).convert("RGB")
                # some torch function require input to be PIL
                try:
                    if hasattr(self, self.library + "_paras"):
                        img = processor(img, **self.torchvision_paras)
                    else:
                        img = processor(img)
                # other torch functions require input to be tensor
                except AttributeError:
                    img_tensor = torch.Compose([torch.ToTensor()])(img)
                    if hasattr(self, self.library+"_paras"):
                        img = processor(img_tensor, **self.torchvision_paras)
                    else:
                        img = processor(img_tensor)
            elif self.library == "kornia":
                # input: need to convert to tensor
                if type(img) == np.ndarray:
                    img = K.image_to_tensor(img)
                    img = K.color.bgr_to_rgb(img)
                img = processor(img, **self.kornia_paras)

            # output
            if as_layer == True:  # return tensor
                return img
            else:                  # return image
                if type(img) == Tensor:
                    return img.permute(1, 2, 0).numpy()
                else:
                    return np.array(img, np.uint8, copy=False)
        else:
            #
            if img_batch is None:
                img_auged = apply_singleAug_exceptTorch(img)
            else:
                img_auged = []
                for batch_i in range(img_batch.shape[3]):
                    img_auged.append(apply_singleAug_exceptTorch(img_batch[:, :, batch_i]))
                img_auged = np.dstack(img_auged)
                img_auged = (img_auged/255).T

        # '''output'''
        if as_layer == False and type(img_auged) != Tensor:
            return img_auged
        else:
            if img_batch is None:
                return torch.Compose([torch.ToTensor()])(img_auged)
            else:
                return from_numpy(img_auged)

    # filter the output args for A.to_dict(operator_class)
    def get_transform_init_args_names(self):
        ls = []
        for key in self.__dict__.keys():
            if key not in _ARGS_NAMES_EXCEPT and self.__dict__[key] != {} and "_op" not in key:
                ls.append(key)
        return tuple(ls)

    def _to_dict(self):      # add library show
        dictionary = super()._to_dict()
        dictionary["library"] = self.library
        # dictionary = {**dictionary, **self.kwargs}
        return dictionary


class DualBATransform(DualTransform, BasicBATransform):
    def apply_to_bboxes(self, bboxes, processor=None, rows=0, cols=0, **params):
        processor = self.processor
        if len(bboxes) > 0:
            bboxes = convert_bboxes_from_albumentations(bboxes, "pascal_voc", rows=rows, cols=cols)

            bboxes_t = ia.BoundingBoxesOnImage(
                [ia.BoundingBox(*bbox[: 4]) for bbox in bboxes], (rows, cols))
            bboxes_t = processor.augment_bounding_boxes([bboxes_t])[0].bounding_boxes
            bboxes_t = [
                [bbox.x1, bbox.y1, bbox.x2, bbox.y2] + list(bbox_orig[4:])
                for (bbox, bbox_orig) in zip(bboxes_t, bboxes)
            ]

            bboxes = convert_bboxes_to_albumentations(bboxes_t, "pascal_voc", rows=rows, cols=cols)
        return bboxes

    """Applies transformation to keypoints.
    Notes:
        Since BA supports only xy keypoints, scale and orientation will remain unchanged.
    TODO:
        Emit a warning message if child classes of DualBATransform are instantiated
        inside Compose with keypoints format other than 'xy'.
    """

    def apply_to_keypoints(self, keypoints, processor=None, rows=0, cols=0, **params):
        processor = self.processor
        if len(keypoints) > 0:
            keypoints = convert_keypoints_from_albumentations(keypoints, "xy", rows=rows, cols=cols)
            keypoints_t = ia.KeypointsOnImage([ia.Keypoint(*kp[: 2])
                                              for kp in keypoints], (rows, cols))
            keypoints_t = processor.augment_keypoints([keypoints_t])[0].keypoints

            bboxes_t = [[kp.x, kp.y] + list(kp_orig[2:])
                        for (kp, kp_orig) in zip(keypoints_t, keypoints)]

            keypoints = convert_keypoints_to_albumentations(bboxes_t, "xy", rows=rows, cols=cols)
        return keypoints


class ImageOnlyBATransform(ImageOnlyTransform, BasicBATransform):
    pass


def Operator_generator(op_name, library=None, *args, **kwargs):
    '''dynamic class definition
    >> operator = type( classname,
                        (superclasses.=,),
                        attributedict= {"__init__": init_function,
                                        "funct2" : funct2} )
    is  the identical to:
    >> class {classname} ({superclasses}):
    >>      def __init__:
    >>          {init_function}
    >>      def funct2:
    >>          {funct2}
    '''

    def albumentation_generator(op_name, operator_paras, library):
        # if the library operator exists, claim it is supportable
        def is_supported_by(self, library):
            return True

        @property
        def __doc__(op_name):
            return generate_doc(op_name)

        A_operator = type("A_" + op_name,                                               # dynamic class definition
                          # inherit from albumentation class
                          (eval(operator_paras["albumentations"]["function"]),),
                          {"is_supported_by":  is_supported_by,
                           "__doc__": __doc__}
                          )
        A_operator.avail_libraries = avail_libraries(operator_paras)

        return A_operator

    def otherlib_generator(op_name, operator_paras, library="imgaug"):

        def BA__init__(self,
                       always_apply=False, p=1, library="imgaug", **kwargs, ):
            super(BA_operator, self).__init__(always_apply, p, library)
            self.avail_libraries = avail_libraries(operator_paras)
            lib_function_name = operator_paras[library]["function"]
            op = eval(lib_function_name) if library not in [
                "custom", "gan_based"] else lib_function_name
            paras_alb2lib = operator_paras[library]["paras_alb2lib"] if (
                "paras_alb2lib" in operator_paras[library]) else {}
            default_lib_paras = operator_paras[library]["default_lib_paras"] if (
                "default_lib_paras" in operator_paras[library]) else None
            default_paras = operator_paras["default_para"] if (
                "default_para" in operator_paras) else None

            lib_kwargs_updated = parameter_exchanger(op, paras_alb2lib, kwargs,
                                                     default_paras, default_lib_paras, verbose=0)

            # construct operators for all libraries
            if self.library == "imgaug":
                self.imgaug_op = op(**lib_kwargs_updated)

            # elif self.library == "albumentation":
            #     self.albumentation_op = op(**lib_kwargs_updated)

            elif self.library == "torchvision":
                torch_lib_class = lib_function_name.split(".")[0]
                if "torch" == torch_lib_class:                                          # call from torchvision.transforms

                    self.torchvision_op = op(**lib_kwargs_updated)
                elif "torch_f" == torch_lib_class:                                      # call from torchvision.transforms.functional, have img arg
                    self.torchvision_op = op
                    self.torchvision_paras = {**lib_kwargs_updated}

            elif self.library == "keras":
                self.keras_op = op
                self.keras_paras = {**lib_kwargs_updated}

            elif self.library == "augly":
                self.augly_op = op(**lib_kwargs_updated)

            elif self.library == "mmcv":
                self.mmcv_op = op(**lib_kwargs_updated)

            elif self.library == "imagenet_c":
                self.imagenet_c_op = op(**lib_kwargs_updated)

            elif self.library == "kornia":
                self.kornia_op = op
                self.kornia_paras = {**lib_kwargs_updated}

            elif self.library == "custom":
                self.custom_op = op(**lib_kwargs_updated)

            elif self.library == "gan_based":
                self.gan_based_op = op(**lib_kwargs_updated)

        BA_operator = type("BA_" + op_name,                                             # dynamic class definition
                           (eval(operator_paras["transform"]), ),
                           {"__init__": BA__init__,
                            "__doc__": generate_doc,
                            }
                           )
        BA_operator.__doc__ = generate_doc(op_name)
        return BA_operator

    # load JSON file
    operator_paras = load_config(op_name)
    # same to self. avail_libraries
    avail_libraries_ls = load_avail_libraries(op_name)
    # set default library == first avail library (mostly "albumentations")
    library = avail_libraries_ls[0] if library is None else library

    library = get_library_fullname(library)
    if library in avail_libraries_ls:
        if library == "albumentations":
            A_op = albumentation_generator(op_name, operator_paras, library)
            A_op.library = library
            A_op.avail_libraries = avail_libraries_ls
            # A_op.__doc__ = generate_doc(op_name)

            # return original alb class
            default_paras = operator_paras["default_para"] if "default_para" in operator_paras else None
            kwargs = parse_default_paras(function=A_op, default_paras=default_paras,
                                         kwargs=kwargs)

            return A_op(*args, **kwargs)
        else:
            BA_op = otherlib_generator(op_name, operator_paras, library)
            # BA_op.__doc__ = generate_doc(op_name)

            # return other lib's class
            return BA_op(library=library, *args, **kwargs)

    else:
        raise TypeError("Library '" + library+"' is not supported in the operator:"+op_name)
