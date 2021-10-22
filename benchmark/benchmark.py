# Copyright 2021 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.
'''
>> python benchmark.py
'''
import glob
import os
import matplotlib.pyplot as plt
from skimage import io
import time
from PIL import Image
import yaml
import beacon_aug as BA
import beacon_aug.generator.custom as custom
from beacon_aug.generator.operator_generator import DEFAULT_LIBRARIES
import pandas as pd
import imgaug.augmenters as iaa 
import numpy as np

def benchmark(BA_operator, image_data,  batch_size=128,  property="runtime", save_individual_fig_path=None, **kwargs):
    '''    benchmark test
    input: 
    @ BA_operator: operator class
    @ image: numpy array input
    @ batch_size: the number of images to run in one data loader
    @ property: "runtime", "differientable"
    @ save_individual_fig_path

    Output:
        dictonary[{property}][{library}: {propterty value}]

    '''
    avail_libraries = BA_operator(**kwargs).avail_libraries

    numOfLibraries = len(avail_libraries)
    attributes_result = {}

    mask_required_ops = ["CropNonEmptyMaskIfExists","MaskDropout"]

    cropping_bbox_required_ops = ["RandomCropNearBBox"]

    for i, library in enumerate(avail_libraries):
        print("Testing for",BA_operator.__name__, " in ",library)
        t_before = time.time()
        op = BA_operator(p=1, library=library, **kwargs)
        for j in range(batch_size):
            if BA_operator.__name__  in mask_required_ops:
                image_auged = op(image=image_data, mask=(image_data[:,:,0] > 128))["image"]
            elif BA_operator.__name__  in cropping_bbox_required_ops:
                image_auged = op(image=image_data, cropping_bbox = [0,64,0,64])["image"]
            else:
                image_auged = op(image=image_data)["image"]

        t_after = time.time()
        runtime = (t_after - t_before)/batch_size

        if property == "runtime":
            attributes_result[library] = runtime
        elif property == "differentiable":
            attributes_result[library] = BA.properties.isOpDifferentiable(op)

        if save_individual_fig_path:
            img_name = os.path.join(save_individual_fig_path,
                                    BA_operator.__name__+"-" + library+".jpeg")
            if os.path.isfile(img_name):
                print("\n...screenshot individual figure already existed as : ", img_name)
            else:
                if image_auged.min() < 0:   # normalzied case, need to
                    image_auged = image_auged - image_auged.min()
                    image_auged = image_auged/image_auged.max()
                    print("@@@@@@@", image_auged.min())

                plt.imsave(img_name, image_auged)
                print("\n...screenshot individual figure save as : ", img_name)

    return attributes_result


if __name__ == "__main__":

    image = io.imread("../data/example.png")
    SAVE_PATH = "../docs/_images/"
    SAVE_PATH = None


    all_ops = []
    '''Extract all operator names '''
    # standard libs
    ym_fnames = glob.glob("../beacon_aug/generator/standard/*.yaml")
    
    for ym_fname in ym_fnames:
        ym_f = open(ym_fname)
        OP_para = yaml.safe_load(ym_f)
        for op_name in OP_para:     # each operator
            all_ops.append(op_name)
    # Customize
    custom_doc_path = "../beacon_aug/generator/custom/"
    for op_name in custom.__all__:
        if op_name not in all_ops:
            all_ops.append(op_name)

    '''Set necessary default parameters '''
    points_sampler = iaa.DropoutPointsSampler(
                        iaa.RelativeRegularGridPointsSampler(
                            n_cols_frac=(0.05, 0.2),
                            n_rows_frac=0.1),
                        0.2)
    extra_paras = { # converesion
                   "Rotate": {"limit": (30, 30)},
                   "Autocontrast": {"limit": (0.5, 0.5)},
                   "Brightness": {"limit": (0.5, 0.5)},
                   "Saturation": {"saturation": (0.5, 0.5)},
                   "Contrast": {"limit": (0.5, 0.5)},
                   "Posterize": {"num_bits": 2},
                   "Solarize": {"threshold": 64},
                   "Sharpen": {"alpha": (0.5, 0.5)},

                   # remain
                    "DropoutPointsSampler":{"other_points_sampler": iaa.RegularGridPointsSampler(10, 20), "p_drop":0.2},
                    "KeepSizeByResize":{"children": iaa.Crop((20, 40), keep_size=False)},
                    "WithPolarWarping":{"children": iaa.Crop((20, 40), keep_size=False)},
                    "Voronoi":{"points_sampler": points_sampler},

                    # augly
                    "MaskedComposite":{"mask": (image[:,:,0] > 128), "transform_function": None},
                    "OverlayImage":{"overlay":  Image.fromarray(np.uint8(image)).convert('RGB')},
                   # custom
                   "OverlayText": {"size": 50},
                   "TextFlow": {"size": 50},
                   }

    height_width_required_ops = ["RandomCrop", "CenterCrop", "RandomResizedCrop","Resize",
                                "CropNonEmptyMaskIfExists","RandomSizedBBoxSafeCrop",
                                "CenterCropToFixedSize","CenterPadToFixedSize", "CenterCropToFixedSize","PadToFixedSize"]
    for op_name in height_width_required_ops:
        extra_paras[op_name] = {"height":64, "width":64}
    for op_name in all_ops:
        if "BlendAlpha" in op_name:
            extra_paras[op_name] = {"foreground": "iaa.Add(100)"}

    '''Test all operators on different batch sizes '''
    for batch_size in [128, 1024]:
        D = {}   # runtime table dictionary
        for i, op_name in enumerate(all_ops):
        # for i , op_name in enumerate( ["AveragePooling",] ):   # bebug specific operator
                print (i, op_name)
                if op_name in extra_paras:      # parse default paras
                    D[op_name] = benchmark(getattr(BA, op_name), image_data=image.copy(), batch_size=batch_size, property="runtime",
                                        save_individual_fig_path=SAVE_PATH, **extra_paras[op_name])
                else:                           # no need to parse
                    D[op_name] = benchmark(getattr(BA, op_name), image_data=image.copy(),  batch_size=batch_size, property="runtime",
                                        save_individual_fig_path=SAVE_PATH)
        # output
        runtime = pd.DataFrame(D).T

        # change precision 
        # runtime = pd.read_csv("runtime_batch128.csv")
        for v in runtime.columns.values[1:]:
            runtime[v] *=100
        runtime.to_csv("benchmark_results/runtime_batch"+str(batch_size)+"(ms).csv",float_format='%.3f')
