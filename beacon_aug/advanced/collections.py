# Copyright 2021 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.


# import beacon_aug as BA
from .. import __init__ as BA
import numpy as np


def Collections(tag="color", p=1.0, n=1,  severity=[1, 5], *args, **kwargs):
    """ Apply run random number of operators according to hint

    Parameters
    ----------
    policy : "color", "geometric", "benign"
    n : number of operators to apply evertime
        default: 1
        n = "all" : all  operators
        n = int : randomly pick n number of operators
        n = [n0, n1] : randomly pick a number of operators, the number is randomly selected from [n0, n1]


    e.g.

    .. code-block::

        auto_pipeline = BA.Collections(tag = "color")
        image_auged = auto_pipeline(image=image)["image"]

    """
    op_list = []
    if tag.lower() == "color":
        op_list = [BA.Saturation(p=p, saturation=0.2),
                   BA.Brightness(p=p, limit=[-0.2, 0.2]),
                   BA.Contrast(p=p, limit=[-0.2, 0.2]),
                   BA.Autocontrast(p=p),
                   BA.Posterize(p=p, num_bits=4),
                   BA.Solarize(p=p, threshold=128),
                   BA.Equalize(p=p),
                   BA.Invert(p=p),
                   BA.Sharpen(p=p, alpha=[0.2, 0.5], lightness=[0.5, 1.0]),
                   BA.EnhanceColor(p=p),
                   BA.EnhanceContrast(p=p),
                   BA.EnhanceSharpness(p=p),
                   ]

    elif tag.lower() == "geometric":
        op_list = [BA.Rotate(p=p),
                   BA.KeepSizeCrop(p=p),
                   BA.HorizontalFlip(p=p),
                   BA.VerticalFlip(p=p),
                   BA.Affine(p=p),
                   BA.Cutout(p=p, num_holes=8),
                   ]

    elif tag.lower() in ["imagenet-c", "imagenet_c"]:
        # [1] D. Hendrycks and T. Dietterich. Benchmarking neural network robustness to
        #    common corruptions and perturbations.In Proc. ICLR, 2019."
        # e.g.        BA.Collections(tag="imagenet-c",  severity=[1, 5])
        severity = [1, 5] if severity == None else severity  # randomly select severity from 0-5
        op_list = [BA.GaussianNoise(p=p, severity=severity,  library="imagenet-c"),
                   BA.ShotNoise(p=p, severity=severity, library="imagenet-c"),
                   BA.ImpulseNoise(p=p, severity=severity, library="imagenet-c"),
                   BA.DefocusBlur(p=p, severity=severity, library="imagenet-c"),
                   # disabled since it takes too long to generate, thus making it unsuitable for using during training
                   BA.GlassBlur(p=p, severity=severity, library="imagenet-c"),
                   BA.MotionBlur(p=p, severity=severity, library="imagenet-c"),
                   BA.ZoomBlur(p=p, severity=severity,  library="imagenet-c"),
                   BA.Frost(p=p, severity=severity, library="imagenet-c"),
                   BA.Fog(p=p, severity=severity, library="imagenet-c"),
                   BA.Brightness(p=p, severity=severity, library="imagenet-c"),
                   BA.Contrast(p=p, severity=severity, library="imagenet-c"),
                   BA.ElasticTransform(p=p, severity=severity, library="imagenet-c"),
                   BA.Pixelization(p=p, severity=severity, library="imagenet-c"),
                   BA.SpeckleNoise(p=p, severity=severity, library="imagenet-c"),
                   BA.GaussianBlur(p=p, severity=severity, library="imagenet-c"),
                   BA.Spatter(p=p, severity=severity, library="imagenet-c"),
                   BA.Saturation(p=p, severity=severity, library="imagenet-c"),
                   BA.JpegCompression(p=p, severity=severity, library="imagenet-c"),
                   # BA.JpegCompression(p=p, quality_lower=0.4, quality_upper=1),  # disabled since it should be applied at the very end
                   ]
    else:
        raise NameError("`tag` is not supported!")

    # number of operators
    if type(n) in [list, tuple]:
        n = np.random.randint(n[0], n[1]+1)
    elif type(n) == int:
        n = n
    else:  # "all"
        n = len(op_list)
    return BA.SomeOf(op_list, n)
