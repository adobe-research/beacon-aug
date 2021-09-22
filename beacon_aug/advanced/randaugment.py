# Copyright 2021 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.


'''
Randaugment
    Cubuk, Ekin D., et al. "Randaugment: Practical automated data augmentation with a reduced search space." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. 2020.

The beacon_aug version is adapted from the imgaug:
    https://github.com/aleju/imgaug-doc/blob/7443efbf66263c0c44581ed62501fae6f88b047a/imgaug/augmenters/collections.py
Changes to above:
1) change all operators from imgaug to Beacon_aug  (support all libraries)
2) result equivalent to calling A.Compose(...)
3) simplify the magnitudes similar to autoaug

e.g.

Create a RandAugment augmenter similar to the suggested hyperparameters in the paper.

.. code-block::

    import Beacon_aug as BA
    aug = BA.RandAugment(n=2, m=9)

Create a RandAugment augmenter for COCO dataset

.. code-block::

    aug = BA.RandAugment(policy= "COCO")
    
'''
from numpy import lib
import beacon_aug as BA
import albumentations as A

import cv2
import numpy as np

from imgaug import parameters as iap
from imgaug import random as iarandom
from imgaug.augmenters import meta
from imgaug.augmenters import arithmetic
from imgaug.augmenters import flip
from imgaug.augmenters import pillike
from imgaug.augmenters import size as sizelib
import random


class RandAugment:
    """Apply RandAugment to inputs as described in the corresponding paper.
    See paper::
        Cubuk et al.
        RandAugment: Practical automated data augmentation with a reduced
        search space
    .. note::
        The paper contains essentially no hyperparameters for the individual
        augmentation techniques. The hyperparameters used here come mostly
        from the official code repository, which however seems to only contain
        code for CIFAR10 and SVHN, not for ImageNet. So some guesswork was
        involved and a few of the hyperparameters were also taken from
        https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py .
        This implementation deviates from the code repository for all PIL
        enhance operations. In the repository these use a factor of
        ``0.1 + M*1.8/M_max``, which would lead to a factor of ``0.1`` for the
        weakest ``M`` of ``M=0``. For e.g. ``Brightness`` that would result in
        a basically black image. This definition is fine for AutoAugment (from
        where the code and hyperparameters are copied), which optimizes
        each transformation's ``M`` individually, but not for RandAugment,
        which uses a single fixed ``M``. We hence redefine these
        hyperparameters to ``1.0 + S * M * 0.9/M_max``, where ``S`` is
        randomly either ``1`` or ``-1``.
        We also note that it is not entirely clear which transformations
        were used in the ImageNet experiments. The paper lists some
        transformations in Figure 2, but names others in the text too (e.g.
        crops, flips, cutout). While Figure 2 lists the Identity function,
        this transformation seems to not appear in the repository (and in fact,
        the function ``randaugment(N, M)`` doesn't seem to exist in the
        repository either). So we also make a best guess here about what
        transformations might have been used.
    .. warning::
        This augmenter only works with image data, not e.g. bounding boxes.
        The used PIL-based affine transformations are not yet able to
        process non-image data. (This augmenter uses PIL-based affine
        transformations to ensure that outputs are as similar as possible
        to the paper's implementation.)
    Added in 0.4.0.
    **Supported dtypes**:
    minimum of (
        :class:`~imgaug.augmenters.flip.Fliplr`,
        :class:`~imgaug.augmenters.size.KeepSizeByResize`,
        :class:`~imgaug.augmenters.size.Crop`,
        :class:`~imgaug.augmenters.meta.Sequential`,
        :class:`~imgaug.augmenters.meta.SomeOf`,
        :class:`~imgaug.augmenters.meta.Identity`,
        :class:`~imgaug.augmenters.pillike.Autocontrast`,
        :class:`~imgaug.augmenters.pillike.Equalize`,
        :class:`~imgaug.augmenters.arithmetic.Invert`,
        :class:`~imgaug.augmenters.pillike.Affine`,
        :class:`~imgaug.augmenters.pillike.Posterize`,
        :class:`~imgaug.augmenters.pillike.Solarize`,
        :class:`~imgaug.augmenters.pillike.EnhanceColor`,
        :class:`~imgaug.augmenters.pillike.EnhanceContrast`,
        :class:`~imgaug.augmenters.pillike.EnhanceBrightness`,
        :class:`~imgaug.augmenters.pillike.EnhanceSharpness`,
        :class:`~imgaug.augmenters.arithmetic.Cutout`,
        :class:`~imgaug.augmenters.pillike.FilterBlur`,
        :class:`~imgaug.augmenters.pillike.FilterSmooth`
    )

    n : int or tuple of int or list of int or imgaug.parameters.StochasticParameter or None, optional
        Parameter ``N`` in the paper, i.e. number of transformations to apply.
        The paper suggests ``N=2`` for ImageNet.
        See also parameter ``n`` in :class:`~imgaug.augmenters.meta.SomeOf`
        for more details.
        Note that horizontal flips (p=50%) and crops are always applied. This
        parameter only determines how many of the other transformations
        are applied per image.
    m : int or tuple of int or list of int or imgaug.parameters.StochasticParameter or None, optional
        Parameter ``M`` in the paper, i.e. magnitude/severity/strength of the
        applied transformations in interval ``[0 .. 30]`` with ``M=0`` being
        the weakest. The paper suggests for ImageNet ``M=9`` in case of
        ResNet-50 and ``M=28`` in case of EfficientNet-B7.
        This implementation uses a default value of ``(6, 12)``, i.e. the
        value is uniformly sampled per image from the interval ``[6 .. 12]``.
        This ensures greater diversity of transformations than using a single
        fixed value.
        * If ``int``: That value will always be used.
        * If ``tuple`` ``(a, b)``: A random value will be uniformly sampled per
          image from the discrete interval ``[a .. b]``.
        * If ``list``: A random value will be picked from the list per image.
        * If ``StochasticParameter``: For ``B`` images in a batch, ``B`` values
          will be sampled per augmenter (provided the augmenter is dependent
          on the magnitude).
    cval : number or tuple of number or list of number or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        The constant value to use when filling in newly created pixels.
        See parameter `fillcolor` in
        :class:`~imgaug.augmenters.pillike.Affine` for details.
        The paper's repository uses an RGB value of ``125, 122, 113``.
        This implementation uses a single intensity value of ``128``, which
        should work better for cases where input images don't have exactly
        ``3`` channels or come from a different dataset than used by the
        paper.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.RandAugment(n=2, m=9)
    Create a RandAugment augmenter similar to the suggested hyperparameters
    in the paper.
    >>> aug = iaa.RandAugment(m=30)
    Create a RandAugment augmenter with maximum magnitude/strength.
    >>> aug = iaa.RandAugment(m=(0, 9))
    Create a RandAugment augmenter that applies its transformations with a
    random magnitude between ``0`` (very weak) and ``9`` (recommended for
    ImageNet and ResNet-50). ``m`` is sampled per transformation.
    >>> aug = iaa.RandAugment(n=(0, 3))
    Create a RandAugment augmenter that applies ``0`` to ``3`` of its
    child transformations to images. Horizontal flips (p=50%) and crops are
    always applied.
    """

    _M_MAX = 30

    # according to paper:
    # N=2, M=9 is optimal for ImageNet with ResNet-50
    # N=2, M=28 is optimal for ImageNet with EfficientNet-B7
    # for cval they use [125, 122, 113]

    def __new__(cls,  policy="imagenet-Resnet50", n=None, m=None, interpolation=cv2.INTER_NEAREST,
                cval=0, library="imgaug", *args, **kwargs):
        obj = super(RandAugment, cls).__new__(cls)
        obj.__init__(policy, interpolation, cval, library, *args, **kwargs)

        if n == None or m == None:
            if policy.lower() == "imagenet-EfficientNetB7":  # Paper  Appendix A.2.3
                # N=2, M=28 is optimal for ImageNet with EfficientNet-B7
                n = 2
                m = 28
            elif policy.lower() == "cifar":
                n = 3
                m = 5
            elif policy.lower() == "svhn":
                n = 3
                m = 5
            elif policy.lower() == "coco":
                n = 1
                m = 5
            else:  # policy.lower() == "imagenet-Resnet50":  # Paper  Appendix A.2.3
                # N=2, M=9 is optimal for ImageNet with ResNet-50
                n = 2
                m = 9
        # The paper says in Appendix A.2.3 "ImageNet", that they actually
        # always execute Horizontal Flips and Crops first and only then a
        # random selection of the other transformations.
        # Hence, we split here into two groups.

        initial_augs = obj._create_initial_augmenters_list(m, interpolation, library)
        main_augs = obj._create_main_augmenters_list(m, cval, interpolation, library)

        # # assign random state to all child augmenters
        # for lst in [initial_augs, main_augs]:
        #     for augmenter in lst:
        #         augmenter.random_state = rng
        return A.Compose([A.Sequential(initial_augs),
                          A.SomeOf(transforms=main_augs, n=n)]
                         )

    @classmethod     # Added in 0.4.0.
    def _create_initial_augmenters_list(cls, m, interpolation, library):
        # It's not really clear what crop parameters they use, so we
        # choose [0..M] here.
        # Random crop image and resize to image size

        return [
            BA.HorizontalFlip(p=0.5, library=library),
            BA. KeepSizeCrop()
        ]

    @classmethod       # Added in 0.4.0.
    def _create_main_augmenters_list(cls, m, cval, interpolation, library):
        # pylint: disable=invalid-name

        # In the paper's code they use the definition from AutoAugment,
        # which is 0.1 + M*1.8/10. But that results in 0.1 for M=0, i.e. for
        # Brightness an almost black image, while M=5 would result in an
        # unaltered image. For AutoAugment that may be fine, as M is optimized
        # for each operation individually, but here we have only one fixed M
        # for all operations. Hence, we rather set this to 1.0 +/- M*0.9/10,
        # so that M=10 would result in 0.1 or 1.9.

        def _get_magnitudes(op_name, level, maxval=1):
            '''
            _BINS  # number of intervals /level
            '''
            val = None

            # name: (magnitudes, signed)
            magnitudes_dict = {
                "ShearX": (np.linspace(0.0, 0.3, level), True),
                "ShearY": (np.linspace(0.0, 0.3, level), True),
                "TranslateX": (np.linspace(0.0, 150.0 / 331.0, level), True),
                "TranslateY": (np.linspace(0.0, 150.0 / 331.0, level), True),
                "Rotate": (np.linspace(0.0, 30.0, level), True),
                "Brightness": (np.linspace(0.0, 0.9, level), False),
                "Color": (np.linspace(0.0, 0.9, level), False),
                "Contrast": (np.linspace(0.0, 0.9, level), False),
                "Sharpness": (np.linspace(0.0, 0.9, level), False),
                "Posterize": (np.linspace(1, maxval, level), False),
                "Solarize": (np.linspace(maxval, 0.0, level), False),
                "Cutout": (np.linspace(0, 20 / 32, level), False),
                "AutoContrast": (None, None),
                "Equalize": (None, None),
                "Invert": (None, None),
            }
            ele = magnitudes_dict[op_name]

            if ele[1] == True:
                magnitudes_list = ele[0].tolist()
                sign = (-1) ** np.random.randint(2, size=1)[0]    # -1 ,1
                val = sign * random.choice(magnitudes_list)
            elif ele[1] == False:
                val = random.choice(ele[0])

            if op_name in ["ShearX", "ShearY", "TranslateX", "TranslateY", "Posterize", "Solarize"]:
                val = int(val)
            return val

        return [
            # meta.Identity(),
            BA.Autocontrast(p=1, library=library),
            BA.Equalize(p=1, library=library),
            BA.Invert(p=1, library=library),

            # they use Image.rotate() for the rotation, which uses
            # the image center as the rotation center
            # BA.Rotate(p=1, library=library,
            #           limit=_get_magnitudes("Rotate", m)),

            # paper uses 4 - int_parameter(M, 4)
            BA.Posterize(p=1, library=library,  num_bits=4 - _get_magnitudes("Posterize", m, 3)),

            # paper uses 256 - int_parameter(M, 256)
            BA.Solarize(p=1, library=library,
                        threshold=256 - _get_magnitudes("Solarize", m, 256)),
            # pillike enhance
            BA.EnhanceColor(factor=_get_magnitudes("Color", m), p=1, library=library),
            BA.EnhanceContrast(factor=_get_magnitudes("Contrast", m), p=1, library=library),
            BA.EnhanceBrightness(factor=_get_magnitudes("Brightness", m), p=1, library=library),
            BA.EnhanceSharpness(factor=_get_magnitudes("Sharpness", m), p=1, library=library),

            # ShearX
            BA.Affine(p=1, interpolation=interpolation, cval=cval, library=library,
                      shear=[_get_magnitudes("ShearX", m), 0]),
            # ShearY
            BA.Affine(p=1, interpolation=interpolation, cval=cval, library=library,
                      shear=[0, _get_magnitudes("ShearY", m)]),
            # TranslateX
            BA.Affine(p=1, interpolation=interpolation, cval=cval, library=library,
                      translate_px=[_get_magnitudes("TranslateX", m), 0]),
            # TranslateY
            BA.Affine(p=1, interpolation=interpolation, cval=cval, library=library,
                      translate_px=[0, _get_magnitudes("TranslateY", m)]),

            # paper code uses 20px on CIFAR (i.e. size 20/32), no information
            # on ImageNet values so we just use the same values
            BA.Cutout(p=1, library=library,
                      size=_get_magnitudes("Cutout", m),
                      squared=True,
                      fill_mode="constant",
                      cval=cval),
            BA.FilterBlur(factor=_get_magnitudes("Sharpness", m), p=1, library=library),
            BA.FilterSmooth(p=1, library=library),
        ]
