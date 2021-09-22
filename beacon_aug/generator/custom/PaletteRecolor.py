# Copyright 2021 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

'''
Adopt from "Palette-based Photo Recoloring. ACM Transactions on Graphics (Proc. SIGGRAPH) 34(4), July 2015."
Implementation inspired from https://github.com/VoidChen/CG_final

Proposed by @Jakub Fiser
Code wrapped by @ Rebecca Li
'''
import numpy as np
import random
import sys
from PIL import Image
from beacon_aug.external.Palette_recolor.palette import build_palette
from beacon_aug.external.Palette_recolor.util import rgb2lab, lab2rgb
from beacon_aug.external.Palette_recolor.transfer import image_transfer


def shift_color(rgb, delta_limit=50):
    '''
    color shift:
    Input:
        rgb:  (R,G,B), 0~255
        delta_limit: shift range of rgb color [-delta_limit,+delta_limit]
    Output:
        RGB: (R+/-delta, G+/-delta,,B+/-delta)
    '''
    def limit_range(input, low=0, high=255):
        output = low if input < low else input
        output = high if output > high else output
        return output

    color_shifted = []
    for c in rgb:
        color_shifted.append(limit_range(c + random.randint(-delta_limit, delta_limit)))

    return color_shifted


class PaletteRecolor:
    """Recolor the  image according to the shifted color palette

    Args:
        image(numpy array): input image
        delta_limit (int or list): value or range of recolor shift range

    Reference: "Palette-based Photo Recoloring. ACM Transactions on Graphics (Proc. SIGGRAPH) 34(4), July 2015."

    e.g.

    -  text PaleteeRecolor function:    

        .. code-block::

            from beacon_aug.generator.custom.PaletteRecolor import PaletteRecolor
            op = PaletteRecolor()
            img_transformed = op(img)

    -  text PaleteeRecolor with augmentation wrapper:    

        .. code-block::

           aug = BA.PaletteRecolor(p=1)
           image_auged = aug(image=image)["image"]

    """

    def __init__(self, delta_limit=20):

        self.library = "custom"
        self.image_only = True  # True only applies  to image; False applies to both image and bbox

        self.delta_limit = delta_limit

    def __call__(self, img):

        image = Image.fromarray(img)

        lab = rgb2lab(image)

        # 1. build palette  (Use K-means to extract the palette of image)

        palette = build_palette(lab)

        # 2. shift  RGB color code of palette with delta
        palette_m = [shift_color(rgb, delta_limit=self.delta_limit) for rgb in palette]

        # 3. tansfer image according to the shifted color
        image_lab_m = image_transfer(lab, palette, palette_m,
                                     sample_level=10, luminance_flag="false")
        image_rgb_m = lab2rgb(image_lab_m)

        return np.asarray(image_rgb_m)
