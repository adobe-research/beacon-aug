# Copyright 2021 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import numpy as np
from .synthesis import TextSynthesis
import sys
from os.path import dirname, realpath, join
DIR_PATH = dirname(realpath(__file__))
sys.path.append(DIR_PATH)
sys.path.append(join(DIR_PATH, 'freetype'))


def text_synthesis(img, text_list, font_list=None, region_list=None, effect_list=None,
                   is_top_left_origin=True, ensure_num_regions=False, is_keep_ratio=True):
    """
    place multiple text on a image
    :param img: str or array, image path or RGB image
    :param text_list: list of str
    :param font_list: list of font file path
    :param region_list: list of regions, a region is a list of 4 corners in x-y coord
    :param effect_list: list of effects, a effect is a dict, please refer to self._place_text_on_region()
    :param color_model_path: str, path to the color model for coloring the text according to bg color
    :param is_top_left_origin: whether keep the first point as the top-left corner in a region vector
    :param ensure_num_regions: whether ensure the number of text placed on the image
    :param is_keep_ratio: whether keep the ratio of rendered text
    :return: dict,
        img: text image
        txt: ground truth, list of str
        bbx: bounding box, list of 4x2 arrays, each array has 4 points in image (x-y) coord
        cbx: character boxes, list of Nx4x2 arrays, each array has N (the number of characters) bounding boxes
        bln: baseline, list of 2x2 arrays, each array has two points, which indicate a line in 2D space
    """
    if font_list is None:
        font_list = [DIR_PATH + '/data/font/Vera.ttf',
                     DIR_PATH + 'data/font/VeraMono.ttf']

    synthesis = TextSynthesis()
    data = synthesis(
        img=img,
        text_list=text_list,
        font_list=font_list,
        region_list=region_list,
        effect_list=effect_list,
        is_top_left_origin=is_top_left_origin,
        ensure_num_regions=ensure_num_regions,
        is_keep_ratio=is_keep_ratio,
    )

    return {
        'img': data['image'].astype(np.uint8),
        'txt': data['text'],
        'bbx': data['wbb'],
        'cbx': data['cbb'],
        'bln': data['base'],
        'mask': data['mask']
    }
