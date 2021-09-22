# Copyright 2021 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.


import random
import numpy as np


class SkinTone:
    """Customize operator: SkinTone Changes
    '''Credit to @Xin Sun  @Manuel Lagunas Arto 
    Source: https://git.corp.adobe.com/rodin/human_relighting/blob/manuel-human-relighting-with-daz-hdrihaven/scene_script/add_textures_to_mtl.py#L34-L43

    Args:
        grid_size:  The size of the grid to hide
                   None (default): randomly choose a grid size from [0, 16, 32, 44, 56]
                   int: a fix grid size
                   tuple/list: randomly choose a grid size from the input
    e.g.

    - Test the function:    

        .. code-block::

            from beacon_aug.generator.custom.SkinTone import SkinTone
            op = SkinTone()
            img_transformed = op(img )

    - Test the augmentation wrapper:   

        .. code-block::

            import beacon_aug as BA
            aug = BA.SkinTone(p=1)
            image_auged = aug(image=image)["image"]

    """

    def __init__(self, ):
        self.image_only = True  # True only applies  to image; False applies to both image and bbox
        self.library = "custom"

    def __call__(self, img):

        # set of colors that represent a diverse variety of skin tones
        SKIN_TONES = [np.array([255, 224, 189]),
                      np.array([255, 205, 148]),
                      np.array([234, 192, 134]),
                      np.array([255, 173, 96]),
                      np.array([255, 227, 159]),
                      np.array([165, 114, 101]),
                      np.array([186, 108, 73]),
                      np.array([173, 100, 82]),
                      np.array([185, 124, 109])]

        skin_tone = random.choices(SKIN_TONES)
        im_skin_tone = np.ones(img.shape) * skin_tone
        img = (img * 0.8 + im_skin_tone * 0.2).astype(np.uint8)
        img = np.clip(img, a_max=255, a_min=0)

        return img
