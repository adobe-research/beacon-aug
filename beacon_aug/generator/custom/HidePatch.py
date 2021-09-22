# Copyright 2021 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

'''
# Credit to : Krishna kumar Singh @krishsin
Paper "Hide-and-Seek: A Data Augmentation Technique for Weakly-Supervised Localization and Beyond",
reveal that By hiding images patches randomly, we can force the network to focus on other relevant object
and improve object localization in images and action localization.
detail: http://krsingh.cs.ucdavis.edu/krishna_files/papers/hide_and_seek/hide_seek.html
We add the "hide_patch" function here to provide and customized function for image patch hiding
'''
import random


class HidePatch:
    """Customize operator: HidePatch (advanced cutout)
    '''Source Code: https://github.com/kkanshul/Hide-and-Seek/blob/master/hide_patch.py'''
    Args:
        grid_size:  The size of the grid to hide
                   None (default): randomly choose a grid size from [0, 16, 32, 44, 56]
                   int: a fix grid size
                   tuple/list: randomly choose a grid size from the input
    e.g.

    - Test the function:    

        .. code-block::

            from beacon_aug.generator.custom.HidePatch import HidePatch
            op = HidePatch(grid_size = 16)
            img_transformed = op(img )

    - Test the augmentation wrapper:   

        .. code-block::

            import beacon_aug as BA
            aug = BA.HidePatch(p=1, hide_prob=0.5, grid_size=8)
            image_auged = aug(image=image)["image"]

    """

    def __init__(self, hide_prob=0.5, grid_size=None, patch_value=None):
        self.image_only = True  # True only applies  to image; False applies to both image and bbox
        self.library = "custom"

        if grid_size == None:         # possible grid size, 0 means no hiding
            grid_size = random.choice([0, 16, 32, 44, 56])
        elif type(grid_size) in [list, tuple]:
            grid_size = random.choice(grid_size)
        if patch_value == None:         # value to be used to fill patch
            patch_value = [0, 0, 0]

        # randomly choose one grid size
        self._grid_size = grid_size
        self._hide_prob = hide_prob
        self._patch_value = patch_value

    def __call__(self, img):

        # get width and height of the image
        s = img.shape
        wd = s[0]
        ht = s[1]

        # hide the patches
        if(self._grid_size != 0):
            for x in range(0, wd, self._grid_size):
                for y in range(0, ht, self._grid_size):
                    x_end = min(wd, x+self._grid_size)
                    y_end = min(ht, y+self._grid_size)
                    if(random.random() <= self._hide_prob):
                        img[x:x_end, y:y_end, 0] = self._patch_value[0]
                        img[x:x_end, y:y_end, 1] = self._patch_value[1]
                        img[x:x_end, y:y_end, 2] = self._patch_value[2]

        return img
