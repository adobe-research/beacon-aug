# Copyright 2021 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.


'''
Adopt from Zhifei Zhang@zzhang 's TextFlow library
https://git.corp.adobe.com/zzhang/TextFlow
'''
import numpy as np
import random
from beacon_aug.external.textflow import text_synthesis


class TextFlow:
    """crop a random part of the input and rescale it to some size.

    Args:
        text (str): overlay text
        x (int or list): value or range of the x_coordinate of text, 
                Default:  None. (random select in range of image)
        y (int or list): value or range of the y_coordinate of text, 
                Default:  None. (random select in range of image)      

        size (int or list): value or range of the size of text, 
                Default:  25    
        color ( RGB value): value of text color
                Default:  (0,255,0)   

    e.g.

    -  text overlap in random place of image:    

        .. code-block::

            aug = BA.TextFlow(p=1, text="Text", library="custom")
           image_auged = aug(image=image)["image"]

    -  a list of text overlap in limit place of image
        .. code-block::

            import beacon_aug as BA
            aug = BA.TextFlow(p=1, text=[ "Text1","Text2" ],  library="custom")
            image_auged = aug(image=image)["image"]

    """

    def __init__(self, text="Text", x=None, y=None, size=10, color=(255, 255, 0)):

        self.library = "custom"
        self.image_only = True  # True only applies  to image; False applies to both image and bbox

        self.x = x
        self.y = y
        self.size = size
        self.color = color  # True only applies  to image; False applies to both image and bbox
        self.text = text

    def __call__(self, img):
        # load image
        size = random.randint(self.size[0], self.size[1]) if type(
            self.size) != int and float else self.size

        # default x, y ( range to whole image shape)
        if self.x is None:
            self.x = [0, min(img.shape[0] - size, 0)]
        if self.y is None:
            self.y = [0, min(img.shape[1] - size, 0)]
        # print(self.x)
        # random x, y location
        x = random.randint(self.x[0], self.x[1]) if type(self.x) != int and float else self.x
        y = random.randint(self.y[0], self.y[1]) if type(self.x) != int and float else self.x

        # region_list: list of regions, a region is a list of 4 corners in x-y coord
        region_list = [[[x, y],
                        [x + size, y],
                        [x + size, y+size],
                        [x, y+size]
                        ]]

        text = random.choice(self.text) if type(self.text) != str else self.text

        text_list = [text]
        effect_list = []

        for _ in range(len(text_list)):
            params = {
                # [feather, text color, opacity]
                'layer_text': [np.random.choice(2), 'rand', np.random.uniform(.5, 1)],
                'layer_border': [True, None],  # [whether random color, RGB]
                'layer_shadow': [None, None, None],  # [theta, shift, opacity]
                'layer_background': None,  # RGB, e.g., (100, 100, 100)
                'text_size': 2*10,
                'text_interval': np.random.uniform(.8, 1.4),
                'mix_bg': np.random.choice(2)
            }
            effect_list.append(params)

        data = text_synthesis(
            img=img,
            text_list=text_list,
            region_list=region_list,  # automatically select suitable text regions on the image
            effect_list=effect_list,
            is_top_left_origin=True,
            ensure_num_regions=False,
            is_keep_ratio=True,
        )

        return data['img']
