# Copyright 2021 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

from __init__ import *
from skimage import io

# img_dir = '/Users/xiaoli/OneDrive - Adobe/Projs/adobe-research/augmenter/data/example.png'
img_dir = 'data/image/input.png'

text_list = ['Text', 'Flow', 'dlg']
font_list = ['data/font/Vera.ttf', 'data/font/VeraMono.ttf']
effect_list = []

for _ in range(len(text_list)):
    params = {
        # [feather, text color, opacity]
        'layer_text': [np.random.choice(2), 'rand', np.random.uniform(.5, 1)],
        'layer_border': [True, None],  # [whether random color, RGB]
        'layer_shadow': [None, None, None],  # [theta, shift, opacity]
        'layer_background': None,  # RGB, e.g., (100, 100, 100)
        'text_size': 2,
        'text_interval': np.random.uniform(.8, 1.4),
        'mix_bg': np.random.choice(2)
    }
    effect_list.append(params)

data = text_synthesis(
    img=img_dir,
    text_list=text_list,
    font_list=font_list,
    region_list=None,  # automatically select suitable text regions on the image
    effect_list=effect_list,
    is_top_left_origin=True,
    ensure_num_regions=False,
    is_keep_ratio=True,
)
img = data['img']
print(img.shape)
io.imsave('data/image/output.png', img)
