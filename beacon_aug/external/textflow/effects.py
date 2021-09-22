# Copyright 2021 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import cv2
import numpy as np
from scipy.ndimage import interpolation
from .poisson_reconstruct import blit_images
import inspect
from os.path import dirname, realpath, join


class Layer(object):
    def __init__(self, alpha, color):
        """
        create alpha layer
        """

        """ alpha map for the image """
        assert alpha.ndim == 2
        self.alpha = alpha
        h, w = alpha.shape

        """ color map for the image """
        assert isinstance(color, (int, float, list, tuple, np.ndarray))
        color = np.atleast_1d(np.array(color)).astype(np.uint8)
        if color.ndim == 1:  # constant color for whole layer
            if color.size == 1:  # gray scale layer
                self.color = color * np.ones((h, w, 3), np.uint8)
            if color.size == 3:
                self.color = np.ones((h, w, 3), np.uint8) * color[None, None, :]
        elif color.ndim == 2:  # gray image
            self.color = np.stack([color] * 3).astype(np.uint8)
        elif color.ndim == 3:  # rgb image
            self.color = color
        else:
            print(color.shape)
            raise Exception("Unrecognized color data type!")


class FontColor(object):

    def __init__(self, col_file):
        self.colorsRGB = np.load(col_file)['font_color']
        self.n_color = self.colorsRGB.shape[0]

        # convert color-means from RGB to LAB for better nearest neighbour
        self.colorsLAB = np.r_[self.colorsRGB[:, 0:3], self.colorsRGB[:, 6:9]].astype(np.uint8)
        self.colorsLAB = np.squeeze(cv2.cvtColor(self.colorsLAB[None, :, :], cv2.COLOR_RGB2LAB))

    def sample_normal(self, color_mean, color_std):
        """
        sample RGB color from a normal distribution
        :param color_mean: mean of the normal distribution
        :param color_std: stander deviation of the normal distribution
        :return: RGB color vector
        """
        color_sample = color_mean + color_std * np.random.randn()
        return np.clip(color_sample, 0, 255).astype(np.uint8)

    def sample_from_data(self, bg_arr, rand=True):
        """
        sample RGB color from the background image
        :param bg_arr: a RGB image
        :param rand: whether sample randomly around the mean color
        :returns: foreground RGB, background RGB
        """

        """ get background color mean """
        bg_arr = cv2.cvtColor(bg_arr, cv2.COLOR_RGB2LAB)
        bg_arr = np.reshape(bg_arr, (np.prod(bg_arr.shape[:2]), 3))
        bg_mean = np.mean(bg_arr, axis=0)

        """ get nearest color in the color model """
        norms = np.linalg.norm(self.colorsLAB - bg_mean[None, :], axis=1)
        # nn = np.random.choice(np.argsort(norms)[:3])  # choose a random color amongst the top 3 closest matches
        nn = np.argmin(norms)
        data_col = self.colorsRGB[np.mod(nn, self.n_color), :]
        if rand:
            col1 = self.sample_normal(data_col[:3], data_col[3:6])
            col2 = self.sample_normal(data_col[6:9], data_col[9:12])
        else:
            col1 = np.array(data_col[:3]).astype(np.uint8)
            col2 = np.array(data_col[6:9]).astype(np.uint8)

        """ set foreground and background colors """
        if nn < self.n_color:
            return col2, col1
        else:
            return col1, col2

    def complement(self, rgb_color):
        """
        return a color which is complementary to the RGB_COLOR.
        """
        col_hsv = np.squeeze(cv2.cvtColor(rgb_color[None, None, :], cv2.COLOR_RGB2HSV))
        col_hsv[0] = col_hsv[0] + 128  # uint8 mods to 255
        col_comp = np.squeeze(cv2.cvtColor(col_hsv[None, None, :], cv2.COLOR_HSV2RGB))
        return col_comp

    def triangle_color(self, col1, col2):
        """
        Returns a color which is "opposite" to both col1 and col2.
        """
        col1, col2 = np.array(col1), np.array(col2)
        col1 = np.squeeze(cv2.cvtColor(col1[None, None, :], cv2.COLOR_RGB2HSV))
        col2 = np.squeeze(cv2.cvtColor(col2[None, None, :], cv2.COLOR_RGB2HSV))
        h1, h2 = col1[0], col2[0]
        if h2 < h1:
            h1, h2 = h2, h1  # swap
        dh = h2 - h1
        if dh < 127:
            dh = 255 - dh
        col1[0] = h1 + dh / 2
        return np.squeeze(cv2.cvtColor(col1[None, None, :], cv2.COLOR_HSV2RGB))


class TextEffects(object):

    # the order of stack layers, from top to bottom
    _layer_order = [
        'layer_text',
        'layer_border',
        'layer_shadow',
        'layer_background'
    ]

    def __init__(self, color_model_path=None):
        if color_model_path is None:
            color_model_path = join(dirname(realpath(__file__)), 'data', 'font_color.npz')
        self.font_color = FontColor(col_file=color_model_path)
        self.layer_type_dict = dict()
        for method in inspect.getmembers(self, predicate=inspect.ismethod):
            if 'layer_' in method[0]:
                m = method[0][1:].lower()
                self.layer_type_dict[m] = method[1]

    def __call__(self, text_arr, bg_arr, min_char_h, layer_type_params=None, is_mix_bg=False):
        assert isinstance(text_arr, np.ndarray) and text_arr.ndim == 2
        assert isinstance(bg_arr, np.ndarray) and bg_arr.ndim == 3
        assert text_arr.shape[0] == bg_arr.shape[0] and text_arr.shape[1] == bg_arr.shape[1]

        return self._stack_layers(
            text_arr=text_arr,
            bg_arr=bg_arr,
            min_char_h=min_char_h,
            layer_type_params=layer_type_params,
            is_mix_bg=is_mix_bg
        )

    def _stack_layers(self, text_arr, bg_arr, min_char_h, layer_type_params=None, is_mix_bg=False):
        """
        stack effect layer to synthesize image with text
        :param text_arr: 2D maks, {0, 255}
        :param bg_arr: RGB image
        :param min_char_h: the minimum character height in pixel
        :param layer_type_params: dict
            'text'      : [feather (True/False), text color (RGB/None/'rand'), opacity (.75~1)]
            'border'    : [is_rand_color (True/False), color (RGB/None)]
            'shadow'    : [theta (float/None), shift (float/None), opacity (.75~1/None)]
            'background': color (RGB)
        :param is_mix_bg: bool, whether combine text and bg by poisson editing
        :return: RGB image with text
        """

        if layer_type_params is None:
            layer_type_params = dict()

        """ create text layer """
        if TextEffects._layer_order[0] in layer_type_params:
            layers = [self.layer_type_dict[TextEffects._layer_order[0]](
                text_arr, bg_arr, min_char_h, layer_type_params[TextEffects._layer_order[0]])]
        else:
            layers = [self.layer_type_dict[TextEffects._layer_order[0]](
                text_arr, bg_arr, min_char_h)]

        """ create other layers except the background layer"""
        for l in TextEffects._layer_order[1:-1]:
            if l in layer_type_params:
                layers.append(self.layer_type_dict[l](
                    layers[0], bg_arr, min_char_h, layer_type_params[l]))

        """ create background layer """
        if TextEffects._layer_order[-1] in layer_type_params:
            layers.append(self.layer_type_dict[TextEffects._layer_order[-1]](
                layers[0], bg_arr, min_char_h, layer_type_params[TextEffects._layer_order[-1]]))
        else:
            layers.append(self.layer_type_dict[TextEffects._layer_order[-1]](
                layers[0], bg_arr, min_char_h))

        """ stack all layers by alpha blending """
        img_merged = self._merge_down(layers)

        """ poisson image editing """
        if is_mix_bg:
            img_merged = blit_images(img_merged, bg_arr)

        return img_merged.astype(np.uint8)

    def color_text(self, text_arr, bg_arr, rand=True, color=None):
        """
        Decide on a color for the text:
            - could be some other random image.
            - could be a color based on the background.
                this color is sampled from a dictionary built
                from text-word images' colors. The VALUE channel
                is randomized.
        """
        if color is not None:
            fg_col = color
        else:
            fg_col, bg_col = self.font_color.sample_from_data(bg_arr, rand=rand)
            bg_mean = np.median(bg_arr, axis=0)
            thres = 1800
            # print(np.linalg.norm(fg_col - bg_mean))
            if np.linalg.norm(fg_col - bg_mean) < thres:
                fg_col = self.font_color.complement(fg_col)

        return fg_col

    def color_border(self, col_text, col_bg, rand=True, color=None):
        """
        Decide on a color for the border:
            - could be the same as text-color but lower/higher 'VALUE' component.
            - could be the same as bg-color but lower/higher 'VALUE'.
            - could be 'mid-way' color b/w text & bg colors.
        """
        if color is not None:
            return np.array(color).astype(np.uint8)
        elif rand:
            choice = np.random.choice(3)
        else:
            choice = 1

        col_text = cv2.cvtColor(col_text, cv2.COLOR_RGB2HSV)
        col_text = np.reshape(col_text, (np.prod(col_text.shape[:2]), 3))
        col_text = np.mean(col_text, axis=0).astype(np.uint8)

        vs = np.linspace(0, 1)

        def get_sample(x):
            ps = np.abs(vs - x/255.0)
            ps /= np.sum(ps)
            v_rand = np.clip(np.random.choice(vs, p=ps) + 0.1*np.random.randn(), 0, 1)
            return 255*v_rand

        # first choose a color, then inc/dec its VALUE:
        if choice == 0:
            # increase/decrease saturation:
            col_text[0] = get_sample(col_text[0])  # saturation
            col_text = np.squeeze(cv2.cvtColor(col_text[None, None, :], cv2.COLOR_HSV2RGB))
        elif choice == 1:
            # get the complementary color to text:
            col_text = np.squeeze(cv2.cvtColor(col_text[None, None, :], cv2.COLOR_HSV2RGB))
            col_text = self.font_color.complement(col_text)
        elif choice == 2:
            # choose a mid-way color:
            col_bg = cv2.cvtColor(col_bg, cv2.COLOR_RGB2HSV)
            col_bg = np.reshape(col_bg, (np.prod(col_bg.shape[:2]), 3))
            col_bg = np.mean(col_bg, axis=0).astype(np.uint8)
            col_bg = np.squeeze(cv2.cvtColor(col_bg[None, None, :], cv2.COLOR_HSV2RGB))
            col_text = np.squeeze(cv2.cvtColor(col_text[None, None, :], cv2.COLOR_HSV2RGB))
            col_text = self.font_color.triangle_color(col_text, col_bg)

        # now change the VALUE channel:
        col_text = np.squeeze(cv2.cvtColor(col_text[None, None, :], cv2.COLOR_RGB2HSV))
        col_text[2] = get_sample(col_text[2])  # value
        return np.squeeze(cv2.cvtColor(col_text[None, None, :], cv2.COLOR_HSV2RGB))

    def shadow(self, alpha, theta, shift, size, op=0.80):
        """
        :param alpha : alpha layer whose shadow need to be cast
        :param theta : [0,2pi] -- the shadow direction
        :param shift : shift in pixels of the shadow
        :param size  : size of the GaussianBlur filter
        :param op    : opacity of the shadow (multiplying factor)
        :return      : alpha of the shadow layer (it is assumed that the color is black/white)
        """
        if size % 2 == 0:
            size -= 1
            size = max(1, size)
        shadow = cv2.GaussianBlur(alpha, (size, size), 0)
        dx, dy = shift * np.array([-np.sin(theta), np.cos(theta)])
        shadow = op * interpolation.shift(shadow, shift=[dx, dy], mode='constant', cval=0)
        return shadow.astype(np.uint8)

    def border(self, alpha, size, kernel_type='RECT'):
        """
        alpha : alpha layer of the text
        size  : size of the kernel
        kernel_type : one of [rect,ellipse,cross]

        @return : alpha layer of the border (color to be added externally).
        """
        kdict = {'RECT': cv2.MORPH_RECT, 'ELLIPSE': cv2.MORPH_ELLIPSE,
                 'CROSS': cv2.MORPH_CROSS}
        kernel = cv2.getStructuringElement(kdict[kernel_type], (size, size))
        border = cv2.dilate(alpha, kernel, iterations=1) - alpha
        return border

    def feather(self, text_mask, min_h):
        # determine the gaussian-blur std:
        if min_h <= 15:
            bsz = 0.25
            ksz = 1
        elif 15 < min_h < 30:
            bsz = max(0.30, 0.5 + 0.1 * np.random.randn())
            ksz = 3
        else:
            bsz = max(0.5, 1.5 + 0.5 * np.random.randn())
            ksz = 5
        return cv2.GaussianBlur(text_mask, (ksz, ksz), bsz)

    def _merge_two(self, fore, back):
        """
        merge two FOREground and BACKground layers.
        ref: https://en.wikipedia.org/wiki/Alpha_compositing
        ref: Chapter 7 (pg. 440 and pg. 444):
             http://partners.adobe.com/public/developer/en/pdf/PDFReference.pdf
        """
        a_f = fore.alpha/255.0
        a_b = back.alpha/255.0
        c_f = fore.color
        c_b = back.color

        a_o = a_f + (1 - a_f) * a_b
        c_o = a_f[:, :, None] * c_f + ((1 - a_f) * a_b)[:, :, None] * c_b

        return Layer((255 * a_o).astype(np.uint8), c_o.astype(np.uint8))

    def _merge_down(self, layers):
        """
        :param layers  : a list of LAYER objects with the same size, in the order of from top to bottom
        :return: the merged-down image
        """
        n_layers = len(layers)
        if n_layers > 1:
            out_layer = layers[-1]
            for i in range(-2, -n_layers-1, -1):
                out_layer = self._merge_two(fore=layers[i], back=out_layer)
            return out_layer.color
        else:
            return layers[0].color

    def _layer_text(self, text_arr, bg_arr, min_char_h, param=(False, None, 1.)):
        """
        :param text_arr:
        :param bg_arr:
        :param min_char_h: minimum char height in pixel
        :param param: list, [feather, text color, opacity]
        :return:
        """

        feather, fg_color, opacity = (param if param is not None else (False, None, 1.))

        if feather:
            text_arr = self.feather(text_arr, min_char_h)

        if fg_color is None:
            l_text = Layer(alpha=text_arr, color=self.color_text(text_arr, bg_arr, rand=False))
        elif isinstance(fg_color, str) and fg_color.lower() == 'rand':
            l_text = Layer(alpha=text_arr, color=self.color_text(text_arr, bg_arr, rand=True))
        else:
            l_text = Layer(alpha=text_arr, color=self.color_text(text_arr, bg_arr, color=fg_color))

        l_text.alpha = l_text.alpha * opacity

        return l_text

    def _layer_border(self, text_layer, bg_arr, min_char_h, param=(True, None)):
        """
        :param text_layer:
        :param bg_arr:
        :param min_char_h: minimum char height in pixel
        :param param: list, [bool (whether random color), RGB]
        :return:
        """

        rand, color = (param if param is not None else (True, None))

        if min_char_h <= 15:
            bsz = 1
        elif 15 < min_char_h < 30:
            bsz = 3
        else:
            bsz = 5
        return Layer(
            alpha=self.border(text_layer.alpha, size=bsz),
            color=self.color_border(text_layer.color, bg_arr, rand=rand, color=color)
        )

    def _layer_shadow(self, text_layer, bg_arr, min_char_h, param=(None, None, None)):
        """
        :param text_layer:
        :param bg_arr:
        :param min_char_h:
        :param param: list, [theta, shift, opacity]
        :return:
        """

        theta, shift, opacity = (param if param is not None else (None, None, None))

        if min_char_h <= 15:
            bsz = 1
        elif 15 < min_char_h < 30:
            bsz = 3
        else:
            bsz = 5

        if theta is None:
            theta = np.pi / 4 * np.random.choice([1, 3, 5, 7]) + 0.5 * np.random.randn()

        if shift is None:
            if min_char_h <= 15:
                shift = 2
            elif 15 < min_char_h < 30:
                shift = 7 + np.random.randn()
            else:
                shift = 15 + 3 * np.random.randn()

        if opacity is None:
            opacity = 0.80 + 0.1 * np.random.randn()

        return Layer(self.shadow(text_layer.alpha, theta, shift, 3 * bsz, opacity), 0)

    def _layer_background(self, text_layer, bg_arr, min_char_h, param=None):
        """
        :param text_layer:
        :param bg_arr:
        :param min_char_h:
        :param param: list, RGB
        :return:
        """

        bg_color = param

        if bg_color is not None:
            return Layer(alpha=255 * np.ones_like(text_layer.alpha, np.uint8), color=bg_color)
        else:
            return Layer(alpha=255 * np.ones_like(text_layer.alpha, np.uint8), color=bg_arr)


if __name__ == '__main__':

    bg_arr = cv2.cvtColor(cv2.imread('./data/image/input.png'), cv2.COLOR_BGR2RGB)
    text_mask = np.zeros(bg_arr.shape[:2])
    text_mask[100:200, 100:200] = 255
    min_char_h = 30

    text_effects = TextEffects()

    """ specific effects """
    layer_type_params = {
        'layer_text': [True, (200, 100, 50), .75],  # [feather, text color, opacity]
        'layer_border': [False, (50, 200, 100)],  # [whether random color, RGB]
        'layer_shadow': [np.pi / 4, 20, .7],  # [theta, shift, opacity]
        'layer_background': None  # RGB, e.g., (100, 100, 100)
    }
    im = text_effects(
        text_arr=text_mask,
        bg_arr=bg_arr,
        min_char_h=min_char_h,
        layer_type_params=layer_type_params,
        is_mix_bg=False
    )
    cv2.imwrite('effects1.png', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

    """ random effects """
    layer_type_params = {
        # [feather, text color, opacity]
        'layer_text': [np.random.choice(2), 'rand', np.random.uniform(.5, 1)],
        'layer_border': [True, None],  # [whether random color, RGB]
        'layer_shadow': [None, None, None],  # [theta, shift, opacity]
        'layer_background': None  # RGB, e.g., (100, 100, 100)
    }
    im = text_effects(
        text_arr=text_mask,
        bg_arr=bg_arr,
        min_char_h=min_char_h,
        layer_type_params=layer_type_params,
        is_mix_bg=False
    )
    cv2.imwrite('effects2.png', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
