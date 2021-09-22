# Copyright 2021 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

from .render import TextRender
import numpy as np
import inspect
from PIL import Image, ImageFilter
import cv2
from collections import OrderedDict
from scipy.misc import imread, imresize
from scipy.ndimage import interpolation
from os.path import isfile


class TextPerturb(TextRender):

    perturb_order = [
        'PT_SCALE',
        'PT_TRANSFORM',
        'PT_ROTATE',
        'PT_COLOR',
        'PT_TEXTURE',
        'PT_FILTER',
        'PT_SHADOW',
        'PT_NOISE',
    ]

    # DO NOT change this list except for adding new method to the end
    # perturbation_type: parameter dimension
    # ALL parameters have the value range from 0 to 1
    _perturb_indices = OrderedDict([
        # [h_ratio, w_ratio, interp_method], keep original size is [.5, .5, .5]
        ('PT_SCALE',        (3, 1)),
        # [(t-l movement), (t-r ...), (b-r ...), (b-l ...)], moving ratio in x-y coord
        ('PT_TRANSFORM',    (4, 2)),
        ('PT_ROTATE',       (1, 1)),    # [angle], projected to 0~360 in degree
        ('PT_COLOR',        (2, 3)),    # [(foreground_RGB), (background_RGB)]
        ('PT_TEXTURE',      (2, 1)),    # not encoded into key
        ('PT_FILTER',       (5, 2)),    # [(filter_type, perform_times), ...]
        ('PT_NOISE',        (2, 1)),    # [noise_type, 1/scale]
        # [min illumination, max illumination, center_x, center_y, exposure]
        ('PT_SHADOW',       (5, 1)),
    ])

    def __init__(self):
        assert all([x in TextPerturb._perturb_indices for x in TextPerturb.perturb_order])
        self.bound_quadrilateral = []  # image coordinate, four points: top-left, top-right, bottom-right, bottom-left
        self.perturb_type_dict = {}
        self.bg_color = None
        self.fg_mask = None
        self.perturb_key = None
        self.bg_lock = False
        for method in inspect.getmembers(self, predicate=inspect.ismethod):
            if '_pt_' in method[0]:
                m = method[0][1:].upper()
                self.perturb_type_dict[m] = method[1]

    def get_perturb_types(self):
        perturb_types = []
        for pt in self.perturb_type_dict:
            perturb_types.append(pt)
            print(pt)
        return perturb_types

    def get_bounding_quadrilateral(self):
        return self.bound_quadrilateral

    def get_perturb_key(self):
        return self.perturb_key

    def set_perturb_key(self, perturb_key):
        self.perturb_key = perturb_key
        return self._key_to_params(perturb_key)

    def plot_bounding_quadrilateral(self, img):
        cv2.drawContours(img, contours=[np.array(self.bound_quadrilateral).astype(np.int32)],
                         contourIdx=0, color=[255 - self.bg_color * 255])
        # pts = self.bound_quadrilateral
        # for i in range(len(pts)):
        #     img = cv2.line(
        #         img,
        #         (int(pts[i][0]), int(pts[i][1])),
        #         (int(pts[(i + 1) % 4][0]), int(pts[(i + 1) % 4][1])),
        #         [255 - self.bg_color * 255] * 3, 1)
        return img.astype(np.uint8)

    def perturb_text_from_render(self, perturb_type_params, text, font, size=100*64, bg=0, interval=1, margin=0):
        """
        perturn rendered text image
        :param perturb_type_params: dict, perturb type and corresponding parameters
        :param text: str, text to be rendered
        :param font: str, dir to the font file
        :param size: int, normalized font size (in points)
        :param bg: 0 or 1, 0-black background, 1-white background
        :param interval: a float greater than .5, shrinking (<1) or increasing (>1) the interval between characters
        :return: perturbed text image
        """

        """ render text image """
        img = self.get_text_image(text=text, font=font, size=size, bg=bg,
                                  interval=interval, margin=margin)
        if img is None:
            return img
        self.fg_mask = None
        self.bg_lock = False
        # h, w = img.shape[:2]
        # self.bound_quadrilateral = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]
        x_min, x_max = self.base_line[0][0], self.base_line[-1][0]
        y = np.array(self.char_box)[..., 1]
        y_min, y_max = np.min(y), np.max(y)
        self.bound_quadrilateral = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        self.bg_color = bg * 255
        self.fg_mask = img if bg == 0 else (255 - img)

        """ perturb text image """
        if len(perturb_type_params) > 0:
            for pt in TextPerturb.perturb_order:
                if pt in perturb_type_params:
                    img = self.perturb_type_dict[pt](img, perturb_type_params[pt])
                    if not self.bg_lock:
                        self.fg_mask = img if bg == 0 else (255 - img)

            # self.perturb_key = self._params_to_key(perturb_type_params)

        return img

    def rand_perturb(self, text, font, size=100*64, bg=0, interval=1):
        # bg = round(np.random.rand())
        # interval = np.random.rand() + .8
        perturb_type_params = {}
        for pt in self._perturb_indices:
            dim = self._perturb_indices[pt]
            param = np.random.rand(dim[0], dim[1])

            """ constraints """
            if 'scale'.upper() in pt:
                param = np.round(param + .5)

            if 'rotate'.upper() in pt:
                param = param * .0

            if 'transform'.upper() in pt:
                param *= .1

            if 'color'.upper() in pt:
                if abs(param[0, 0] - param[1, 0]) + abs(param[0, 1] - param[1, 1]) + abs(param[0, 2] - param[1, 2]) < .6:
                    param[0, :], param[1, :] = 1., 0

            if 'texture'.upper() in pt:
                continue

            if 'filter'.upper() in pt:
                param = np.zeros(dim)
                param[0, 0] = np.random.rand()
                param[0, 1] = .1  # (np.random.rand() + 1) * .1

            if 'noise'.upper() in pt:
                param[-1] = .5

            if 'shadow'.upper() in pt:
                pass

            if dim[1] > 1:
                try:
                    perturb_type_params[pt] = [[float(y) for y in x] for x in param]
                except:
                    pass
            else:
                perturb_type_params[pt] = [float(x) for x in param]

        return self.perturb_text_from_render(perturb_type_params, text, font, size, bg, interval)

    def _key_to_params(self, perturb_key):
        """
        decode pertubation key
        :param perturb_key: str
        :return:
        """
        perturb_type_params = {}
        while perturb_key:
            idx = int(perturb_key[:2])
            perturb_key = perturb_key[2:]
            key = list(TextPerturb._perturb_indices.keys())[idx]
            perturb_type_params[key] = []
            dim = TextPerturb._perturb_indices[key]
            for i in range(dim[0]):
                if dim[1] > 1:
                    tmp = []
                    for j in range(dim[1]):
                        tmp.append(float(perturb_key[:4]) / 1000)
                        perturb_key = perturb_key[4:]
                else:
                    tmp = float(perturb_key[:4]) / 1000
                    perturb_key = perturb_key[4:]
                if isinstance(tmp, list):
                    tmp = tuple(tmp)
                perturb_type_params[key].append(tmp)

        return perturb_type_params

    def _params_to_key(self, perturb_type_params):
        """
        encode perturbation types and params
        :param perturb_type_params: dict
        :return: str
        """
        perturb_key = ''
        for idx, pt in enumerate(TextPerturb._perturb_indices):
            if 'texture'.upper() in pt:
                continue
            if pt in perturb_type_params:
                dim = TextPerturb._perturb_indices[pt]
                param = perturb_type_params[pt]
                key_str = '%02d' % list(TextPerturb._perturb_indices.keys()).index(pt)
                for i in range(dim[0]):
                    if dim[1]:
                        if dim[1] > 1:
                            for j in range(dim[1]):
                                key_str += '%04d' % (round(param[i][j] * 1000.))
                        else:
                            key_str += '%04d' % (round(param[i] * 1000.))
                    else:
                        pass
                perturb_key += key_str
        return perturb_key

    def _pt_scale(self, img, params):
        """
        scale the text image
        :param img: array
        :param params:  list or tuple, [h_ratio, w_ratio, interp_method],
                        h_ratio and w_ratio are float, range [0, 1], projected to [0, max_ratio]
                        interpolation method is float, range [0, 1], projected to [0, len(interp_methods)-1]
        :return: array, dtype=uint8
        """
        max_ratio = 2.
        interp_methods = [
            Image.NEAREST,
            Image.BILINEAR,
            Image.BICUBIC,
            Image.LANCZOS,
            Image.CUBIC
        ]
        assert isinstance(img, np.ndarray) and 2 <= len(img.shape) <= 3
        assert isinstance(params, (list, tuple)) and len(params) == 3
        h_ratio, w_ratio, interp = params[0], params[1], params[2]
        assert 0 <= h_ratio <= 1 and 0 <= w_ratio <= 1 and 0 <= interp <= 1
        h_old, w_old = img.shape[:2]
        h = round(max_ratio * h_ratio * h_old)
        w = round(max_ratio * w_ratio * w_old)
        im = Image.fromarray(img)
        im = im.resize((w, h), resample=interp_methods[int(
            round(interp * (len(interp_methods) - 1)))])

        w, h = im.size[:2]
        old_boarder = np.float32([(0, 0), (w_old-1, 0), (w_old-1, h_old-1), (0, h_old-1)])
        new_boarder = np.float32([(0, 0), (w-1, 0), (w-1, h-1), (0, h-1)])
        M = cv2.getPerspectiveTransform(old_boarder, new_boarder)

        self.char_box = self._perspective_warp_pts_with_M(M, self.char_box)
        self.base_line = self._perspective_warp_pts_with_M(M, self.base_line)
        self.bound_quadrilateral = self._perspective_warp_pts_with_M(M, self.bound_quadrilateral)

        return np.array(im, dtype=np.uint8)

    def _pt_transform(self, img, params):
        """
        perspective transformation
        :param img: array
        :param params: list or tuple, movements of 4 corner points on x and y axis in image coord, value range [0, 1]
        :return: array, dtype=uint8
        """
        assert isinstance(params, (list, tuple)) and len(params) == 4
        assert all([0 <= x <= 1 and 0 <= y <= 1 for x, y in params])

        h, w = img.shape[:2]
        old_boarder = np.float32([(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)])
        new_boarder = np.float32([
            (w * params[0][0], h * params[0][1]),
            (w-1 - w * params[1][0], h * params[1][1]),
            (w-1 - w * params[2][0], h-1 - h * params[2][1]),
            (w * params[3][0], h-1 - h * params[3][1])
        ])

        M = cv2.getPerspectiveTransform(old_boarder, new_boarder)
        img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=self.bg_color, flags=cv2.INTER_CUBIC)

        self.char_box = self._perspective_warp_pts_with_M(M, self.char_box)
        self.base_line = self._perspective_warp_pts_with_M(M, self.base_line)
        self.bound_quadrilateral = self._perspective_warp_pts_with_M(M, self.bound_quadrilateral)

        return img.astype(np.uint8)

    def _pt_rotate(self, img, params):
        """
        rotate text image
        :param img: array
        :param params: float, range [0, 1], projected to [0, 360] in degree
        :return: array, dtype=uint8
        """
        params = params[0]
        assert isinstance(params, (int, float)) and 0 <= params <= 1
        angle_in_degree = params * 360.

        # im = Image.fromarray(img)
        # im = im.rotate(angle_in_degree, Image.BICUBIC, expand=True)
        # return np.array(im, dtype=np.uint8)

        # h, w = img.shape
        # M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_in_degree, scale=1)
        # return cv2.warpAffine(img, M, (w, h*5), borderMode=cv2.BORDER_CONSTANT, borderValue=self.bg_color).astype(np.uint8)

        (h, w) = img.shape[:2]
        (cX, cY) = (w / 2, h / 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle_in_degree, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = (h * sin) + (w * cos)
        nH = (h * cos) + (w * sin)

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        new_bound_quadrilateral = [
            np.matmul(M, [self.bound_quadrilateral[0][0], self.bound_quadrilateral[0][1], 1]),
            np.matmul(M, [self.bound_quadrilateral[1][0], self.bound_quadrilateral[1][1], 1]),
            np.matmul(M, [self.bound_quadrilateral[2][0], self.bound_quadrilateral[2][1], 1]),
            np.matmul(M, [self.bound_quadrilateral[3][0], self.bound_quadrilateral[3][1], 1]),
        ]

        self.char_box = self._perspective_warp_pts(
            self.bound_quadrilateral, new_bound_quadrilateral, self.char_box)
        self.base_line = self._perspective_warp_pts(
            self.bound_quadrilateral, new_bound_quadrilateral, self.base_line)
        self.bound_quadrilateral = new_bound_quadrilateral

        return cv2.warpAffine(img, M, (int(nW), int(nH)), borderMode=cv2.BORDER_CONSTANT, borderValue=self.bg_color, flags=cv2.INTER_CUBIC).astype(np.uint8)

    def _pt_color(self, img, params):
        """
        change foreground and background color
        :param img: 2D or 3D array
        :param params: list or tuple, [(R, G, B), (R, G, B)], value range [0, 1].
        :return: array, dtype=uint8
        """
        assert isinstance(img, np.ndarray) and 2 <= len(img.shape) <= 3
        assert isinstance(self.fg_mask, np.ndarray) and len(self.fg_mask.shape) == 2
        assert isinstance(params, (list, tuple)) and len(params) == 2

        is_gray = len(img.shape) == 2
        fg_color, bg_color = params[0], params[1]
        assert isinstance(fg_color, (list, tuple)) and len(fg_color) == 3
        assert isinstance(bg_color, (list, tuple)) and len(bg_color) == 3
        assert all([0 <= x <= 1 for x in fg_color]) and all([0 <= x <= 1 for x in bg_color])

        if is_gray:
            for item in params:
                if item[0] != item[1] or item[0] != item[2]:
                    img = np.stack([img] * 3, axis=-1)
                    is_gray = False
                    break

        if is_gray:
            if fg_color[0] == bg_color[0]:
                return img
            img_fg = np.ones_like(img) * fg_color[0] * 255
            img_bg = np.ones_like(img) * bg_color[0] * 255
            img = img_fg * self.fg_mask + img_bg * (255 - self.fg_mask)
        else:
            if all([x == y for x, y in zip(fg_color, bg_color)]):
                return img
            h, w = img.shape[:2]
            for i in range(3):
                img_fg = np.ones((h, w)) * fg_color[i] * 255
                img_bg = np.ones((h, w)) * bg_color[i] * 255
                img[..., i] = self.fg_mask / 255. * img_fg + (1 - self.fg_mask / 255.) * img_bg

        self.bg_lock = True

        return img.astype(np.uint8)

    def _pt_texture(self, img, params):
        """
        put texture on text or background
        :param img: 2D or 3D array
        :param params: [array/str/None, array/str/None]
        :return: array
        """
        try:
            assert isinstance(params, (list, tuple)) and len(params) == 2

            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)

            if params[0]:  # text texture
                im_fg = imread(params[0], mode='RGB') if isinstance(params[0], str) else params[0]
                h, w = img.shape[:2]
                im_fg = imresize(im_fg, (h, w), interp='bicubic')
                img = self.fg_mask[:, :, None] / 255. * im_fg + \
                    (1 - self.fg_mask[:, :, None] / 255.) * img

            if params[1]:  # background texture
                im_bg = imread(params[1], mode='RGB') if isinstance(params[1], str) else params[1]
                h, w = img.shape[:2]

                # random crop from background image
                h2, w2 = im_bg.shape[:2]
                top = np.random.choice(max(h2-h, 1))
                left = np.random.choice(max(w2-w, 1))
                im_bg = im_bg[top:min(top+10, h2), left:min(left+10, w2), ...]
                im_bg = imresize(im_bg, (h, w), interp='bicubic')

                # adjust text color for higher contrast
                bg_color_mean = np.mean(im_bg[self.fg_mask > 0, :], axis=0)
                # if img.shape[0] != self.fg_mask.shape[0] or img.shape[1] != self.fg_mask.shape[1]:
                #     print(img.shape, self.fg_mask.shape)
                text_color = img[self.fg_mask > 0, :][0]
                contrast = self._contrast(bg_color_mean, text_color)
                threshold = 100
                if abs(contrast) < threshold:
                    # print('Text color is adjusted to make it readable on the background image!')
                    # print(contrast)
                    # text_color = text_color - text_color * (threshold - contrast) / threshold
                    text_color = text_color - np.sign(contrast) * (threshold - abs(contrast))
                    img = self.fg_mask[:, :, None] / np.max(self.fg_mask).astype(np.float32) * np.ones_like(
                        img) * np.clip(text_color, 0, 255)[None, None, :]  # + (1 - self.fg_mask[:, :, None] / 255.) * img

                img = self.fg_mask[:, :, None] / np.max(self.fg_mask).astype(np.float32) * img + (
                    1 - self.fg_mask[:, :, None] / np.max(self.fg_mask).astype(np.float32)) * im_bg
                # img = img + (1 - self.fg_mask[:, :, None] / 255.) * im_bg

            self.bg_lock = True
            return img.astype(np.uint8)
        except:
            import traceback
            traceback.print_exc()
            print(img.shape, self.fg_mask.shape)
            print(img[self.fg_mask >= 255, :])
            print(np.max(self.fg_mask), np.min(self.fg_mask))
            # cv2.imwrite('%d.png' % np.random.randint(0, 100), self.fg_mask.astype(np.uint8))
            exit()

    def _pt_filter(self, img, params):
        """
        filter text image
        :param img: array
        :param params: list or tuple, [(filter_idx, perform_times), ...], value range [0, 1],
        :return: array, dtype=uint8
        """
        filters = [
            ImageFilter.BLUR,
            ImageFilter.CONTOUR,
            # ImageFilter.EMBOSS,
            # ImageFilter.FIND_EDGES,
            ImageFilter.SMOOTH
        ]
        n_filters = len(filters)
        assert isinstance(params, (list, tuple))
        im = Image.fromarray(img)
        for item in params:
            assert isinstance(item, (list, tuple)) and len(item) == 2
            assert 0 <= item[0] <= 1 and 0 <= item[1] <= 1
            idx = int(round(item[0] * (n_filters - 1)))
            times = int(round(item[1] * 10))
            for _ in range(times):
                im = im.filter(filters[idx])

        self.bg_lock = True

        return np.array(im, dtype=np.uint8)

    def _pt_noise(self, img, params):
        """
        add noise to text image: Gaussian, uniform, or salt pepper.
        :param img: array
        :param params: list or tuple, [noise_type, std], value range [0, 1]
        :return: array, dtype=uint8
        """
        assert isinstance(params, (list, tuple)) and len(params) == 2
        assert 0 <= params[0] <= 1 and 0 <= params[1] <= 1

        def noise_normal(param):
            return np.random.normal(scale=param, size=img.shape).astype(np.float32)

        def noise_uniform(param):
            return np.random.uniform(low=-param, high=param, size=img.shape).astype(np.float32)

        def noise_salt_pepper(param):
            param = round(param)
            min_param = 10
            tmp = np.random.randint(max(param, min_param), size=img.shape).astype(np.float32)
            tmp[tmp == 0] = -255
            tmp[tmp == (param - 1)] = 255
            return tmp

        noise_types = [
            noise_normal,
            noise_uniform,
            noise_salt_pepper
        ]
        if params[1] == 0:
            params[1] = 1
        idx_noise = int(round(params[0] * (len(noise_types) - 1)))
        im_noise = noise_types[idx_noise](1. / params[1])

        self.bg_lock = True

        return np.clip(img + im_noise, 0, 255).astype(np.uint8)

    def _pt_shadow(self, img, params):
        """
        add shadow to text image
        :param img: array
        :param params: list or tuple, [lowest_illumination, highest_illumination, center_x, center_y, exposure],
                        value range [0, 1]
        :return: array, dtype=uint8
        """

        assert isinstance(params, (list, tuple)) and len(params) == 5
        assert all([0 <= x <= 1 for x in params])
        h, w = img.shape[:2]
        low, high = params[0], params[1] + params[4]
        if low > high:
            low, high = high, low
        # center_x, center_y = w * params[2], h * params[3]
        if len(img.shape) == 2:
            c = 1
        else:
            c = img.shape[2]

        """ character shadow """
        shadow = 255 - self.fg_mask
        shadow = cv2.GaussianBlur(shadow.astype(np.uint8), (15, 15), 0)
        shift, theta = min(h, w) * .2 * np.random.rand(), np.pi * np.random.rand() * 2
        dx, dy = shift * np.array([-np.sin(theta), np.cos(theta)])
        shadow = interpolation.shift(shadow, shift=[dx, dy], mode='constant', cval=255) / 255.
        if c > 1:
            shadow = np.stack([shadow] * c, axis=-1)
        img_shadow = img * shadow
        img = self.fg_mask[:, :, None] / 255. * img + \
            (1 - self.fg_mask[:, :, None] / 255.) * img_shadow

        # """ a simple version: light from top, right, bottom, or left """
        # center_x, center_y = round(params[2]), round(params[3])
        # if center_x == 0 and center_y == 0:  # from top
        #     shadow = np.linspace(high, low, h)
        #     shadow = np.stack([shadow] * w, axis=-1)
        # elif center_x == 1 and center_y == 0:  # from right
        #     shadow = np.linspace(low, high, w)
        #     shadow = np.stack([shadow] * h, axis=0)
        # elif center_x == 1 and center_y == 1:  # from bottom
        #     shadow = np.linspace(low, high, h)
        #     shadow = np.stack([shadow] * w, axis=-1)
        # elif center_x == 0 and center_y == 1:  # from left
        #     shadow = np.linspace(high, low, w)
        #     shadow = np.stack([shadow] * h, axis=0)
        #
        #
        #
        # if c == 1:
        #     img = img * shadow
        # else:
        #     img = img * np.stack([shadow] * c, axis=-1)

        self.bg_lock = True

        return img.clip(0, 255).astype(np.uint8)

    def _perspective_warp_pts(self, pts_src, pts_dst, pts):
        """
        warp character box and baseline
        :param pts_src: original quadrilateral bounding box
        :param pts_des: new quadrilateral bounding box
        :param pts: points need to be warped
        :return array of warped points
        """
        M = cv2.getPerspectiveTransform(src=np.float32(pts_src), dst=np.float32(pts_dst))

        pts = np.array(pts)
        pts_shape = pts.shape
        pts = pts.reshape((-1, 2))

        tmp = np.concatenate([np.array(pts), np.ones((pts.shape[0], 1))], axis=-1)
        res = M.dot(tmp.T)

        x = np.round(res[0, :] / (res[2, :] + 1e-16))
        y = np.round(res[1, :] / (res[2, :] + 1e-16))

        pts = np.stack([x, y], axis=1)

        pts = pts.reshape(pts_shape)

        return pts.astype(np.int32)

    def _perspective_warp_pts_with_M(self, M, pts):
        """
        warp character box and baseline
        :param M: transform matrix
        :param pts: points need to be warped
        :return array of warped points
        """
        pts = np.array(pts)
        pts_shape = pts.shape
        pts = pts.reshape((-1, 2))

        tmp = np.concatenate([np.array(pts), np.ones((pts.shape[0], 1))], axis=-1)
        res = M.dot(tmp.T)

        x = np.round(res[0, :] / (res[2, :] + 1e-16))
        y = np.round(res[1, :] / (res[2, :] + 1e-16))

        pts = np.stack([x, y], axis=1)

        pts = pts.reshape(pts_shape)

        return pts.astype(np.int32)

    def _contrast(self, color1, color2):
        """
        calculate contrast between two colors
        :param color1: [R, G, B], range [0, 255]
        :param color2: [R, G, B], range [0, 255]
        :return: float, differnece between the luminance of the two colors
        """

        def luminance(color):
            param = [0.2126, 0.7152, 0.0722]
            RGB = [c / 3294. if c <= 10 else (c / 269 + 0.0513) ** 2.4 for c in color]
            return np.matmul(param, RGB)

        def gray(color):
            # 0.2989 * R + 0.5870 * G + 0.1140 * B
            param = [0.2989, 0.5870, 0.1140]
            return np.matmul(param, color)

        # return luminance(color1) - luminance(color2)

        return gray(color1) - gray(color2)


if __name__ == '__main__':
    from time import time

    text = 'The'
    font = '/home/zzhang/Documents/TextFlow/text/data/font/Vera.ttf'
    fg_fn = '/home/zzhang/Documents/font_render_zhaowen/tensorflow/fontnet-script/save/a.png'
    bg_fn = '/home/zzhang/Documents/font_render_zhaowen/tensorflow/fontnet-script/save/bg2.jpg'
    perturb = TextPerturb()

    # perturb_type_params = {
    #     'PT_COLOR': [(.1, .1, .1), (1, 1, 1)],  # [fg_color, bg_color] = [(R, G, B), (R, G, B)]
    #     'PT_FILTER': [(0, .1), (.25, .0), (.5, .0), (.75, .0), (1., .0)],  # Gaussian blur with radius
    #     'PT_NOISE': [0.5, .01],  # [noise_type, 1/scale]
    #     'PT_SCALE': [.8, .8, 0.5],  # [h_ratio, w_ratio, interp_method]
    #     'PT_TRANSFORM': [(0.1, 0.1), (0.1, 0.1), (0., 0.1), (0., 0.)],  # for corners, [top-left, t-r, b-r, b-l]
    #     'PT_ROTATE': [.01],  # 0~1, projected to 0~360 degree
    #     # 'PT_SHADOW': [.1, 1, .0, 1.0, .0],  # illumination range, light direction, and exposure
    # }
    param = np.random.rand(2, 3)
    if abs(param[0, 0] - param[1, 0]) + abs(param[0, 1] - param[1, 1]) + abs(param[0, 2] - param[1, 2]) < .6:
        param[0, :], param[1, :] = 1., 0
    rand_params = {
        # for corners, [top-left, t-r, b-r, b-l]
        'PT_TRANSFORM': [np.random.rand(2) * .1 for _ in range(4)],
        # 'PT_ROTATE': [(np.random.rand() / 16) if (np.random.rand() >= .5) else (1 - np.random.rand() / 16)],
        # 0~1, projected to 0~360 degree
        #     'PT_COLOR': [[float(y) for y in x] for x in param],  # [fg_color, bg_color] = [(R, G, B), (R, G, B)]
        #     'PT_TEXTURE': [None, bg_fn],
        # 'PT_NOISE': [0, np.random.rand() * .1 + .05],  # [noise_type, 1/scale]
        # 'PT_SHADOW': [np.random.rand() * .5 + .5, 1, np.random.rand(), np.random.rand(), 0],
        # illumination range, light direction, and exposure
    }

    t_start = time()
    for _ in range(100):
        im = perturb.perturb_text_from_render(
            perturb_type_params=rand_params,
            text=text, font=font, size=200 * 64, bg=0, interval=1)
    print(time() - t_start)

    Image.fromarray(im.astype(np.uint8)).save(
        '/home/zzhang/Documents/font_render_zhaowen/tensorflow/fontnet-script/save/tmp2.png')

    exit()

    """ font file """
    font_file = './data/font/Vera.ttf'

    """ text to render """
    text = 'Deep Learning Group @ Adobe Research'

    """ object of TextPerturb """
    perturb = TextPerturb()

    """ print supported perturbations """
    perturb.get_perturb_types()

    """ paramter setting """
    perturb_type_params = {
        'PT_COLOR': [(.1, .1, .1), (1, 1, 1)],  # [fg_color, bg_color] = [(R, G, B), (R, G, B)]
        # Gaussian blur with radius
        'PT_FILTER': [(0, .1), (.25, .0), (.5, .0), (.75, .0), (1., .0)],
        'PT_NOISE': [0.5, .01],  # [noise_type, 1/scale]
        'PT_SCALE': [.8, .8, 0.5],  # [h_ratio, w_ratio, interp_method]
        # for corners, [top-left, t-r, b-r, b-l]
        'PT_TRANSFORM': [(0.1, 0.1), (0.1, 0.1), (0., 0.1), (0., 0.)],
        'PT_ROTATE': [.01],  # 0~1, projected to 0~360 degree
        'PT_SHADOW': [.1, 1, .0, 1.0, .0],  # illumination range, light direction, and exposure
    }

    """ render perturbed text image """
    im = perturb.perturb_text_from_render(
        perturb_type_params=perturb_type_params,
        text=text, font=font_file, size=100 * 96, bg=0, interval=.8)

    """ plot bounding quadrilateral """
    im = perturb.plot_bounding_quadrilateral(im)
    # print(perturb.bound_quadrilateral)
    base_line = perturb.base_line
    cv2.line(im, tuple(base_line[0, :]), tuple(base_line[1, :]), color=255, thickness=1)
    char_box = perturb.char_box
    cv2.drawContours(im, contours=char_box, contourIdx=-1, color=255, thickness=1)

    """ get key/code corresponding to a perturbation process"""
    perturb_key = perturb.get_perturb_key()
    # print(perturb_key)

    """ set parameters by key """
    perturb_param = perturb.set_perturb_key(perturb_key)
    # for key in sorted(perturb_param):
    #     print(key, perturb_param[key])

    """ save image """
    im = Image.fromarray(im)
    im.save('./perturb1.png')

    """ render perturbed text image from the parameter parsed from key """
    im = perturb.perturb_text_from_render(
        perturb_type_params=perturb_param,
        text=text, font=font_file, bg=0, interval=.75)

    """ plot bounding quadrilateral """
    im = perturb.plot_bounding_quadrilateral(im)
    base_line = perturb.base_line
    cv2.line(im, tuple(base_line[0, :]), tuple(base_line[1, :]), color=255, thickness=1)
    char_box = perturb.char_box
    cv2.drawContours(im, contours=char_box, contourIdx=-1, color=255, thickness=1)

    """ print corresponding perturbation key """
    # print(perturb.get_perturb_key())

    """ save image """
    im = Image.fromarray(im)
    im.save('./perturb2.png')

    """ randomly render perturbed text image """
    im = perturb.rand_perturb(text=text, font=font_file)
    im = perturb.plot_bounding_quadrilateral(im)

    base_line = perturb.base_line
    cv2.line(im, tuple(base_line[0, :]), tuple(base_line[1, :]), color=255, thickness=1)
    char_box = perturb.char_box
    cv2.drawContours(im, contours=char_box, contourIdx=-1, color=255, thickness=1)

    im = Image.fromarray(im)
    im.save('./perturb3.png')
