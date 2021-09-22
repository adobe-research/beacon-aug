# Copyright 2021 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

from .perturb import TextPerturb
import numpy as np
import cv2
from skimage import segmentation
from skimage.future import graph
from .effects import TextEffects
import inspect
import logging


class TextSynthesis(TextPerturb):
    def __init__(self, color_model_path=None):
        self.image = None
        self.image_path = None
        self.text_mask = None
        self.list_text = []
        self.list_bound = []
        self.list_char_box = []
        self.list_base_line = []
        self.effects = TextEffects(color_model_path=color_model_path)

    def _set_image(self, img):
        """
        set the image to place text on
        :param img: array or str (file dir)
        """
        assert isinstance(img, (np.ndarray, str))
        if isinstance(img, np.ndarray):
            assert img.ndim == 3  # RGB
            self.image = img
        elif isinstance(img, str):
            self.image = cv2.imread(img, cv2.IMREAD_COLOR)
            if self.image is None:
                raise Exception('''Failed reading the image file "%s"''' % img)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.image_path = img

        self.image = self.image.astype(np.uint8)
        self.text_mask = np.zeros(self.image.shape[:2])

    def _pt_transform(self, src_size, dst_size, pts):
        pts = np.array(pts)
        pts_shape = pts.shape
        pts = pts.reshape((-1, 2))
        pts[:, 0] = pts[:, 0] * dst_size[1] / src_size[1]
        pts[:, 1] = pts[:, 1] * dst_size[0] / src_size[0]
        pts = pts.reshape(pts_shape)
        return pts.astype(np.int32)

    def _get_text_regions(self, num_regions=5, is_top_left_origin=True, ensure_num_regions=False, check_collision=True):
        """
        get smooth quadrilateral regions for placing text
        :param n_regions: int, number of regions required, 1 to 10
        :param is_top_left_origin: whether keep the first point in a region vector as the top-left corner
        :param ensure_num_regions: sometimes, there are not enough suitable regions for placing text
            True: force to get num_regions regions
            False: only return suitable regions, may less than num_regions
        :param check_collision: bool, whether check collision of regions
        :return: list, [4x2 array, ...], each item denotes four corners of a quadrilateral region
        """

        def top_left_origin(pts):
            dist, pts_new_order = [], []
            n_pts = len(pts)
            for pt in pts:
                dist.append(np.linalg.norm(pt))
            idx_top_left = int(np.argmin(dist))
            for idx in range(idx_top_left, idx_top_left + n_pts):
                pts_new_order.append(pts[idx % n_pts])
            return np.array(pts_new_order)

        # assert isinstance(num_regions, (int, float)) and 1 <= num_regions <= 30
        num_regions = round(num_regions)

        """ constraints """
        min_seg_img_area_ratio = .01
        min_seg_bbox_area_ratio = .6

        """ segmentation """
        h, w = self.image.shape[:2]
        n_segments = min(max(1, int(h * w / 128. ** 2)), 10)
        segments, labels, areas = TextSynthesis.image_segmentation(
            self.image, n_segments=n_segments)

        """ get smooth regions """
        # TODO: region_filter

        """ define quadrilateral for text """
        indices = np.arange(len(labels)).astype(np.int32)
        np.random.shuffle(indices)
        text_regions = []

        h, w = self.image.shape[:2]
        collision_mask = np.zeros((h, w))
        trash_regions = []
        cnt = 0
        for idx in indices:
            if cnt >= num_regions:
                break
            if float(areas[idx]) / (h * w) < min_seg_img_area_ratio:
                continue

            """ min bounding rectangle """
            ys, xs = np.where(segments == labels[idx])
            coords = np.c_[xs, ys].astype('float32')
            # hull = cv2.convexHull(coords, clockwise=False, returnPoints=True)
            # hull = np.array(hull).squeeze()
            rect = cv2.minAreaRect(coords)
            if float(areas[idx]) / (rect[1][0] * rect[1][1]) < min_seg_bbox_area_ratio:
                continue
            box = np.array(cv2.boxPoints(rect))

            """ shrink the rectangle  """
            # mask_center = np.array([np.mean(xs), np.mean(ys)])
            # shift = np.abs(mask_center - np.array(rect[0]))
            # rect = (tuple(mask_center), tuple((np.array(rect[1]) - shift)), rect[-1])

            """ perspective transformation according to depth info """
            # TODO:

            """ fit inside of the image """
            box[box < 0] = 0
            box[box[:, 0] > w - 1, 0] = w - 1
            box[box[:, 1] > h - 1, 1] = h - 1

            """ check collision """
            if check_collision:
                mask = cv2.drawContours(
                    np.zeros((h, w)), [box.astype(np.int32)], 0, 1, thickness=cv2.FILLED)
                if np.sum(mask * collision_mask):
                    # shrink
                    continue
                else:
                    collision_mask += mask
            else:
                collision_mask = cv2.drawContours(
                    collision_mask, [box.astype(np.int32)], 0, 1, thickness=cv2.FILLED)

            """ arrange the corners to keep the first corner is the top-left corner """
            if is_top_left_origin:
                box = top_left_origin(box)

            if rect[1][0] * rect[1][1] / float(h * w) < min_seg_img_area_ratio:
                trash_regions.append([box, rect[1][0] * rect[1][1]])
                continue

            text_regions.append(box)
            cnt += 1

        if cnt < num_regions and ensure_num_regions:
            trash_regions = sorted(trash_regions, key=lambda x: x[-1], reverse=True)
            for r in trash_regions:
                text_regions.append(r[0])
                cnt += 1
                if cnt >= num_regions:
                    break

        return text_regions

    def _place_text_on_region(self, text, font, region, effects=None, is_keep_ratio=True):
        """
        place text to self.image
        :param region: 4x2 array
        :param text: str
        :param font: str
        :param effects: dict, text effect types and parameters, all items are optional
            'layer_text'      : [feather (True/False), text color (RGB/None/'rand'), opacity (.75~1)]
            'layer_border'    : [is_rand_color (True/False), color (RGB/None)]
            'layer_shadow'    : [theta (float/None), shift (float/None), opacity (.75~1/None)]
            'layer_background': color (RGB), default None (original background)
            'mix_bg'    : whether mix background
            'text_interval'  : text interval
            'text_size'      : text size
        :param is_keep_ratio: whether keep the ratio of rendered text
        """

        """ map region patch to rectangle for placing text """
        w, h = int(max(region[:, 0]) - min(region[:, 0])
                   ), int(max(region[:, 1]) - min(region[:, 1]))
        region_rect = np.float32([
            (0, 0),
            (w, 0),
            (w, h),
            (0, h)
        ])
        region = np.float32(region)
        M = cv2.getPerspectiveTransform(region, region_rect)
        region_rect_rgb = cv2.warpPerspective(self.image, M, (w, h)).astype(np.uint8)

        """ render text """
        # if effects is not None and 'text_size' in effects:
        #     size = effects['text_size'] if effects['text_size'] is not None else (h * 96)
        # else:
        #     size = h * 96
        size = 96 * 256

        if effects is not None and 'text_interval' in effects:
            interval = effects['text_interval'] if effects['text_interval'] is not None else 1
        else:
            interval = 1

        text_arr = self.perturb_text_from_render(
            perturb_type_params={}, text=text, font=font, size=size, bg=0, interval=interval)

        if text_arr is None:
            return False
        txt_h, txt_w = text_arr.shape[:2]
        if txt_h == 0 or txt_w == 0:
            return False
        text_mask = self.fg_mask

        """ fit rendered text to region """
        img_h, img_w = region_rect_rgb.shape[:2]
        r_h, r_w = float(img_h) / txt_h, float(img_w) / txt_w
        if is_keep_ratio:
            r_w = min(r_h, r_w)
            r_h = r_w if r_h / r_w < 10 else (r_w * 2)
            text_arr = cv2.resize(text_arr, (int(txt_w * r_w), int(txt_h * r_h)))
            text_mask = cv2.resize(text_mask, (int(txt_w * r_w), int(txt_h * r_h)))
            # text_arr = self._pt_scale(text_arr, [.5 * r_h, .5 * r_w, 0])
        else:
            text_arr = cv2.resize(text_arr, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            text_mask = cv2.resize(text_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            # text_arr = self._pt_scale(text_arr, [.5 * r_h, .5 * r_w, 0])
        txt_h_new, txt_w_new = text_arr.shape[:2]
        self.char_box = self._pt_transform((txt_h, txt_w), (txt_h_new, txt_w_new), self.char_box)
        self.base_line = self._pt_transform((txt_h, txt_w), (txt_h_new, txt_w_new), self.base_line)
        self.bound_quadrilateral = self._pt_transform(
            (txt_h, txt_w), (txt_h_new, txt_w_new), self.bound_quadrilateral)

        """ offset bounding quadrilateral, character box,  and baseline """
        txt_h, txt_w = text_arr.shape[:2]
        delta_w = img_w - txt_w
        delta_h = img_h - txt_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        text_arr = cv2.copyMakeBorder(text_arr, top=top, bottom=bottom, left=left, right=right,
                                      borderType=cv2.BORDER_CONSTANT, value=0)
        text_mask = cv2.copyMakeBorder(text_mask, top=top, bottom=bottom, left=left, right=right,
                                       borderType=cv2.BORDER_CONSTANT, value=0)
        self.bound_quadrilateral = np.array(self.bound_quadrilateral) + np.array([left, top])
        self.char_box = np.array(self.char_box) + np.array([left, top])
        self.base_line = np.array(self.base_line) + np.array([left, top])

        """ blend text and region """
        min_char_h = min([np.linalg.norm(d)
                         for d in self.char_box[:, 0, :] - self.char_box[:, -1, :]])
        if effects is not None and 'mix_bg' in effects:
            is_mix_bg = effects['mix_bg']
        else:
            is_mix_bg = False
        patch_blend = self.effects(text_arr=text_arr, bg_arr=region_rect_rgb, min_char_h=min_char_h,
                                   layer_type_params=effects, is_mix_bg=is_mix_bg)

        """ map blended patch back to image """
        M = cv2.getPerspectiveTransform(region_rect, region)
        blend_region = cv2.warpPerspective(patch_blend, M, self.image.shape[:2][::-1],
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        blend_mask = cv2.warpPerspective(np.zeros(patch_blend.shape[:2]), M, self.image.shape[:2][::-1],
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=1)
        text_mask = cv2.warpPerspective(text_mask, M, self.image.shape[:2][::-1],
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        self.image = (self.image * blend_mask[:, :, None] + blend_region).astype(np.uint8)
        self.text_mask = np.clip(self.text_mask + text_mask, 0, 255)

        """ map bounding quadrilateral and baseline back to image """
        self.bound_quadrilateral = self._perspective_warp_pts_with_M(
            M, self.bound_quadrilateral).astype(np.int32)
        self.char_box = self._perspective_warp_pts_with_M(M, self.char_box).astype(np.int32)
        self.base_line = self._perspective_warp_pts_with_M(M, self.base_line).astype(np.int32)

        return True

    def _place_text_on_image(self, text_list, font_list, region_list=None, effect_list=None,
                             is_top_left_origin=True, ensure_num_regions=True, is_keep_ratio=True):
        """
        place multiple text on a image
        :param text_list: list of str
        :param font_list: list of font file path
        :param region_list: list of regions, a region is a list of 4 corners in x-y coord
        :param effect_list: list of effects, a effect is a dict, please refer to self._place_text_on_region()
        :param is_top_left_origin: whether keep the first point as the top-left corner in a region vector
        :param ensure_num_regions: whether ensure the number of text placed on the image
        :param is_keep_ratio: whether keep the ratio of rendered text
        :return: RGB image with text on it, and annotations, i.e., bounding box, character box, baseline, and true text
        """

        n_text, n_font = len(text_list), len(font_list)
        # assert 1 <= n_text <= 10
        self.list_text = []
        self.list_bound = []
        self.list_char_box = []
        self.list_base_line = []

        """ get regions """
        if region_list is None:
            region_list = self._get_text_regions(
                num_regions=n_text, is_top_left_origin=is_top_left_origin, ensure_num_regions=ensure_num_regions)
            """ match text length to regains """
            sorted_idx_region = np.argsort([max(r[:, 0]) - min(r[:, 0]) for r in region_list])
            sorted_idx_text = np.argsort([len(t) for t in text_list])
        else:
            if not len(region_list) == n_text:
                logging.info('Mismatched length between region_list and text_list')
                return None
            """ match text length to regains """
            sorted_idx_region = np.arange(n_text)
            sorted_idx_text = np.arange(n_text)

        """ place text on each region """
        if effect_list is not None:
            if not len(effect_list) == n_text:
                logging.info('Mismatched length between effect_list and text_list')
                return None
        for idx in range(len(region_list)):
            if self._place_text_on_region(
                    text=text_list[sorted_idx_text[idx]],
                    font=font_list[idx % n_font],  # font[np.random.randint(len(font))],
                    region=np.array(region_list[sorted_idx_region[idx]]),
                    effects=effect_list[sorted_idx_text[idx]] if effect_list is not None else None,
                    is_keep_ratio=is_keep_ratio):
                self.list_text.append(self.text)
                self.list_bound.append(self.bound_quadrilateral)
                self.list_char_box.append(self.char_box)
                self.list_base_line.append(self.base_line)

        """ prepare return data """
        return {
            'image': self.image.astype(np.uint8),  # synthesize image with text
            'text': self.list_text,  # list of text placed on the image
            'wbb': self.list_bound,  # list of bounding quadrilateral
            'cbb': self.list_char_box,  # list of character-wise bounding quadrilateral
            'base': self.list_base_line,  # list of baseline
            'mask': self.text_mask
        }

    def plot_annotations(self):
        """
        plot bounding quadrilateral of the text placed on self.image, as well as their baseline and character-wise bbox
        """
        output = np.copy(self.image)
        cv2.drawContours(output, contours=np.array(self.list_bound),
                         contourIdx=-1, color=(255, 0, 0), thickness=1)
        for i in range(len(self.list_bound)):
            cv2.drawContours(
                output, contours=self.list_char_box[i], contourIdx=-1, color=(255, 255, 0), thickness=1)
            cv2.line(output, tuple(self.list_base_line[i][0]), tuple(
                self.list_base_line[i][1]), color=(0, 0, 255))
            cv2.putText(img=output, text=self.list_text[i],
                        org=tuple([np.min(self.list_bound[i][:, 0]), max(
                            10, np.min(self.list_bound[i][:, 1]) - 5)]),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=(255, 0, 255), thickness=1)
        return output

    @staticmethod
    def image_segmentation(img, n_segments=3):
        """
        segmentation algorithm
        :param img: 3D array
        :return: segments (2D array), lables (1D array), area of each segment (1D array).
        """
        segs = segmentation.slic(img, n_segments=n_segments, compactness=30)
        # g = graph.rag_mean_color(img, segs, mode='similarity')
        # segs = graph.cut_normalized(segs, g)
        labels = np.unique(segs)
        areas = []
        for l in labels:
            areas.append(np.sum(segs == l))

        # cv2.imwrite('segs.png', segs)
        return segs, np.array(labels), np.array(areas)

    def __call__(self, img, text_list, font_list, region_list=None, effect_list=None,
                 is_top_left_origin=True, ensure_num_regions=True, is_keep_ratio=True):
        """
        place multiple text on a image
        :param text_list: list of str
        :param font_list: list of font file path
        :param region_list: list of regions, a region is a list of 4 corners in x-y coord
        :param effect_list: list of effects, a effect is a dict, please refer to self._place_text_on_region()
        :param is_top_left_origin: whether keep the first point as the top-left corner in a region vector
        :param ensure_num_regions: whether ensure the number of text placed on the image
        :param is_keep_ratio: whether keep the ratio of rendered text
        :return: RGB image with text on it, and annotations, i.e., bounding box, character box, baseline, and true text
        """
        self._set_image(img)
        assert isinstance(text_list, (list, tuple))
        assert isinstance(font_list, (list, tuple))
        assert isinstance(region_list, (list, tuple)) or region_list is None
        assert isinstance(effect_list, (list, tuple)) or effect_list is None

        return self._place_text_on_image(
            text_list=text_list,
            font_list=font_list,
            region_list=region_list,
            effect_list=effect_list,
            is_top_left_origin=is_top_left_origin,
            ensure_num_regions=ensure_num_regions,
            is_keep_ratio=is_keep_ratio

        )


class ImageProcess(object):
    def __init__(self):
        self.methods = dict()
        self.image = None
        for method in inspect.getmembers(ImageProcess, predicate=inspect.isfunction):
            method_name = method[0].lower()
            if '__' not in method_name:
                self.methods[method_name] = method[1]

    def __call__(self, image):
        assert isinstance(img, (np.ndarray, str))
        if isinstance(img, np.ndarray):
            assert img.ndim == 3  # RGB
            self.image = img
        elif isinstance(img, str):
            self.image = cv2.imread(img, cv2.IMREAD_COLOR)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            if self.image is None:
                raise Exception('''Failed reading the image file "%s"''' % img)
        self.image = self.image.astype(np.uint8)

        """ performing order and enable of the functions """
        n_func = len(self.methods)
        order = np.arange(n_func)
        np.random.shuffle(order)
        enable = np.random.randint(2, size=n_func)
        method_names = self.img_process.keys()
        for idx in order:
            if enable[idx]:
                self.image = self.methods[method_names[idx]](self.image)

    @staticmethod
    def add_noise(image, noise_typ='gauss', param=None):
        """
        add noise to image
        :param image: array
        :param noise_typ: 'gauss', 's&p', 'speckle'
        :param param: float, vary with different noise
        :return:
        """
        if noise_typ == "gauss":
            if param is None:
                param = 10
            row, col, ch = image.shape
            mean = 0
            sigma = param
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = np.clip(image + gauss, 0, 255)
            return noisy.astype(np.uint8)
        elif noise_typ == "s&p":
            if param is None:
                param = .1
            s_vs_p = 0.5
            amount = param
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
            out[tuple(coords)] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
            out[tuple(coords)] = 0
            return out.astype(np.uint8)
        elif noise_typ == "speckle":
            if param is None:
                param = .1
            row, col, ch = image.shape
            gauss = np.random.randn(row, col, ch) * param
            gauss = gauss.reshape(row, col, ch)
            noisy = np.clip(image + image * gauss, 0, 255)
            return noisy.astype(np.uint8)
        else:
            raise Exception('Unrecognized noise_type!')

    @staticmethod
    def increase_contrast(img, param=2.0):

        # -----Converting image to LAB Color model-----------------------------------
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        # -----Splitting the LAB image to different channels-------------------------
        l, a, b = cv2.split(lab)

        # -----Applying CLAHE to L-channel-------------------------------------------
        clahe = cv2.createCLAHE(clipLimit=param, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
        limg = cv2.merge((cl, a, b))

        # -----Converting image from LAB Color model to RGB model--------------------
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return img.astype(np.uint8)

    @staticmethod
    def invert_color(img):
        return 255 - img

    @staticmethod
    def add_blur(img, blur_type='gauss', param=5):
        """
        blur image
        :param img: array
        :param blur_type: 'gauss', 'average', 'median'
        :param param: float
        :return:
        """
        if blur_type == 'gauss':
            img = cv2.GaussianBlur(img, (param, param), 0)

        elif blur_type == 'average':
            img = cv2.blur(img, (param, param))
        elif blur_type == 'median':
            img = cv2.medianBlur(img, param)
        else:
            raise Exception('Unrecognized blur_type!')

        return img.astype(np.uint8)


if __name__ == '__main__':
    img_dir = './data/image/input.png'
    text_list = ['Adobe', 'Group!']
    font_list = ['./data/font/Vera.ttf', './data/font/VeraMono.ttf']
    txt_synth = TextSynthesis()

    """ synthesis without effects """
    output = txt_synth(
        img=img_dir,
        text_list=text_list,
        font_list=font_list,
        region_list=None,
        effect_list=None,
        is_top_left_origin=True,
        ensure_num_regions=True,
        is_keep_ratio=True,
    )
    output_annotation = txt_synth.plot_annotations()
    cv2.imwrite('synthesis_no_effects.png', cv2.cvtColor(
        output_annotation.astype(np.uint8), cv2.COLOR_RGB2BGR))

    """ specific synthesis """
    region_list = [
        [(50, 50), (500, 80), (280, 400), (60, 320)],
        [(200, 300), (300, 350), (320, 400), (150, 350)]
    ]
    effect_list = [
        {
            'layer_text': [True, (200, 100, 50), .75],  # [feather, text color, opacity]
            'layer_border': [False, (50, 200, 100)],  # [whether random color, RGB]
            'layer_shadow': [np.pi / 4, 20, .7],  # [theta, shift, opacity]
            'layer_background': None,  # RGB, e.g., (100, 100, 100)
            'text_size': 96 * 400,
            'text_interval': 1.6,
            'mix_bg': True
        },
        {
            'layer_text': [True, None, .85],  # [feather, text color, opacity]
            'layer_border': [False, (50, 200, 100)],  # [whether random color, RGB]
            'layer_shadow': [np.pi / 3, 20, .7],  # [theta, shift, opacity]
            'layer_background': (100, 50, 200),  # RGB, e.g., (100, 100, 100)
            'text_size': 96 * 50,
            'text_interval': .8,
            'mix_bg': False
        }
    ]

    output = txt_synth(
        img=img_dir,
        text_list=text_list,
        font_list=font_list,
        region_list=region_list,
        effect_list=effect_list,
        is_top_left_origin=True,
        ensure_num_regions=True,
        is_keep_ratio=True,
    )
    output_annotation = txt_synth.plot_annotations()
    cv2.imwrite('synthesis_target_effects.png', cv2.cvtColor(
        output_annotation.astype(np.uint8), cv2.COLOR_RGB2BGR))

    """ random synthesis """
    text_list = ['Adobe', 'Group!']
    font_list = ['./data/font/Vera.ttf', './data/font/VeraMono.ttf']
    region_list = None
    effect_list = [
        {
            # [feather, text color, opacity]
            'layer_text': [np.random.choice(2), 'rand', np.random.uniform(.5, 1)],
            'layer_border': [True, None],  # [whether random color, RGB]
            'layer_shadow': [None, None, None],  # [theta, shift, opacity]
            'layer_background': None,  # RGB, e.g., (100, 100, 100)
            'text_size': None,
            'text_interval': np.random.uniform(.8, 1.4),
            'mix_bg': np.random.choice(2)
        },
        {
            # [feather, text color, opacity]
            'layer_text': [np.random.choice(2), 'rand', np.random.uniform(.5, 1)],
            'layer_border': [True, None],  # [whether random color, RGB]
            'layer_shadow': [None, None, None],  # [theta, shift, opacity]
            'layer_background': None,  # RGB, e.g., (100, 100, 100)
            'text_size': None,
            'text_interval': np.random.uniform(.8, 1.4),
            'mix_bg': np.random.choice(2)
        }
    ]

    output = txt_synth(
        img=img_dir,
        text_list=text_list,
        font_list=font_list,
        region_list=region_list,
        effect_list=effect_list,
        is_top_left_origin=True,
        ensure_num_regions=True,
        is_keep_ratio=True,
    )
    output_annotation = txt_synth.plot_annotations()
    cv2.imwrite('synthesis_rand_effects.png', cv2.cvtColor(
        output_annotation.astype(np.uint8), cv2.COLOR_RGB2BGR))
