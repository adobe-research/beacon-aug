# Copyright 2021 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.


#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Wrapper of FreeType of text render
# Only support English
# Date: 12/20/2018
# Contact: zzhang@adobe.com
#

from os.path import isfile, exists, join
from os import makedirs
from freetype import *
import numpy as np
import logging
from scipy.misc import imsave


class TextRender(object):

    def __init__(self):
        self.text = None
        self.font_dir = None
        self.face = None
        self.char_box = []
        self.base_line = []

    def __call__(self, text, font, size=100*64, bg=0, interval=0, margin=0):
        return self.get_text_image(text=text, font=font, size=size, bg=bg, interval=interval, margin=margin)

    def get_text_image(self, text, font, size=100*64, bg=0, interval=0, margin=0):
        """
        render text image from string text and font file
        :param text: str, text to be rendered
        :param font: str, dir to the font file
        :param size: int, normalized font size (in points)
        :param bg: 0 or 1, 0-black background, 1-white background
        :param interval: a float greater than .5, shrinking (<1) or increasing (>1) the interval between characters
        :return: rendered text image
        """
        assert isinstance(text, str)

        self.text = text

        self._load_font(font=font, size=size)

        """ visualize the font file """
        # self.get_face_info()
        # self.get_sfnt_info()
        # self.list_charmap()

        """ render text """
        # self._render_char(bg=bg)
        return self._render_text(bg=bg, interval_offset=interval, margin=margin)

    def get_face_info(self):
        """
        print basic information of a font file
        """
        if not self.face:
            return

        print('***********************\n'
              '****** Face Info ******\n'
              '***********************')
        print('Face index:          {}'.format(self.face.face_index))
        print('Family name:         {}'.format(self.face.family_name.decode()))
        print('Style name:          {}'.format(self.face.style_name.decode()))
        print('Format:              {}'.format(self.face.get_format))
        print('Charmaps:            {}'.format(
            [charmap.encoding_name for charmap in self.face.charmaps]))
        print('')
        print('Face number:         {}'.format(self.face.num_faces))
        print('Glyph number:        {}'.format(self.face.num_glyphs))
        print('Available sizes:     {}'.format(self.face.available_sizes))
        print('')
        print('units per em:        {}'.format(self.face.units_per_EM))
        print('ascender:            {}'.format(self.face.ascender))
        print('descender:           {}'.format(self.face.descender))
        print('height:              {}'.format(self.face.height))
        print('')
        print('max_advance_width:   {}'.format(self.face.max_advance_width))
        print('max_advance_height:  {}'.format(self.face.max_advance_height))
        print('')
        print('underline_position:  {}'.format(self.face.underline_position))
        print('underline_thickness: {}'.format(self.face.underline_thickness))
        print('')
        print('Has horizontal:      {}'.format(self.face.has_horizontal))
        print('Has vertical:        {}'.format(self.face.has_vertical))
        print('Has kerning:         {}'.format(self.face.has_kerning))
        print('Is fixed width:      {}'.format(self.face.is_fixed_width))
        print('Is scalable:         {}'.format(self.face.is_scalable))
        print('')

    def get_sfnt_info(self):
        """
        print SFNT info. SFNT stands for spline font or scalable font
        """
        if not self.face:
            return

        print('***********************\n'
              '****** SFNT Info ******\n'
              '***********************')

        def platform_name(platform_id):
            for key, value in TT_PLATFORMS.items():
                if value == platform_id:
                    return key
            return 'Unknown platform'

        def encoding_name(platform_id, encoding_id):
            if platform_id == TT_PLATFORM_APPLE_UNICODE:
                encodings = TT_APPLE_IDS
            elif platform_id == TT_PLATFORM_MACINTOSH:
                encodings = TT_MAC_IDS
            elif platform_id == TT_PLATFORM_MICROSOFT:
                encodings = TT_MS_IDS
            elif platform_id == TT_PLATFORM_ADOBE:
                encodings = TT_ADOBE_IDS
            else:
                return 'Unknown encoding'
            for key, value in encodings.items():
                if value == encoding_id:
                    return key
            return 'Unknown encoding'

        def language_name(platform_id, language_id):
            if platform_id == TT_PLATFORM_MACINTOSH:
                languages = TT_MAC_LANGIDS
            elif platform_id == TT_PLATFORM_MICROSOFT:
                languages = TT_MS_LANGIDS
            else:
                return 'Unknown language'
            for key, value in languages.items():
                if value == language_id:
                    return key
            return 'Unknown language'

        name = self.face.get_sfnt_name(0)
        print('platform_id:', platform_name(name.platform_id))
        print('encoding_id:', encoding_name(name.platform_id, name.encoding_id))
        print('language_id:', language_name(name.platform_id, name.language_id))
        for i in range(self.face.sfnt_name_count):
            name = self.face.get_sfnt_name(i).string
            print(i, name.decode('utf-8', 'ignore'))

    def list_charmap(self, verb=True):
        """
        list current charmap by index, code, and name
        :param verb: boolean, whether print the list
        :return: the charmap in terms of indices (array), codes (array), names (list)
        """
        if not self.face:
            return

        try:
            charmap_name = self.face.charmap.encoding_name
        except:
            logging.warning('No charmap selected!')
            return []
        else:
            chars = sorted([(c[1], c[0]) for c in self.face.get_chars()])
            if verb:
                print('***********************\n'
                      '******* Charmap *******\n'
                      '***********************')
                print('Current charmap: %s' % charmap_name)
                print('{:>10}{:>10}{:>10}'.format('Index', 'Code', 'Name'))
            indices, codes, names = [], [], []
            for char in chars:
                glyph_name = self.face.get_glyph_name(char[0]).decode()
                indices.append(char[0])
                codes.append(char[1])
                names.append(glyph_name)
                if verb:
                    print('{:>10}{:>10}{:6}{}'.format(indices[-1], codes[-1], '', names[-1]))
            return np.array(indices), np.array(codes), names

    def _load_font(self, font, size=100*64):
        """
        read font file
        :param font: str, dir to the font file
        :param size: int, normalized font size (in points)
        """
        assert isfile(font)
        assert isinstance(size, (int, float))
        size = round(size)

        self.font_dir = font
        self.face = Face(font)

        """ set normalized font size (in points) """
        self.face.set_char_size(width=size)

        """ select character map """
        self._select_charmap()

    def _select_charmap(self):
        """
        select charactor map
        """
        if not self.face:
            return

        charmaps = self.face.charmaps
        charmap_names = [charmap.encoding_name for charmap in charmaps]
        if len(charmaps) > 0:
            idx = 0
            if 'FT_ENCODING_UNICODE' in charmap_names:
                idx = charmap_names.index('FT_ENCODING_UNICODE')
            self.face.select_charmap(charmaps[idx].encoding)

    def _render_char(self, save_dir='./chars', bg=0):
        """
        render characters, assuming horizontal metrics
        :param save_dir: str, dir to save character images
        :param bg: 0 or 1, 0-black background, 1-white background
        :return:
        """
        if not exists(save_dir):
            makedirs(save_dir)

        """ get the glyph structure that contains a given glyph image """
        slot = self.face.glyph
        for c in self.text:
            # self.face.load_char(c)

            """ load a glyph into slot """
            idx = self.face.get_char_index(c)
            self.face.load_glyph(idx)

            """ get the glyph image """
            bitmap = slot.bitmap
            width = bitmap.width
            rows = bitmap.rows
            pitch = bitmap.pitch  # number of bytes per row

            """ convert to image matrix """
            data = []
            for i in range(rows):
                data.extend(bitmap.buffer[i * pitch:i * pitch + width])
            data = np.array(data, dtype=np.ubyte).reshape(rows, width)
            if bg == 1:
                data = 255 - data
            imsave(join(save_dir, '{:04d}_{}.png'.format(idx, c)), data)

    def _render_text(self, bg=0, interval_offset=1, margin=0):
        """
        render text image, assuming horizontal metrics
        for more details:
        https://www.freetype.org/freetype2/docs/tutorial/step2.html
        https://www.freetype.org/freetype2/docs/glyphs/glyphs-3.html
        :param bg: 0 or 1, 0-black background, 1-white background
        :param interval_offset: an float from .5 to 2, 1 is the original interval
        :param margin: int width of margin in pixel
        :return: array, rendered text image, value range [0, 255]
        """

        # assert bg == 0 or bg == 1
        # assert isinstance(interval_offset, (int, float)) and interval_offset >= .5
        margin = max(int(margin), 0)
        """ Zero pass to determine character indices """
        indices = []
        for c in self.text:
            idx = self.face.get_char_index(c)
            try:
                self.face.load_glyph(idx)
                indices.append(idx)
            except:
                logging.info('''Failed loading the glyph "{}" in "{}"'''.format(c, self.text))
                return None

        """ get the glyph structure that contains a given glyph image """
        slot = self.face.glyph

        """ First pass to compute bbox """
        height, width, origin_y, bearing_y = 0, 0, 0, 0
        for idx, c in enumerate(self.text):
            self.face.load_glyph(indices[idx])
            metrics = slot.metrics

            # print('****************************************')
            # bitmap = slot.bitmap
            # print('bitmap.rows = {}'.format(bitmap.rows))
            # print('bitmap.width = {}'.format(bitmap.width))
            # print('bitmap.pitch = {}'.format(bitmap.pitch))
            # print('')
            # print('slot.bitmap_top = {}'.format(slot.bitmap_top))
            # print('slot.bitmap_left = {}'.format(slot.bitmap_left))
            # print('slot.advance.x = {}'.format(slot.advance.x >> 6))
            # print('slot.advance.y = {}'.format(slot.advance.y >> 6))
            # print('')
            # print('metrics.width = {}'.format(metrics.width >> 6))
            # print('metrics.height = {}'.format(metrics.height >> 6))
            # print('metrics.horiBearingX = {}'.format(metrics.horiBearingX >> 6))
            # print('metrics.horiBearingY = {}'.format(metrics.horiBearingY >> 6))
            # print('metrics.horiAdvance = {}'.format(metrics.horiAdvance >> 6))
            # print('metrics.vertBearingX = {}'.format(metrics.vertBearingX >> 6))
            # print('metrics.vertBearingY = {}'.format(metrics.vertBearingY >> 6))
            # print('metrics.vertAdvance = {}'.format(metrics.vertAdvance >> 6))
            # print('self.face.get_kerning.x = {}'.format(self.face.get_kerning(self.text[idx - 1], c).x >> 6))

            """ 
            IMPORTANT NOTE:
            Under FreeType, scaled pixel positions are all expressed in the 26.6 fractional pixel format 
            (made of a 26-bit integer mantissa, and a 6-bit fractional part). 
            In other words, all coordinates are multiplied by 64 (<< 6). 
            http://chanae.walon.org/pub/ttf/ttf_glyphs.htm
            """

            bearing_y = max(bearing_y, max(0, metrics.horiBearingY >> 6))
            origin_y = max(origin_y, max(0, (metrics.height >> 6) - (metrics.horiBearingY >> 6)))
            width += (metrics.horiAdvance >> 6)
            if idx == 0:
                width += abs(metrics.horiBearingX >> 6) if (metrics.horiBearingX >> 6) < 0 else 0

        """ height of text image """
        height = bearing_y + origin_y

        """ adjust interval between two adjacent characters """
        if len(self.text) > 1:
            interval_offset = int(round(width * (interval_offset - 1.) / (len(self.text) - 1)))
        else:
            interval_offset = 0

        """ adjusted width of text image """
        width += interval_offset * (len(self.text) - 1)

        """ baseline of the text (two end points in image coord)"""
        self.base_line = [
            [0, bearing_y],
            [width - 1, bearing_y]
        ]

        """ storage for text image """
        if height < 1 or width < 1:
            return None
        data = np.zeros((height, width), dtype=np.float32)

        """ Second pass for actual rendering """
        x, y, previous = 0, 0, 0
        self.char_box = []
        for idx, c in enumerate(self.text):
            # self.face.load_char(c)
            self.face.load_glyph(indices[idx])
            bitmap = slot.bitmap
            metrics = slot.metrics

            """ height and width of a glyph image"""
            h, w = metrics.height >> 6, metrics.width >> 6

            """ top-left coord to paste the glyph image onto the text image """
            y = height - origin_y - (metrics.horiBearingY >> 6)
            x_offset = (self.face.get_kerning(previous, c).x >> 6) + (metrics.horiBearingX >> 6)
            x += x_offset
            if x < 0:
                x, x_offset = 0, 0

            """ paste the glyph to the text image """
            img_glyph = np.array(bitmap.buffer, dtype=np.ubyte).reshape(h, w)
            try:
                data[y:y + h, x:x + w] += img_glyph
            except ValueError:
                if data.shape[0] < y + h:
                    data = np.concatenate(
                        [data, np.zeros((y + h - data.shape[0], data.shape[1]))], axis=0)
                elif data.shape[1] < x + w:
                    data = np.concatenate(
                        (data, np.zeros((data.shape[0], x + w - data.shape[1]))), axis=1)
                # patch = img_glyph[:data.shape[0]-y, :data.shape[1]-x]
                # h, w = patch.shape
                # data[y:y + h, x:x + w] += patch
                data[y:y + h, x:x + w] += img_glyph

            """ recode character bounding box (top-left, CW, image coord) """
            if c is not ' ':
                self.char_box.append([
                    [x, y],
                    [x + w - 1, y],
                    [x + w - 1, y + h - 1],
                    [x, y + h - 1]
                ])

            """ move to the start point for the next glyph """
            x += (metrics.horiAdvance >> 6) - (metrics.horiBearingX >> 6) + interval_offset
            previous = c
        data = data.clip(0, 255)

        """ trim tail """
        # data = data[:, :self.char_box[-1][1][0] + 1]
        self.base_line[0][0] = self.char_box[0][0][0]
        self.base_line[1][0] = self.char_box[-1][1][0]

        """ add margin """
        height, width = data.shape[:2]
        if margin > 0:
            tmp = np.zeros((height + margin * 2, width + margin * 2), dtype=np.float32)
            tmp[margin:margin + height, margin:margin + width] = data
            data = tmp
            for i in range(len(self.char_box)):
                for j in range(4):
                    self.char_box[i][j][0] += margin
                    self.char_box[i][j][1] += margin
            self.base_line[0][0] += margin
            self.base_line[1][0] += margin
            self.base_line[0][1] += margin
            self.base_line[1][1] += margin

        """ switch background and foreground color """
        if bg == 1:  # white background
            data = 255 - data

        return data.astype(np.uint8)


if __name__ == '__main__':

    import cv2
    from time import time
    import string

    font_file = './data/font/Vera.ttf'
    font_file = '/mnt/ilcompf3d0/data/font/font/typekit+mono+sys-21718/fontlinks-21717/'
    text_str = string.ascii_letters

    render = TextRender()

    """ render text with white background and black text, shrink text interval """
    bg = 1
    t_start = time()
    for _ in range(1):
        img = render(text=text_str, font=font_file, bg=bg, interval=1, size=200*64)
    print(time() - t_start)
    # exit()
    # draw baseline
    base_line = render.base_line
    # cv2.line(img, base_line[0], base_line[1], color=255 * (1 - bg), thickness=1)
    # draw character boxes
    # char_box = render.char_box
    # cv2.drawContours(img, contours=np.array(char_box), contourIdx=-1, color=255 * (1 - bg), thickness=1)
    imsave('./render1.png', img)
    exit()

    """ render text with black background and white text, enlarge text interval """
    bg = 0
    img = render(text=text_str, font=font_file, bg=bg, interval=1.4)
    # draw baseline
    base_line = render.base_line
    cv2.line(img, base_line[0], base_line[1], color=255 * (1 - bg), thickness=1)
    # draw character boxes
    char_box = render.char_box
    cv2.drawContours(img, contours=np.array(char_box), contourIdx=-
                     1, color=255 * (1 - bg), thickness=1)
    imsave('./render2.png', img)
