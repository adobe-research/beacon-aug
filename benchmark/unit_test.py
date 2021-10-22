# Copyright 2021 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.
'''
>> python -m unittest unit_test.TestSingle_operator

'''

from skimage import io
import time
import beacon_aug as BA
# from beacon_aug.generator.operator_generator import DEFAULT_LIBRARIES
import numpy as np

import unittest
from datetime import timedelta
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays


class TestSingle_operator(unittest.TestCase):
    # @given(image=arrays(np.float32, st.tuples(*[st.integers(1, 500), st.integers(1, 500), st.just(3)])),
    #        limit=st.tuples(*[st.integers(-128, 0), st.integers(1, 128)]))
    def testSingle_operator(self):
        image = io.imread("../data/example.png")
        limit = (-128, 128)
        print("\n" + "="*5 + " Unit test 1: Single augmentation operator " + "="*5 + "\n")

        for library in BA.Rotate().avail_libraries:
            aug_pipeline = BA.Rotate(p=1, limit=limit, library=library)

            augmented_image1 = aug_pipeline(image=image)['image'].copy()
            augmented_image2 = aug_pipeline(image=image)['image'].copy()

            # self.assertNotEqual(image, augmented_image1)
            assert not np.array_equal(
                image, augmented_image1), "Failed to generate augmentation different to input"

            # self.assertNotEqual(augmented_image1, augmented_image2)

            assert not np.array_equal(
                augmented_image1, augmented_image2), "Failed to generate different augmentation results when calling "


if __name__ == "__main__":
    unittest.main()
