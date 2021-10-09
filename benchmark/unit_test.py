# Copyright 2021 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.
import glob
import os
import matplotlib.pyplot as plt
from skimage import io
import time
from PIL import Image
import yaml
import beacon_aug as BA
from beacon_aug.generator.operator_generator import DEFAULT_LIBRARIES
import pandas as pd
import numpy as np

def simple_augmentation  ():
    print ("\n" + "="*5 + " Unit test 1: simple augmentation operator " + "="*5  + "\n")

    image = io.imread("../data/example.png")

    for library in ["imgaug", "albumentations","torchvision","keras","augly"]:

        print (library)
        aug_pipeline =  BA.Rotate(p=1, limit = (-45,45), library = library)

        augmented_image1 = aug_pipeline(image=image)['image'].copy()
        augmented_image2 = aug_pipeline(image=image)['image'].copy()

        assert (np.array_equal(image,augmented_image1) == False) 
        print ("\t Passed! beacon_aug generate augmentation different to input")
        
        assert (np.array_equal(augmented_image1,augmented_image2) == False )
        print ("\t Passed! beacon_aug generate different augmentation results when calling ")



if __name__ == "__main__":
    simple_augmentation()
