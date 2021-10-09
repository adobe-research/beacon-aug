# Copyright 2021 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.


"""

e.g.
if use a a third party 
>> python setup.py install 
if use for edit
>> python setup.py develop 

"""

import argparse
import setuptools
import subprocess
import os,io,re

def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, "beacon_aug", "__init__.py")
    with io.open(version_file, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


if __name__ == "__main__":
    """Main method"""

    with open('requirements.txt') as f:
        required = f.read().splitlines()

    setuptools.setup(name='Beacon_aug',
                     version=get_version(),
                     description='Cross-library augmentation module for deep learning training',
                     author='Rebecca Li, Yannick Hold-Geoffroy, Geoffrey Oxholm, etc',
                     author_email='xiaoli@adobe.com',
                     url='https://github.com/adobe-research/beacon-aug',
                     packages=setuptools.find_packages(exclude=["workspace"]),
                     package_dir={'beacon_aug': 'beacon_aug'},
                    #  package_data={"beacon_aug.generator.standard":["*.yaml"],
                    #                 "beacon_aug.external.imagenet_c.frost":['*']}        
                    include_package_data=True,

                     )
    # for augly dependency
    subprocess.run(["conda", "install", "-c", "conda-forge", "python-magic"])
    subprocess.run(["conda", "install", "-c", "conda-forge/label/cf202003", "imagemagick"])

    print('''\n\n  Package "beacon_aug"  has successfully installed! You can try e.g.
=======================
>>>> import beacon_aug as BA
>>>> aug = BA.HorizontalFlip(p=1,  library="a")
>>>> aug = BA.HorizontalFlip(p=1,  library="iaa")        
>>>> image_auged = aug(image=image)["image"]
    '''
          )
