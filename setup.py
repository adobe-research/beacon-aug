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


if __name__ == "__main__":
    """Main method"""

    with open('requirements.txt') as f:
        required = f.read().splitlines()

    setuptools.setup(name='Beacon_aug',
                     version='1.0',
                     description='Cross-library augmentation module for deep learning training',
                     author='Rebecca Li, Yannick Hold-Geoffroy, Geoffrey Oxholm, etc',
                     author_email='xiaoli@adobe.com',
                     url='https://git.corp.adobe.com/xiaoli/augmenter',
                     packages=['beacon_aug'],
                     package_dir={'beacon_aug': 'beacon_aug'},
                     install_requires=required,
                     )
    # for augly dependency
    subprocess.run(["conda", "install", "-c", "conda-forge", "python-magic"])

    print('''\n\n  Package "beacon_aug"  has successfully installed! You can try e.g.
=======================
>>>> import beacon_aug as BA
>>>> aug = BA.HorizontalFlip(p=1,  library="a")
>>>> aug = BA.HorizontalFlip(p=1,  library="iaa")        
>>>> image_auged = aug(image=image)["image"]
    '''
          )
