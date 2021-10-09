# Copyright 2021 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.


from .operators import *           # the class are not known until run time
from .advanced.autoaugment import AutoAugment
from .advanced.randaugment import RandAugment
from .advanced.collections import Collections
from .advanced.benign_transforms import Benign
from .generator import *
from . import screenshot
from . import properties

# Inherit the core module from albumentations
from albumentations.core.composition import *
from albumentations.core.serialization import *

__version__ = "Opensource-09142021"
__release__ = __version__
