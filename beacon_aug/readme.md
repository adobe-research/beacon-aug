# Beacon_aug


## How to use

```python
import beacon_aug as BA
aug = BA.HorizontalFlip(p=1,  library="albumentations")
image_auged = aug(image=image)["image"]
```


## Directory Structure 
<p align="center">
  <img width="300"  src="docs/_images/flowchart.png">
</p>

     .
    ├── beacon_aug
    │   ├── beacon_aug.py               # class definition for all operators in each library
    │   ├── generator                           
    │   │    ├── standard               # folder for build-in library_convertion [.json] file
    │   │    ├── custom                 # folder for customized function [.py] file
    │   │    └── operator_generator.py  # operator dynamic class generator
    │   ├── advanced                                               
    │   │    ├── autoaugment.py         # AugAugment algorithm implementation          
    │   │    ├── randomaugment.py       # RandAugment algorithm implementation       
    │   │    └── collections.py         # Collection of operators
    │   ├── properties.py               # properties functions for operators
    │   └── screenshot.py               # visualization result for library comparisons 
    └── setup.py                        # dependencies installation

