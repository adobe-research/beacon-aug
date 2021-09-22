#bin/bash
### This script is adopted from https://github.com/chail/gan-ensembling


## 1. Download the GAN-ensembling source code

cd ../../external/   # beacon_aug/external
git clone https://github.com/chail/gan-ensembling
mv gan-ensembling gan_ensembling     # rename the folder to  avoid hyphen threw an error...

cd gan_ensembling



## 2. Download dataset

bash resources/download_resources.sh
rm -rf *zip

## 2. Install the dependencies to env `aug`


#rename utils to uutls
mv  utils  uutils    # rename the folder to  avoid replicate module error...
find . -name '*.py' -exec sed -i -e 's/from utils/from uutils/g' {} \;

conda env update --file environment.yml
pip install --upgrade --force-reinstall torchvision

