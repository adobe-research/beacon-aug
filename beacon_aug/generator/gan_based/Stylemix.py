# Copyright 2021 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

'''
Example of using and GAN model as an augmentation customized function
To run this Stylemix module
    
'''

import random
import sys
import numpy as np
from PIL import Image
import torch
import beacon_aug as BA
import sys
import os

# add external package in module path
package_path = os.path.join(os.path.dirname(BA.__file__), "external", "gan_ensembling")
if os.path.exists(package_path):
    os.chdir(package_path)
    sys.path.insert(0, package_path)

    from uutils import renormalize, show
    import data
    from networks import domain_generator, domain_classifier


# add external package in module path
package_path = os.path.join(os.path.dirname(BA.__file__), "external", "gan_ensembling")
if os.path.exists(package_path):
    os.chdir(package_path)
    sys.path.insert(0, package_path)


class Stylemix:
    """Gan-ensembling Stylemix augmentation
    Source: https://chail.github.io/gan-ensembling/
    Args:
        dataset(str): domain target 
        layer(str): 'fine' or 'coarse', control the GAN reconstruction layer, "find" cause less change, "coarse" cause more changes
        img_choice (int): the image id in the latent space. Select in Class construction step
    Output: sythetic generated image same size with input image

    First run 
        .. code-block::

            bash prepare_gan_ensembling.sh

    e.g.

    - Test the function:    

        .. code-block::

            import beacon.beacon_aug as BA
            aug = BA.Stylemix(p=1, dataset="cat")
            image_auged = aug(image=image)["image"]
    """

    def __init__(self, dataset="cat", img_choice=None, layer="coarse"):
        self.image_only = True  # True only applies  to image; False applies to both image and bbox
        self.library = "gan_based"

        assert layer in ["fine", 'coarse']

        self.img_choice = img_choice
        self.dataset = dataset

        # construct config
        if self.dataset == 'celebahq':
            dataset_name = 'celebahq'
            generator_name = 'stylegan2'
            classifier_name = 'Smiling'
            val_transform = data.get_transform(dataset_name, 'imval')
            self.dset = data.get_dataset(dataset_name, 'val', classifier_name,
                                         load_w=True, transform=val_transform)
            self.generator = domain_generator.define_generator(
                generator_name, dataset_name, load_encoder=False)
            # classifier = domain_classifier.define_classifier(dataset_name, classifier_name)
            # centercrop to the appropriate dimension for classifier
            # tensor_transform_val = data.get_transform(dataset_name, 'tensorbase')
            self.tensor_transform_ensemble = data.get_transform(
                dataset_name, 'tensormixed')  # alternatively, can just use tensorbase
        elif self.dataset == 'car':
            dataset_name = 'car'
            generator_name = 'stylegan2'
            classifier_name = 'latentclassifier_stylemix_fine'
            val_transform = data.get_transform(dataset_name, 'imval')
            self.dset = data.get_dataset(dataset_name, 'val', load_w=True, transform=val_transform)
            self.generator = domain_generator.define_generator(
                generator_name, dataset_name, load_encoder=False)
            # classifier = domain_classifier.define_classifier(dataset_name, classifier_name)
            # centercrop to the appropriate dimension for classifier
            # tensor_transform_val = data.get_transform(dataset_name, 'tensorbase')
            self.tensor_transform_ensemble = data.get_transform(
                dataset_name, 'tensormixed')  # alternatively, can just use tensorbase
        elif self.dataset == 'cat':
            dataset_name = 'cat'
            generator_name = 'stylegan2'
            classifier_name = 'latentclassifier_stylemix_coarse'
            val_transform = data.get_transform(dataset_name, 'imval')
            self.dset = data.get_dataset(dataset_name, 'val', load_w=True, transform=val_transform)
            self.generator = domain_generator.define_generator(
                generator_name, dataset_name, load_encoder=False)
            # classifier = domain_classifier.define_classifier(dataset_name, classifier_name)
            # centercrop to the appropriate dimension for classifier
            # tensor_transform_val = data.get_transform(dataset_name, 'tensorbase')
            self.tensor_transform_ensemble = data.get_transform(
                dataset_name, 'tensormixed')  # alternatively, can just use tensorbase
        else:
            assert(False)

    def __call__(self, img):
        layer = "fine"

        random_seed = random.randint(0, 1048)
        batch_size = 1

        index = random.randint(0, 599) if self.img_choice == None else self.img_choice
        assert index in range(0, 599)
        print("index = ", index)

        with torch.no_grad():
            # stylemix fine
            latent = self.dset[index][1][None].cuda()
            mix_latent = self.generator.seed2w(n=batch_size, seed=random_seed)
            perturbed_im = self.generator.perturb_stylemix(latent, layer, mix_latent, n=batch_size)
            postprocessed_im = self.tensor_transform_ensemble(perturbed_im)

            original_image = self.dset[index][1][None].cuda()

        generated_img = renormalize.as_image(postprocessed_im[0])

        return generated_img.resize((img.shape[0], img.shape[1]))
