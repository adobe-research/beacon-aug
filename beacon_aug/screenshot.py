# Copyright 2021 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
import albumentations as A
from skimage import data
import os
from copy import deepcopy
import random
import time
from PIL import Image
from skimage.color import label2rgb
import beacon_aug as BA

''' flatten the pipeline tree'''
def extract_single_operation(augPipeline):
    def flatten(dict, flatten_ls=[]):
        '''use DFS to unfold the operations'''
        for operation in dict["transforms"]:  # "OneOf" or "OneOrOther", etc
            class_name = operation['__class_fullname__']
            if "." in class_name:
                if operation['__class_fullname__'].split(".")[-2] == "composition":
                    flatten(operation, flatten_ls)
                    continue
            flatten_ls.append(operation)
        return flatten_ls
    transform_dict = A.to_dict(augPipeline)
    flatten_ls = flatten(transform_dict["transform"])
    return [{'__version__': transform_dict['__version__'], 'transform':opr} for opr in flatten_ls]


def screenshot_pipeline(augPipeline, image, save_fig_path=None):
    ''' Visualize an augmentation pipeline by displaying the extreme case for all the parameters
    '''
    # get the flattened operator sequence avoiding hierarchical structure
    single_operation_ls = extract_single_operation(augPipeline)

    numOfOperation = len(single_operation_ls)
    fig, axs = plt.subplots(numOfOperation, 3,
                            figsize=(6, 2*numOfOperation),
                            constrained_layout=True)

    axs[0, 1].set_title("Lower Limit")
    axs[0, 2].set_title("Upper Limit")

    for i, single_operation in enumerate(single_operation_ls):
        # Extract the upper and lower limit
        transform_name = single_operation["transform"]['__class_fullname__'].split(".")[-1]
        # deep copy to avoid pointing save location in dict
        lowerAndUpper = [single_operation, deepcopy(single_operation)]
        limit_para_name = None
        # Extract all the limit parameters
        for para in single_operation["transform"]:
            if para == "p":   # change prob to 1 to make it always happen
                lowerAndUpper[0]["transform"][para] = 1
                lowerAndUpper[1]["transform"][para] = 1

            if "limit" in para:
                limit_para_name = para
                original_values = list(single_operation["transform"][para])
                lowerAndUpper[0]["transform"][para] = [original_values[0]]*2
                lowerAndUpper[1]["transform"][para] = [original_values[1]]*2

        # plot
        for lu in range(2):                          # lower or upper limit
            lu_transform = A.from_dict(lowerAndUpper[lu])
            axs[i, lu+1].imshow(lu_transform(image=image)["image"])
            axs[i, lu+1].axis("off")

        if limit_para_name:
            axs[i, 0].text(0.15, 0.5, transform_name+"\n" + limit_para_name+":" +
                           str(lowerAndUpper[0]["transform"][limit_para_name][0]) + "," +
                           str(lowerAndUpper[1]["transform"][limit_para_name][1]), dict(size=10))
        else:
            axs[i, 0].text(0.15, 0.5, transform_name, dict(size=10))
        axs[i, 0].axis("off")
    if save_fig_path:
        figname = os.path.join(save_fig_path, "aug_pipeline-screenshot.png")
        print("\n...screenshot figure save as : ", figname)
        plt.savefig(figname)
    return fig


def screenshot_library(BA_operator, image_data, save_fig_path=None, individual_fig=False, **kwargs):
    '''    Visualize the augmentation result comparision to all available libraries
    e.g.
    ----
    import beacon_aug as BA
    from  beacon_aug import screenshot
    fig, __ = BA.screenshot.screenshot_library(BA.Brightness(), image_data=image)
    fig.show()
    '''
    avail_libraries = BA_operator(**kwargs).avail_libraries

    numOfLibraries = len(avail_libraries)
    fig, axs = plt.subplots(2, 1 + numOfLibraries,
                            figsize=(4*numOfLibraries, 4),
                            constrained_layout=True)
    fig.suptitle("beacon_aug."+BA_operator.__name__ + " with " +
                 str(kwargs))  # or plt.suptitle('Main title')

    axs[0][0].imshow(image_data)
    axs[0][0].set_title("Raw")
    axs[1][0].text(0.3, 0.5, "Difference to\n" + "raw")
    axs[1][0].axis("off")

    attributes_result = {"runtime": {}, "differentiable": {}}
    # axs[1][0].text(0.3, 0.5, "Sanity Check:\n p=0  ->", dict(size=10))
    for i, library in enumerate(avail_libraries):
        t_before = time.time()
        op = BA_operator(always_apply=False, p=1, library=library, **kwargs)
        image_auged = op(image=image_data)["image"]

        t_after = time.time()

        runtime = t_after - t_before

        image_auged_vis = image_auged

        attributes_result["runtime"][library] = runtime

        attributes_result["differentiable"][library] = BA.properties.isOpDifferentiable(op)

        axs[0][1+i].set_title(library + ":" + '{0:.1f}'.format(runtime*1000) + " (ms)")
        axs[0][1+i].imshow(image_auged)

        # display the difference of original to augmented images
        if image_auged.shape == image_data.shape:
            axs[1][1+i].imshow(image_auged - image_data)

        if save_fig_path and individual_fig == True:
            img_name = os.path.join(save_fig_path, BA_operator.__name__+"-" + library+".jpeg")
            if os.path.isfile(img_name):
                print("\n...screenshot individual figure already existed as : ", img_name)
            else:
                if image_auged.min() < 0:   # normalzied case, need to
                    image_auged = image_auged - image_auged.min()
                    image_auged = image_auged/image_auged.max()
                    print("@@@@@@@", image_auged.min())

                plt.imsave(img_name, image_auged)
                print("\n...screenshot individual figure save as : ", img_name)

    fig.subplots_adjust(wspace=0)
    if save_fig_path and individual_fig == False:
        fig_name = os.path.join(save_fig_path, BA_operator.__name__+"aug_library-screenshot.png")
        print("\n...screenshot figure save as : ", fig_name)
        plt.savefig(fig_name)
    return fig, attributes_result


def visualize_bboxes(img, bboxes, color=(255, 0, 0), thickness=2, **kwargs):
    '''
    color = BOX_COLOR (BOX_COLOR = (255, 0, 0)  # Red
    '''
    image = img.copy()
    for bbox in bboxes:
        # x_min, y_min, w, h = bbox
        if len(bbox) == 5:
            bbox = bbox[:4]   # the last one is label
        x_min, y_min, x_max, y_max = map(int, bbox)    # need to make sure bbox is integer

        # x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
        img = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    return image


def visualize_kps(img, kps, color=(0, 255, 0), key_point_diameter=2, **kwargs):
    '''
    '''
    image = img.copy()
    for kp in kps:
        x, y = kp
        image = cv2.circle(image, (int(x), int(y)), key_point_diameter, color, -1)
    return image


def visualize_titles(img, bbox, title, color=(255, 0, 0), thickness=2, font_thickness=2, font_scale=0.35, **kwargs):
    x_min, y_min, x_max, y_max = map(int, bbox)    # x_min, y_min, w, h = bbox
    # x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    ((text_width, text_height), _) = cv2.getTextSize(
        title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)),
                  (x_min + text_width, y_min), color=(255, 0, 0))
    cv2.putText(img, title, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                font_thickness, lineType=cv2.LINE_AA)
    return img


def visualize_targets(image,  mask=None, bboxes=None, keypoints=None, image0=None):
    ''' Stack all the targets '''
    target_list = []
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    target_list.append(image.copy())

    if image0 is not None:
        if image0.ndim == 2:
            image0 = cv2.cvtColor(image0, cv2.COLOR_GRAY2RGB)
        target_list.append(image0)
    if mask is not None:
        target_list.append(cv2.cvtColor((mask*255).astype('uint8'), cv2.COLOR_GRAY2RGB))
    if bboxes is not None:
        target_list.append(visualize_bboxes(image, bboxes, thickness=10))
    if keypoints is not None:
        target_list.append(visualize_kps(image, keypoints, key_point_diameter=15))

    return np.hstack(target_list)


def augment_and_show(aug, image, mask=None, bboxes=[], keypoints=[], categories=[], category_id_to_name=[], filename=None,
                     font_scale_orig=0.35, font_scale_aug=0.35, key_point_diameter=15,
                     show_title=True, **kwargs):
    """
    Use from: https://albumentations.ai/docs/examples/showcase/
    visualize the image,(mask), (bbox),(kp) superimposed result before and after augmentation
    Args:
        aug: augmentation pipelineg
        image: single image
        mask: original mask
        bbox: original bounding boxes
        keypoints: original keypoints
    output:
        augmented: augmented image components
        f: visualize image
    """

    if mask is None:
        augmented = aug(image=image, bboxes=bboxes,
                        keypoints=keypoints, category_id=categories)
    else:
        augmented = aug(image=image, mask=mask, bboxes=bboxes,
                        keypoints=keypoints, category_id=categories)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image_aug = cv2.cvtColor(augmented['image'], cv2.COLOR_BGR2RGB)
    image_aug = augmented['image']

    visualize_bboxes(image, bboxes, **kwargs)
    visualize_bboxes(image_aug, augmented['bboxes'], **kwargs)

    visualize_kps(image, keypoints, **kwargs)
    visualize_kps(image, augmented["keypoints"], **kwargs)

    if show_title:
        for bbox, cat_id in zip(bboxes, categories):
            visualize_titles(
                image, bbox, category_id_to_name[cat_id], font_scale=font_scale_orig, **kwargs)
        for bbox, cat_id in zip(augmented['bboxes'], augmented['category_id']):
            visualize_titles(
                image_aug, bbox, category_id_to_name[cat_id], font_scale=font_scale_aug, **kwargs)

    if mask is None:
        f, ax = plt.subplots(1, 2, figsize=(16, 8))

        ax[0].imshow(image)
        ax[0].set_title('Original image')

        ax[1].imshow(image_aug)
        ax[1].set_title('Augmented image')
    else:
        f, ax = plt.subplots(2, 2, figsize=(16, 16))

        if len(mask.shape) != 3:
            mask = label2rgb(mask, bg_label=0)
            mask_aug = label2rgb(augmented['mask'], bg_label=0)
        else:
            import pdb
            pdb.set_trace()
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask_aug = cv2.cvtColor(augmented['mask'], cv2.COLOR_BGR2RGB)

        ax[0, 0].imshow(image)
        ax[0, 0].set_title('Original image')

        ax[0, 1].imshow(image_aug)
        ax[0, 1].set_title('Augmented image')

        ax[1, 0].imshow(mask, interpolation='nearest')
        ax[1, 0].set_title('Original mask')

        ax[1, 1].imshow(mask_aug, interpolation='nearest')
        ax[1, 1].set_title('Augmented mask')

    f.tight_layout()

    if filename is not None:
        f.savefig(filename)

    return augmented, f


if __name__ == "__main__":
    # Load an example image (uint8, 128x128x3).
    image = data.astronaut()

    # Example of an augmentation pipeline
    augPipeline = A.Compose([
        A.RandomCrop(256, 256),
        A.OneOf([A.RGBShift(),
                A.HueSaturationValue()])])

    os.makedirs("tmp", exist_ok=True)
    screenshot_pipeline(augPipeline, image, save_fig_path="tmp/")
