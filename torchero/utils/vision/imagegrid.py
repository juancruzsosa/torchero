import math
from collections import defaultdict, OrderedDict

import torch
import numpy as np
from PIL.Image import Image
from matplotlib import pyplot as plt

def normalize_tensor_image(img):
    if img.ndim not in (2, 3):
        raise TypeError(
            "Dataset images must have shape (3, HEIGHT, WIDTH) or "
            "(HEIGHT, WIDTH)"
        )
    if (img.ndim == 3) and (img.shape[0] not in (1, 3)):
        raise TypeError(
            "Invalid image shape: {}. Dataset images must have shape "
            "(3, HEIGHT, WIDTH) or (HEIGHT, WIDTH)"
            .format(img.shape)
        )
    if img.ndim == 3 and img.shape[0] == 1:
        img = img.squeeze(0)
    if img.ndim == 3:
        min_c = (img.view(3, -1)
                    .min(axis=1)
                    .values
                    .unsqueeze(-1)
                    .unsqueeze(-1))
        max_c = (img.view(3, -1)
                    .max(axis=1)
                    .values
                    .unsqueeze(-1)
                    .unsqueeze(-1))
        img = (img - min_c)/(max_c-min_c)
    return img


def get_imagegrid(dataset,
                  num=10,
                  shuffle=True):
    if shuffle:
        indices = torch.randperm(len(dataset))
    else:
        indices = torch.arange(0, len(dataset))
    return [dataset[i] for i in indices[:num]]

def get_labeled_imagegrid(dataset,
                          num=10,
                          shuffle=True,
                          classes='auto'):
    if isinstance(classes, str):
        if classes != 'auto':
            raise TypeError(
                "Bad parameter classes. classes mut be 'auto', None, or an "
                "array with classes names"
            )
        if hasattr(dataset, 'classes'):
            classes = dataset.classes
        else:
            classes = None

    if shuffle:
        indices = torch.randperm(len(dataset))
    else:
        indices = torch.arange(0, len(dataset))

    images_per_class = defaultdict(list)
    for i in indices:
        image, class_id = dataset[i]

        if classes is not None:
            class_name = classes[class_id]
        else:
            class_name = str(class_id)

        if len(images_per_class[class_name]) < num:
            images_per_class[class_name].append(image)

        if ((classes is None or (len(classes) == len(images_per_class))) and
            (all(len(class_images) == num for class_images in images_per_class.values()))):
            break

    sorted_images_per_class = OrderedDict(sorted(images_per_class.items(),
                                          key=lambda x: x[0]))

    return sorted_images_per_class



def show_image(img, ax, image_attr={'cmap': plt.cm.Greys_r}):
    if ax is None:
        ax = plt.gca()
    if isinstance(img, torch.Tensor):
        img = normalize_tensor_image(img)
        if img.ndim == 3:
            img = img.permute(1, 2, 0)
    ax.imshow(img, **image_attr)
    ax.set_xticks([])
    ax.set_yticks([])


def show_imagegrid_dataset(dataset,
                           num=10,
                           shuffle=True,
                           classes='auto',
                           figsize=None,
                           fontsize=20,
                           image_attr={'cmap': plt.cm.Greys_r}):
    """ Plot a image grids of the dataset. If dataset is labeld it will
    generate one row of images per class and if it's unlabeled it will generate
    a bidimentional (almost square) grid of images sampled from dataset


    Parameters:
        num (str):
            Number of images per grid
        shuffle (bool):
            True to sample the images randomly, false for sequential sampling
        class (str, or list):
            If 'auto' is passed, it willl use the class names
            from dataset `classes` attribute (or use the class number if the
            dataset has such property). If a list is passed then the name the
        figsize (tuple):
            If None infers the matplotlib figure size from the number of columns and rows,
            if not none then the tuple for the matplotlib figuresize.
        fontfize (float):
            Text font size
        image_attr (str):
            Matplotlib image attributes for all images
    """
    sample = dataset[0]
    if isinstance(sample, tuple) and len(sample) == 2:
        images_per_class = get_labeled_imagegrid(dataset,
                                                 num=num,
                                                 shuffle=shuffle,
                                                 classes=classes)
        num = min(num, max(map(len, images_per_class.values())))
        classes = list(images_per_class.keys())

        if figsize is None:
            figsize = (2 * num, 2 * len(classes))
        fig, axs = plt.subplots(figsize=figsize, nrows=len(classes), ncols=num)
        if len(classes) == 1:
            axs = np.expand_dims(axs, 0)
        if num == 1:
            axs = np.expand_dims(axs, -1)
        for i, (class_name, class_images) in enumerate(images_per_class.items()):
            for j, img in enumerate(class_images):
                show_image(img, axs[i][j], image_attr)
            axs[i][0].set_ylabel(str(class_name), fontsize=fontsize)
    elif isinstance(sample, (Image, torch.Tensor, np.ndarray)):
        image_list = get_imagegrid(dataset,
                                   num=num,
                                   shuffle=shuffle)
        num = min(len(image_list), num)
        nrows = math.ceil(math.sqrt(num))
        ncols = math.ceil(num / nrows)
        if figsize is None:
            figsize = (2 * nrows, 2 * ncols)
        fig, axs = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)
        axs = axs.flatten()
        for i, img in enumerate(image_list):
            show_image(img, axs[i], image_attr)
