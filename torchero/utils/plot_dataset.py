import math
from collections import defaultdict

import torch
import numpy as np
from PIL.Image import Image
from matplotlib import pyplot as plt


def show_image(img, ax, image_attr={'cmap': plt.cm.Greys_r}):
    if ax is None:
        ax = plt.gca()
    if isinstance(img, torch.Tensor):
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

    sample = dataset[indices[0]]
    if isinstance(sample, tuple) and len(sample) == 2:
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

        classes = list(images_per_class.keys())

        if figsize is None:
            figsize = (2 * num, 2 * len(classes))
        fig, axs = plt.subplots(figsize=figsize, nrows=len(classes), ncols=num)

        sorted_images_per_class = sorted(images_per_class.items(),
                                         key=lambda x: x[0])

        for i, (class_name, class_images) in enumerate(sorted_images_per_class):
            for j, img in enumerate(class_images):
                show_image(img, axs[i][j], image_attr)
            axs[i][0].set_ylabel(str(class_name), fontsize=fontsize)
    elif isinstance(sample, (Image, torch.Tensor, np.ndarray)):
        num = min(len(indices), num)
        indices = indices[:num]
        nrows = math.ceil(math.sqrt(num))
        ncols = math.ceil(len(indices) / nrows)
        if figsize is None:
            figsize = (2 * nrows, 2 * ncols)
        fig, axs = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)
        axs = axs.flatten()
        for i, ind in enumerate(indices):
            show_image(dataset[ind], axs[i], image_attr)
