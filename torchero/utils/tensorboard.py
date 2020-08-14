import torch
import numpy as np
from torch.utils import tensorboard
from torch.utils.tensorboard import SummaryWriter
from PIL.Image import Image
from torchvision import transforms

from torchero.utils.vision import get_imagegrid, get_labeled_imagegrid

def write_image(img, writer, tag, global_step=0):
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
    writer.add_image(tag, img, global_step=global_step)

to_pil_image = transforms.ToPILImage()

def prepare_images(ims, imsize='auto'):
    if isinstance(ims[0], torch.Tensor):
        ims = [to_pil_image(im) for im in ims]

    if imsize == 'auto':
        widths, heights = zip(*map(lambda x: x.size, ims))
        width, height = max(widths), max(heights)
    elif isinstance(imsize, float):
        width, height = imsize, imsize
    elif isinstance(imsize, [tuple, list]) and (len(imsize) == 2):
        width, height = imsize
    else:
        raise ValueError("Invalid imsize parameter. It should be one of: 'auto', tuple or float")

    transform = transforms.Compose([transforms.Resize((width, height)), transforms.ToTensor()])
    ims = torch.stack([ transform(im) for im in ims ])
    return ims

def write_imagegrid_dataset(writer,
                            tag,
                            dataset,
                            num=16,
                            shuffle=True,
                            classes='auto',
                            global_step=0,
                            imsize='auto'):
    sample = dataset[0]
    if isinstance(sample, tuple) and len(sample) == 2:
        images_per_class = get_labeled_imagegrid(dataset,
                                                 num=num,
                                                 shuffle=shuffle,
                                                 classes=classes)
        classes = list(images_per_class.keys())
        for i, (class_name, class_images) in enumerate(images_per_class.items()):
            images = prepare_images(class_images, imsize=imsize)
            writer.add_images(tag + '/' + class_name, images, global_step=global_step)
        writer.flush()
    elif isinstance(sample, (Image, torch.Tensor, np.ndarray)):
        image_list = get_imagegrid(dataset,
                                   num=num,
                                   shuffle=shuffle)
        image_list = prepare_images(image_list, imsize=imsize)
        writer.add_images(tag, image_list, global_step=global_step)
        writer.flush()
