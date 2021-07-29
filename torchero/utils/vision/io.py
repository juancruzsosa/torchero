import tqdm

from io import BytesIO
from functools import partial
from multiprocessing.pool import ThreadPool

from PIL import Image
from torchero.utils.io import download_from_url

def download_image(url):
    """ Download an image from an url

    Arguments:
        url (str): Url of the image

    Returns:
        The downloaded images as a PIL Image object
    """
    buffer = BytesIO()
    download_from_url(url, buffer, pbar=False)
    buffer.seek(0)
    return Image.open(buffer)

def download_images(urls, num_workers=1, pbar=True):
    """ Download multiples images

    Arguments:
        url (list of str): List of urls to download

    Returns:
        An iterator of PIL Images for the downloaded images
    """
    with ThreadPool(num_workers) as pool:
        images = pool.imap(download_image, urls)
        if pbar:
            images = tqdm.tqdm(images, total=len(urls), unit='image')
        yield from images
