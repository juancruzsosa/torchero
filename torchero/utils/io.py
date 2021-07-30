import os
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import requests
from tqdm import tqdm

import torchero

from zipfile import ZipFile

def download_from_url(url, dst, pbar=True, chunk_size=16*1024):
    """ Download file from a given url

    Arguments:
        url (str): url to download file
        dst (str): place to put the file
    """
    file_size = int(urlopen(url).info().get('Content-Length', -1))
    first_byte = 0
    header = {"Range": "bytes=%s-%s" % (first_byte, file_size),
              "User-Agent": "torchero/{}".format(torchero.__version__)}
    if pbar:
        progress = tqdm(
            total=file_size, initial=first_byte,
            unit='B', unit_scale=True, desc=url.split('/')[-1]
        )
    req = requests.get(url, headers=header, stream=True)
    is_path = isinstance(dst, (str, Path)) # dst is a path
    if is_path:
        dst = open(dst, 'w+b')
    for chunk in req.iter_content(chunk_size=chunk_size):
        if chunk:
            dst.write(chunk)
            if pbar:
                progress.update(chunk_size)
    if is_path:
        dst.close()
    if pbar:
        progress.close()
