import errno
import os

import numpy as np
import torch
from PIL import Image


def mkdir_p(path):
    """Linux mkdir -p"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_image_stack(samples, num_rows, num_columns, filename, margin=5, margin_gray_val=1., frame=0, frame_gray_val=0.0):
    """Save image stack in a tiled image"""

    # for gray scale, convert to rgb
    if len(samples.shape) == 3:
        samples = np.stack((samples,) * 3, -1)

    height = samples.shape[1]
    width = samples.shape[2]

    samples -= samples.min()
    samples /= samples.max()

    img = margin_gray_val * np.ones((height*num_rows + (num_rows-1)*margin, width*num_columns + (num_columns-1)*margin, 3))
    for h in range(num_rows):
        for w in range(num_columns):
            img[h*(height+margin):h*(height+margin)+height, w*(width+margin):w*(width+margin)+width, :] = samples[h*num_columns + w, :]

    framed_img = frame_gray_val * np.ones((img.shape[0] + 2*frame, img.shape[1] + 2*frame, 3))
    framed_img[frame:(frame+img.shape[0]), frame:(frame+img.shape[1]), :] = img

    img = Image.fromarray(np.round(framed_img * 255.).astype(np.uint8))

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    img.save(filename)
