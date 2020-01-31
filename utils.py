import os

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
import random
import cv2

import scipy
import numpy as np
from functools import reduce


def get_gray_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='L')   #if GRAY images,then mode = 'L',and change the code of tensorlayer.visualize.save_image

def save_image(image, path):
    """Save an image as a png file."""
    min_val = image.min()
    if min_val < 0:
        image = image + min_val

    # image = (image.squeeze() * 1.0 / image.max()) * 255
    # image = image.astype(np.uint8)

    scipy.misc.imsave(path, image)
    print('[#] Image saved {}.'.format(path))


def norm(x):
    x = x / (255. / 2.)
    x = x - 1.
    return x


def inv_norm(x):
    x = x + 1
    x = x * (255. / 2.)
    # x = x * 255
    return x


def augm(x):
    size = x.shape
    x = tl.prepro.flip_axis(x, axis=0, is_random=True)
    x = tl.prepro.flip_axis(x, axis=1, is_random=True)
    x = np.reshape(x, (size[0], size[1], 1))
    rg = random.sample([0, 90, 180, 270], 1)
    rg = rg[0]
    x = tl.prepro.rotation(x, rg=rg, is_random=False)
    return x


def to_bad_img(x, mask):
    x = (x + 1.) / 2.
    fft = np.fft.fft2(x)
    fft = np.squeeze(fft)
    fft = np.multiply(fft, mask)
    return fft


def tf_img_to_fft(x, mask):
    x = (x + 1.) / 2
    x = tf.complex(x, tf.zeros_like(x))
    fft = tf.fft2d(x)
    fft = tf.squeeze(fft)
    fft = tf.multiply(fft, mask)
    return fft


def tf_fft_to_img(fft):
    fft = tf.ifft2d(fft)
    x = tf.abs(fft)
    x = x * 2 - 1
    return x
