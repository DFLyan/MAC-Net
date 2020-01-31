#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time
import numpy as np
import logging
import scipy.io as sio
import tensorflow as tf
import tensorlayer as tl
import math as ma
from model_MAC_Net import *
from utils import *
from skimage.measure import compare_psnr
import skimage.io as ski


batch_size = 1
beta1 = 0.9
ni = int(4)
ni_ = int(batch_size//4)
block_size = 16
MR = 0.3
imagesize = block_size * block_size
size_y = ma.ceil(block_size * block_size * MR)

def train():
    save_dir = ("samples/cascade_mri/test/%s_g/" % (MR)).format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint/cascade_mri/%s" % (MR)  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    y1_image = tf.placeholder('complex64', [1, 256, 256, 1], name='y1_image')

    print('[*] load data ... ')
    testing_data_path = os.path.join('data', 'MICCAI13_SegChallenge', 'testing.pickle')

    with open(testing_data_path, 'rb') as f:
        X_test = pickle.load(f)

    print('X_test shape/min/max: ', X_test.shape, X_test.min(), X_test.max())

    n_test_examples = len(X_test)

    mask = sio.loadmat(
                os.path.join(os.path.join('phi', 'mri', 'mask', 'Gaussian1D'), "GaussianDistribution1DMask_{}.mat".format(int(MR*100))))[
                'maskRS1']
    mask = np.fft.ifftshift(mask)
    ski.imsave('mask1D%s.png' % MR, mask*255, as_gray=True)
    ski.imshow(mask)
    ski.show()

    y1, y2, y3, y4, \
    r1, r2, r3, r4, \
    x1, x2, x3, t = cascade_MRI(y1_image, phi=mask, is_train=False, reuse=False)

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(
        config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options))

    sess.run(tf.global_variables_initializer())
    tl.files.load_and_assign_npz(
            sess=sess, name=checkpoint_dir + '/g_{}_1.npz'.format(tl.global_flag['mode']), network=t)

###============================= TRAINING ===============================###
    total_psnr, n_iter_DTCS = 0, 0
    for idex in range(0, len(X_test)):
        X_good = X_test[idex]
        X_good = np.expand_dims(X_good, axis=0)
        print(np.max(X_good))
        X_bad = threading_data(X_good, fn=to_bad_img, mask=mask)
        X_bad = np.expand_dims(X_bad, -1)

        img = sess.run(t.outputs, feed_dict={y1_image: X_bad})

        # if idex == 46:
        #     img1, img2, img3 = sess.run([r1.outputs, r2.outputs, r3.outputs],
        #                                 feed_dict={y1_image: X_bad})
        #     img1 = np.array(img1)
        #     img1 = np.squeeze(img1)
        #     save_image(img1,
        #                save_dir + '/%s_genr1.png' % (idex))
        #     img2 = np.array(img2)
        #     img2 = np.squeeze(img2)
        #     save_image(img2,
        #                save_dir + '/%s_genr2.png' % (idex))
        #     img3 = np.array(img3)
        #     img3 = np.squeeze(img3)
        #     save_image(img3,
        #                save_dir + '/%s_genr3.png' % (idex))

        img = np.array(img)
        img = np.squeeze(img)
        print(np.max(img))
        print(np.min(img))
        psnr = compare_psnr(np.squeeze(X_test[idex]), img, data_range=2)
        if idex % 50 == 0:
            save_image(img,
                       save_dir + '/%s_%s_gen.png' % (idex, psnr))
            X_hr = np.squeeze(X_test[idex])
            save_image(X_hr,
                       save_dir + '/%s_hr.png' % (idex))

        total_psnr += psnr
        n_iter_DTCS += 1
    print("average_psnr:%.4f" % (total_psnr/len(X_test)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='GRAY', help='GRAY, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'GRAY':
        train()
    else:
        raise Exception("Unknow --mode")
