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

logging.getLogger().setLevel(logging.INFO)

batch_size = 4
lr_decay = 0.8
beta1 = 0.9
n_epoch = 60
decay_every = 5
ni = int(4)
ni_ = int(batch_size//4)
lr = 0.0001
block_size = 16
MR = 0.3
imagesize = block_size * block_size
size_y = ma.ceil(block_size * block_size * MR)
y_weight = 0.0001


def train():
    save_dir_DTCS_1 = ("samples/cascade_mri/train/%s_g/" % (MR)).format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_DTCS_1)
    checkpoint_dir = "checkpoint/cascade_mri/%s" % (MR)
    tl.files.exists_or_mkdir(checkpoint_dir)

    t_target_image = tf.placeholder('float32', [batch_size, 256, 256, 1], name='t_target_image')
    y1_image = tf.placeholder('complex64', [batch_size, 256, 256, 1], name='y1_image')

    print('[*] load data ... ')
    training_data_path = os.path.join('data', 'MICCAI13_SegChallenge', 'training.pickle')

    with open(training_data_path, 'rb') as f:
        X_train = pickle.load(f)
    print('X_train shape/min/max: ', X_train.shape, X_train.min(), X_train.max())
    n_training_examples = X_train.shape[0]
    train_hr_img_list = np.arange(0, n_training_examples, 1).tolist()
    n_step_epoch = round(n_training_examples / batch_size)

    mask = sio.loadmat(
                os.path.join(os.path.join('phi','mri', 'mask', 'Gaussian1D'), "GaussianDistribution1DMask_{}.mat".format(int(MR*100))))[
                'maskRS1']
    mask = np.fft.ifftshift(mask)

    y1, y2, y3, y4,\
    r1, r2, r3, r4,\
    x1, x2, x3, t = cascade_MRI(y1_image, phi=mask, is_train=True, reuse=False)

    zeros_target = np.zeros_like(y1)
    y_loss = 0.01 * tl.cost.absolute_difference_error(y1, zeros_target, is_mean=True) + \
             0.1 * tl.cost.absolute_difference_error(y2, zeros_target, is_mean=True) + \
             0.1 * tl.cost.absolute_difference_error(y3, zeros_target, is_mean=True) + \
             1 * tl.cost.absolute_difference_error(y4, zeros_target, is_mean=True)

    ade_loss_4 = tl.cost.absolute_difference_error(t.outputs, t_target_image, is_mean=True)
    ade_loss = ade_loss_4 + y_weight * y_loss

    tf.summary.scalar('ade_loss', ade_loss)

    t1_vars = tl.layers.get_variables_with_name('cascade', True, True)

    with tf.variable_scope('learning_rate'):
            lr_v_1 = tf.Variable(lr, trainable=False)

    t1_optim = tf.train.AdamOptimizer(lr_v_1).minimize(ade_loss, var_list=t1_vars)

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(
        config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options))

    merged = tf.summary.merge_all()
    writer_1 = tf.summary.FileWriter("logs/cascade/c%s" % (MR), tf.get_default_graph())

    sess.run(tf.global_variables_initializer())
    tl.files.load_and_assign_npz(
            sess=sess, name=checkpoint_dir + '/g_{}_1.npz'.format(tl.global_flag['mode']), network=t)


###============================= TRAINING ===============================###
    for epoch in range(0, n_epoch):
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v_1, lr * new_lr_decay))
            log = " ** new learning rate: %f " % (lr * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v_1, lr))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f " % (lr, decay_every, lr_decay)
            print(log)

        epoch_time = time.time()
        total_ade_loss, n_iter_DTCS = 0, 0

        if epoch == 0:
            global sum1
            sum1 = 0
        else:
            pass

        random.shuffle(train_hr_img_list)
        for idx in range(0, n_step_epoch):
            step_time = time.time()
            idex = train_hr_img_list[idx * batch_size: (idx + 1) * batch_size]
            X_good = X_train[idex]
            X_good = tl.prepro.threading_data(X_good, fn=augm)
            X_bad = threading_data(X_good, fn=to_bad_img, mask=mask)
            X_bad = np.expand_dims(X_bad, -1)
            errt, erry, img, _, summary = sess.run(
                [ade_loss, y_loss, t.outputs, t1_optim, merged],
                {y1_image: X_bad, t_target_image: X_good})

            if n_iter_DTCS % 1000 == 0:
                print(
                    "Epoch1 [%2d/%2d] %4d time:%4.4fs,ade_loss: %.4f(y_loss: %.4f)" % (
                    epoch, n_epoch, n_iter_DTCS, time.time() - step_time, errt, erry*y_weight))
                print(np.min(img), np.max(img))

            if sum1 % 50 == 0:
                writer_1.add_summary(summary, sum1)

            total_ade_loss += errt
            n_iter_DTCS += 1
            sum1 += 1

        log = "[*] Epoch1: [%2d/%2d] time: %4.4fs, ade_loss:%.8f" % (
        epoch, n_epoch, time.time() - epoch_time, total_ade_loss / n_iter_DTCS)
        print(log)

        ## save model
        if (epoch != 0) and (epoch % 1 == 0):
            tl.files.save_npz(t.all_params, name=checkpoint_dir + '/g_{}_1.npz'.format(tl.global_flag['mode']), sess=sess)

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
