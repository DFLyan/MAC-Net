#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time
import numpy as np
import logging, scipy
import tensorflow as tf
import tensorlayer as tl
import math as ma
from model_MAC_Net import *
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 16
lr_decay = 0.8
beta1 = 0.9
n_epoch = 200
decay_every = 50
ni = int(4)
ni_ = int(batch_size//4)
lr = 0.0001
block_size = 16
MR = 0.25
num_stage = 8
imagesize = block_size * block_size
size_y = ma.ceil(block_size * block_size * MR)


def train():
    save_dir_DTCS_1 = ("samples/cascade/%sstage/train/%s_g/" % (num_stage, MR)).format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_DTCS_1)
    checkpoint_dir = "checkpoint/cascade/%sstage/%s" % (num_stage, MR)  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    train_hr_img_list = sorted(tl.files.load_file_list(path='/data/91_crop96/', regx='.*.png', printable=False))

    t_target_image = tf.placeholder('float32', [batch_size, 96, 96, 1], name='t_target_image')
    t_block_image = tf.placeholder('float32', [batch_size, 16, 16, 1], name='t_block_image')
    y1_image = tf.placeholder('float32', [batch_size, 6, 6, size_y], name='y1_image')

    if not os.path.isfile("phi/nature/Gaussian%s_16.npy" % MR):
        A = np.random.normal(loc=0, scale=(1/size_y), size=[imagesize, int(size_y)])
        A = A.astype(np.float32)
        np.save("phi/nature/Gaussian%s_16.npy" % MR, A)
    else:
        A = np.load("phi/nature/Gaussian%s_16.npy" % MR, encoding='latin1')

    x_image = tf.reshape(t_block_image, [batch_size, imagesize])
    y_meas = tf.matmul(x_image, A)

    y1, y2, y3, y4, y5, y6, y7, y8,\
    r1, r2, r3, r4, r5, r6, r7, r8,\
    x1, x2, x3, x4, x5, r6, r7, \
    t = cascade(y1_image, phi=A, is_train=True, reuse=False)

    zeros_target = np.zeros_like(y1)
    y_loss = 0.01 * tl.cost.absolute_difference_error(y1, zeros_target, is_mean=True) + \
             0.1 * tl.cost.absolute_difference_error(y2, zeros_target, is_mean=True) + \
             0.1 * tl.cost.absolute_difference_error(y3, zeros_target, is_mean=True) + \
             0.1 * tl.cost.absolute_difference_error(y4, zeros_target, is_mean=True) + \
             0.1 * tl.cost.absolute_difference_error(y5, zeros_target, is_mean=True) + \
             0.1 * tl.cost.absolute_difference_error(y6, zeros_target, is_mean=True) + \
             0.1 * tl.cost.absolute_difference_error(y7, zeros_target, is_mean=True) + \
             1 * tl.cost.absolute_difference_error(y8, zeros_target, is_mean=True)

    ade_loss_4 = tl.cost.absolute_difference_error(t.outputs, t_target_image, is_mean=True)

    ade_loss = ade_loss_4 + 0.1 * y_loss

    tf.summary.scalar('reconstruction_loss', ade_loss_4)
    tf.summary.scalar('measurement_loss', y_loss)

    t1_vars = tl.layers.get_variables_with_name('cascade', True, True)

    with tf.variable_scope('learning_rate'):
            lr_v_1 = tf.Variable(lr, trainable=False)

    t1_optim = tf.train.AdamOptimizer(lr_v_1).minimize(ade_loss, var_list=t1_vars)

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(
        config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options))

    merged = tf.summary.merge_all()
    writer_1 = tf.summary.FileWriter("logs/cascade/%sblock/c%s" % (num_stage, MR), tf.get_default_graph())

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
        for idx in range(0, int(len(train_hr_img_list)//batch_size)):
            step_time = time.time()
            b_imgs_list = train_hr_img_list[idx * batch_size: (idx + 1) * batch_size]
            b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_gray_imgs_fn, path='/home/cjw/data/91_crop96/')
            b_imgs = tl.prepro.threading_data(b_imgs, fn=norm)
            b_imgs = tl.prepro.threading_data(b_imgs, fn=augm)

            size = b_imgs.shape
            a = int(ma.ceil(size[1] / block_size))
            global y_fullimg
            y_fullimg = np.zeros((batch_size, a, a, size_y))
            for num_r in range(1, int(ma.ceil(size[1] / block_size)) + 1):
                for num_c in range(1, int(ma.ceil(size[2] / block_size)) + 1):
                    img_block = b_imgs[:, (num_r - 1) * block_size:num_r * block_size,
                                (num_c - 1) * block_size:num_c * block_size]
                    img_block = np.reshape(img_block, [batch_size, block_size, block_size, 1])
                    y_meas_ = sess.run(y_meas, feed_dict={t_block_image: img_block})
                    y_meas_ = np.reshape(y_meas_, [batch_size, 1, 1, size_y])
                    y_fullimg[:, (num_r - 1):num_r, (num_c - 1):num_c, :] = y_meas_

            b_imgs = np.reshape(b_imgs, [batch_size, size[1], size[2], 1])
            errt, erry, _, summary = sess.run(
                [ade_loss, y_loss, t1_optim, merged],
                {y1_image: y_fullimg, t_target_image: b_imgs})


            if n_iter_DTCS % 100 == 0:
                print(
                    "Epoch1 [%2d/%2d] %4d time:%4.4fs,ade_loss: %.4f(y_loss: %.4f)" % (
                    epoch, n_epoch, n_iter_DTCS, time.time() - step_time, errt, erry))

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
