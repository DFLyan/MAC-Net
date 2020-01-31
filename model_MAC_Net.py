import tensorflow as tf
import tensorlayer as tl
import numpy as np
import math as ma
from utils import *
from tensorlayer.layers import *

MR = 0.25
num_stage = 8
block_size = 16
imagesize = block_size * block_size
size_y = int(ma.ceil(block_size * block_size * MR))


def cascade(y, phi, is_train=False, reuse=False):
    name = globals()
    j = 1
    with tf.variable_scope("cascade", reuse=reuse) as vs:
        y = tl.layers.InputLayer(y, name='1/y')
        y, n = building_block1(y, phi=phi, is_train=is_train, stage_num=1)
        name['y1'] = y
        name['x1'] = n
        name['r1'] = n
        x = n
        x_c = n
        for i in range(2, num_stage + 1):
            y = tl.layers.InputLayer(y, name='%s/y' % i)
            y, n = building_blockn(y, x_c=x_c, phi=phi, is_train=is_train, stage_num=i)
            x = tl.layers.ElementwiseLayer([x, n], combine_fn=tf.add, act=tl.act.hard_tanh, name='%s/add' % i)
            x_c = tl.layers.ConcatLayer([x_c, n], concat_dim=3, name='%s/concat' % i)
            if i % 1 == 0:
                name['y' + str(i)] = y
                name['r' + str(i)] = n
                # name['spa' + str(i)] = spa
                name['x' + str(i)] = x
        return x
        # return name.get('y1'), name.get('y2'), name.get('y3'), name.get('y4'), name.get('y5'), \
        #        name.get('y6'), name.get('y7'), name.get('y8'), name.get('y9'), \
        #        name.get('r1'), name.get('r2'), name.get('r3'), name.get('r4'), name.get('r5'), \
        #        name.get('r6'), name.get('r7'), name.get('r8'), name.get('r9'), \
        #        name.get('x1'), name.get('x2'), name.get('x3'), name.get('x4'), \
        #        name.get('x5'), name.get('x6'), name.get('x7'), name.get('x8'), x

def building_block1(y, phi, is_train=False, stage_num=1):
    g_init = tf.random_normal_initializer(1., 0.02)
    n = Conv2d(y, 256, (5, 5), (1, 1), act=tf.nn.selu, padding='SAME', name='%s/init_all/1' % stage_num)

    temp1 = n
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/init1/b1' % stage_num)
    n = Conv2d(n, 64, (1, 1), (1, 1), act=None, padding='SAME', name='%s/init1/1' % stage_num)
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/init1/b2' % stage_num)
    n = Conv2d(n, 64, (5, 5), (1, 1), act=None, padding='SAME', name='%s/init1/2' % stage_num)
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/init1/b3' % stage_num)
    n = Conv2d(n, 256, (1, 1), (1, 1), act=None, padding='SAME', name='%s/init1/3' % stage_num)
    n = tl.layers.ElementwiseLayer([temp1, n], combine_fn=tf.add, act=tf.nn.selu, name='%s/init1/block_add' % stage_num)
    n = SubpixelConv2d(n, scale=4, n_out_channel=None, act=tf.nn.selu, name='%s/init1/pixelshufflerx16/1' % stage_num)
    n = Conv2d(n, 256, (5, 5), (1, 1), act=tf.nn.selu, padding='SAME', name='%s/init1/4' % stage_num)

    temp2 = n
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/init2/b1' % stage_num)
    n = Conv2d(n, 64, (1, 1), (1, 1), act=None, padding='SAME', name='%s/init2/1' % stage_num)
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/init2/b2' % stage_num)
    n = Conv2d(n, 64, (5, 5), (1, 1), act=None, padding='SAME', name='%s/init2/2' % stage_num)
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/init2/b3' % stage_num)
    n = Conv2d(n, 256, (1, 1), (1, 1), act=None, padding='SAME', name='%s/init2/3' % stage_num)
    n = tl.layers.ElementwiseLayer([temp2, n], combine_fn=tf.add, act=tf.nn.selu, name='%s/init2/block_add' % stage_num)
    n = SubpixelConv2d(n, scale=4, n_out_channel=None, act=tf.nn.selu, name='%s/init2/pixelshufflerx16/1' % stage_num)
    n = Conv2d(n, 64, (5, 5), (1, 1), act=tf.nn.selu, padding='SAME', name='%s/init2/4' % stage_num)

    tem1 = n
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/core1/b1' % stage_num)
    n = Conv2d(n, 16, (1, 1), (1, 1), act=None, padding='SAME', name='%s/core1/1' % stage_num)
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/core1/b2' % stage_num)
    n = Conv2d(n, 16, (5, 5), (1, 1), act=None, padding='SAME', name='%s/core1/2' % stage_num)
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/core1/b3' % stage_num)
    n = Conv2d(n, 64, (1, 1), (1, 1), act=None, padding='SAME', name='%s/core1/3' % stage_num)
    n = tl.layers.ElementwiseLayer([tem1, n], combine_fn=tf.add, act=tf.nn.selu, name='%s/block_add1' % stage_num)
    tem2 = n
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/core2/b1' % stage_num)
    n = Conv2d(n, 16, (1, 1), (1, 1), act=None, padding='SAME', name='%s/core2/1' % stage_num)
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/core2/b2' % stage_num)
    n = Conv2d(n, 16, (5, 5), (1, 1), act=None, padding='SAME', name='%s/core2/2' % stage_num)
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/core2/b3' % stage_num)
    n = Conv2d(n, 64, (1, 1), (1, 1), act=None, padding='SAME', name='%s/core2/3' % stage_num)
    n = tl.layers.ElementwiseLayer([tem2, n], combine_fn=tf.add, act=tf.nn.selu, name='%s/block_add2' % stage_num)
    n = tl.layers.ElementwiseLayer([tem1, n], combine_fn=tf.add, act=tf.nn.selu, name='%s/block_add3' % stage_num)
    n = Conv2d(n, 1, (5, 5), (1, 1), act=tl.act.hard_tanh, padding='SAME', name='%s/n128/6' % stage_num)

    size = n.outputs.get_shape().as_list()
    I = n.outputs
    X = tf.split(I, int(size[1] / block_size), 1)
    Y = []
    for x in X:
        X_ = tf.split(x, int(size[2] / block_size), 2)
        Y_ = []
        for x_ in X_:
            x_ = tf.reshape(x_, [size[0], imagesize])
            y_meas_ = tf.matmul(x_, phi)
            y_meas_ = tf.reshape(y_meas_, [size[0], 1, 1, size_y])
            Y_.append(y_meas_)
        y_meas_c = tf.concat([y_ for y_ in Y_], 2)
        Y.append(y_meas_c)
    y_fullimg = tf.concat([y for y in Y], 1)
    y_res_fullimg = y.outputs - y_fullimg
    return y_res_fullimg, n

def building_blockn(y, x_c, phi, is_train=False, stage_num=1):
    g_init = tf.random_normal_initializer(1., 0.02)

    x_c = Conv2d(x_c, 64, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', name='%s/x_c/1' % stage_num)
    n = Conv2d(y, 256, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', name='%s/init_all/1' % stage_num)

    temp1 = n
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/init1/b1' % stage_num)
    n = Conv2d(n, 64, (1, 1), (1, 1), act=None, padding='SAME', name='%s/init1/1' % stage_num)
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/init1/b2' % stage_num)
    n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', name='%s/init1/2' % stage_num)
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/init1/b3' % stage_num)
    n = Conv2d(n, 256, (1, 1), (1, 1), act=None, padding='SAME', name='%s/init1/3' % stage_num)
    n = tl.layers.ElementwiseLayer([temp1, n], combine_fn=tf.add, act=tf.nn.selu, name='%s/init1/block_add' % stage_num)
    n = SubpixelConv2d(n, scale=4, n_out_channel=None, act=tf.nn.selu, name='%s/init1/pixelshufflerx16/1' % stage_num)
    n = Conv2d(n, 256, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', name='%s/init1/4' % stage_num)

    temp2 = n
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/init2/b1' % stage_num)
    n = Conv2d(n, 64, (1, 1), (1, 1), act=None, padding='SAME', name='%s/init2/1' % stage_num)
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/init2/b2' % stage_num)
    n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', name='%s/init2/2' % stage_num)
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/init2/b3' % stage_num)
    n = Conv2d(n, 256, (1, 1), (1, 1), act=None, padding='SAME', name='%s/init2/3' % stage_num)
    n = tl.layers.ElementwiseLayer([temp2, n], combine_fn=tf.add, act=tf.nn.selu, name='%s/init2/block_add' % stage_num)
    n = SubpixelConv2d(n, scale=4, n_out_channel=None, act=tf.nn.selu, name='%s/init2/pixelshufflerx16/1' % stage_num)
    n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', name='%s/init2/4' % stage_num)

    n_a = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', name='%s/3Dattention/1' % stage_num)
    n_a = Conv2d(n_a, 64, (3, 3), (1, 1), act=tf.nn.softmax, padding='SAME', name='%s/3Dattention/2' % stage_num)

    x_c = tl.layers.ElementwiseLayer([x_c, n_a], combine_fn=tf.multiply, act=tf.nn.selu, name='%s/attention/spatial' % stage_num)

    n = tl.layers.ConcatLayer([n, x_c], concat_dim=3, name='%s/concat' % stage_num)
    n = Conv2d(n, 64, (1, 1), (1, 1), act=tf.nn.selu, padding='SAME', name='%s/fusion/2' % stage_num)

    tem1 = n
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/core1/b1' % stage_num)
    n = Conv2d(n, 16, (1, 1), (1, 1), act=None, padding='SAME', name='%s/core1/1' % stage_num)
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/core1/b2' % stage_num)
    n = Conv2d(n, 16, (3, 3), (1, 1), act=None, padding='SAME', name='%s/core1/2' % stage_num)
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/core1/b3' % stage_num)
    n = Conv2d(n, 64, (1, 1), (1, 1), act=None, padding='SAME', name='%s/core1/3' % stage_num)
    n = tl.layers.ElementwiseLayer([tem1, n], combine_fn=tf.add, act=tf.nn.selu, name='%s/block_add1' % stage_num)
    tem2 = n
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/core2/b1' % stage_num)
    n = Conv2d(n, 16, (1, 1), (1, 1), act=None, padding='SAME', name='%s/core2/1' % stage_num)
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/core2/b2' % stage_num)
    n = Conv2d(n, 16, (3, 3), (1, 1), act=None, padding='SAME', name='%s/core2/2' % stage_num)
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/core2/b3' % stage_num)
    n = Conv2d(n, 64, (1, 1), (1, 1), act=None, padding='SAME', name='%s/core2/3' % stage_num)
    n = tl.layers.ElementwiseLayer([tem2, n], combine_fn=tf.add, act=tf.nn.selu, name='%s/block_add2' % stage_num)
    n = tl.layers.ElementwiseLayer([tem1, n], combine_fn=tf.add, act=tf.nn.selu, name='%s/block_add3' % stage_num)
    n = Conv2d(n, 1, (3, 3), (1, 1), act=tl.act.hard_tanh, padding='SAME', name='%s/n128/6' % stage_num)

    size = n.outputs.get_shape().as_list()
    I = n.outputs
    X = tf.split(I, int(size[1] / block_size), 1)
    Y = []
    for x in X:
        X_ = tf.split(x, int(size[2] / block_size), 2)
        Y_ = []
        for x_ in X_:
            x_ = tf.reshape(x_, [size[0], imagesize])
            y_meas_ = tf.matmul(x_, phi)
            y_meas_ = tf.reshape(y_meas_, [size[0], 1, 1, size_y])
            Y_.append(y_meas_)
        y_meas_c = tf.concat([y_ for y_ in Y_], 2)
        Y.append(y_meas_c)
    y_fullimg = tf.concat([y for y in Y], 1)
    y_res_fullimg = y.outputs - y_fullimg
    return y_res_fullimg, n


def cascade_MRI(y, phi, is_train=False, reuse=False):
    name = globals()
    j = 1
    with tf.variable_scope("cascade", reuse=reuse) as vs:
        y = tl.layers.InputLayer(y, name='1/y')
        y, n = building_block1_MRI(y, mask=phi, is_train=is_train, stage_num=1)
        name['y1'] = y
        name['x1'] = n
        name['r1'] = n
        x = n
        x_c = n
        for i in range(2, num_stage + 1):
            y = tl.layers.InputLayer(y, name='%s/y' % i)
            y, n = building_blockn_MRI(y, x_c=x_c, mask=phi, is_train=is_train, stage_num=i)
            x = tl.layers.ElementwiseLayer([x, n], combine_fn=tf.add, act=tl.act.hard_tanh, name='%s/add' % i)
            x_c = tl.layers.ConcatLayer([x_c, n], concat_dim=3, name='%s/concat' % i)
            if i % 1 == 0:
                name['y' + str(i)] = y
                name['r' + str(i)] = n
                name['x' + str(i)] = x
        return name.get('y1'), name.get('y2'), name.get('y3'), name.get('y4'),\
               name.get('r1'), name.get('r2'), name.get('r3'), name.get('r4'),\
               name.get('x1'), name.get('x2'), name.get('x3'), x

def building_block1_MRI(y, mask, is_train=False, stage_num=1):
    g_init = tf.random_normal_initializer(1., 0.02)

    n = tf_fft_to_img(y.outputs)
    n = InputLayer(n, name='%s/block_input' % stage_num)

    n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', name='%s/init_all/1' % stage_num)

    tem1 = n
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/core1/b1' % stage_num)
    n = Conv2d(n, 16, (1, 1), (1, 1), act=None, padding='SAME', name='%s/core1/1' % stage_num)
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/core1/b2' % stage_num)
    n = Conv2d(n, 16, (3, 3), (1, 1), act=None, padding='SAME', name='%s/core1/2' % stage_num)
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/core1/b3' % stage_num)
    n = Conv2d(n, 64, (1, 1), (1, 1), act=None, padding='SAME', name='%s/core1/3' % stage_num)
    n = tl.layers.ElementwiseLayer([tem1, n], combine_fn=tf.add, act=tf.nn.selu, name='%s/block_add1' % stage_num)
    tem2 = n
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/core2/b1' % stage_num)
    n = Conv2d(n, 16, (1, 1), (1, 1), act=None, padding='SAME', name='%s/core2/1' % stage_num)
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/core2/b2' % stage_num)
    n = Conv2d(n, 16, (3, 3), (1, 1), act=None, padding='SAME', name='%s/core2/2' % stage_num)
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/core2/b3' % stage_num)
    n = Conv2d(n, 64, (1, 1), (1, 1), act=None, padding='SAME', name='%s/core2/3' % stage_num)
    n = tl.layers.ElementwiseLayer([tem2, n], combine_fn=tf.add, act=tf.nn.selu, name='%s/block_add2' % stage_num)
    n = tl.layers.ElementwiseLayer([tem1, n], combine_fn=tf.add, act=tf.nn.selu, name='%s/block_add3' % stage_num)
    n = Conv2d(n, 1, (3, 3), (1, 1), act=tl.act.hard_tanh, padding='SAME', name='%s/n128/6' % stage_num)

    y_local = tf_img_to_fft(n.outputs, mask)
    y_local = tf.expand_dims(y_local, -1)
    y_res = y.outputs - y_local

    return y_res, n

def building_blockn_MRI(y, x_c, mask, is_train=False, stage_num=1):
    g_init = tf.random_normal_initializer(1., 0.02)

    x_c = Conv2d(x_c, 64, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', name='%s/x_c/1' % stage_num)
    n = tf_fft_to_img(y.outputs)
    n = InputLayer(n, name='%s/block_input' % stage_num)
    n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', name='%s/init_all/1' % stage_num)
    n_a = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', name='%s/attention/1' % stage_num)
    n_a = Conv2d(n_a, 64, (3, 3), (1, 1), act=tf.nn.softmax, padding='SAME', name='%s/attention/2' % stage_num)

    x_c = tl.layers.ElementwiseLayer([x_c, n_a], combine_fn=tf.multiply, act=tf.nn.selu, name='%s/attention' % stage_num)

    n = tl.layers.ConcatLayer([n, x_c], concat_dim=3, name='%s/concat' % stage_num)
    n = Conv2d(n, 64, (1, 1), (1, 1), act=tf.nn.selu, padding='SAME', name='%s/fusion/2' % stage_num)

    tem1 = n
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/core1/b1' % stage_num)
    n = Conv2d(n, 16, (1, 1), (1, 1), act=None, padding='SAME', name='%s/core1/1' % stage_num)
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/core1/b2' % stage_num)
    n = Conv2d(n, 16, (3, 3), (1, 1), act=None, padding='SAME', name='%s/core1/2' % stage_num)
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/core1/b3' % stage_num)
    n = Conv2d(n, 64, (1, 1), (1, 1), act=None, padding='SAME', name='%s/core1/3' % stage_num)
    n = tl.layers.ElementwiseLayer([tem1, n], combine_fn=tf.add, act=tf.nn.selu, name='%s/block_add1' % stage_num)
    tem2 = n
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/core2/b1' % stage_num)
    n = Conv2d(n, 16, (1, 1), (1, 1), act=None, padding='SAME', name='%s/core2/1' % stage_num)
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/core2/b2' % stage_num)
    n = Conv2d(n, 16, (3, 3), (1, 1), act=None, padding='SAME', name='%s/core2/2' % stage_num)
    n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='%s/core2/b3' % stage_num)
    n = Conv2d(n, 64, (1, 1), (1, 1), act=None, padding='SAME', name='%s/core2/3' % stage_num)
    n = tl.layers.ElementwiseLayer([tem2, n], combine_fn=tf.add, act=tf.nn.selu, name='%s/block_add2' % stage_num)
    n = tl.layers.ElementwiseLayer([tem1, n], combine_fn=tf.add, act=tf.nn.selu, name='%s/block_add3' % stage_num)
    n = Conv2d(n, 1, (3, 3), (1, 1), act=tl.act.hard_tanh, padding='SAME', name='%s/output' % stage_num)

    y_local = tf_img_to_fft(n.outputs, mask)
    y_local = tf.expand_dims(y_local, -1)
    y_res = y.outputs - y_local

    return y_res, n
