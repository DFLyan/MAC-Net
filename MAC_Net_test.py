#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time
import numpy as np
from skimage.measure import compare_ssim as ssim_c
import tensorflow as tf
import tensorlayer as tl
import math as ma
from model_MAC_Net import *
from utils import *
from skimage.measure import compare_psnr

block_size = 16
MR = 0.25
num_stage = 8
imagesize = block_size * block_size
size_y = ma.ceil(block_size * block_size * MR)
test_path = "/data/SET11/"

def read_all_imgs(img_list, path='', n_threads=32):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_gray_imgs_fn, path=path)
        # print(b_img16s.shape)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs



def evaluate():
    save_dir = ("samples/cascade/%sstage/test/%s_g" % (num_stage, MR)).format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint/cascade/%sstage/%s" % (num_stage, MR)

    name = globals()

    ###====================== PRE-LOAD DATA ===========================###
    test_hr_img_list = sorted(tl.files.load_file_list(path=test_path, regx='.*.*', printable=False))
    test_hr_imgs = read_all_imgs(test_hr_img_list, path=test_path, n_threads=32)

    ###========================== DEFINE MODEL ============================###
    # x_init = tf.placeholder('float32', [1, None, None, 1], name='a')
    if test_path == "/data/BSD68/":
        y1_image = tf.placeholder('float32', [1, 31, 21, size_y], name='y1_image')
    elif test_path == "/data/SET11/":
        y1_image = tf.placeholder('float32', [1, 16, 16, size_y], name='y1_image')
        y2_image = tf.placeholder('float32', [1, 32, 32, size_y], name='y2_image')

    t_block_image = tf.placeholder('float32', [1, block_size, block_size, 1], name='t_block_image')

    A = np.load("phi/nature/Gaussian%s_16.npy" % MR, encoding='latin1')

    x_image = tf.reshape(t_block_image, [1, imagesize])
    y_meas = tf.matmul(x_image, A)


    y1, y2, y3, y4, y5, y6, y7, y8, \
    name['res1_'], name['res2_'], name['res3_'], \
    name['res4_'], name['res5_'], name['res6_'], \
    name['res7_'], name['res8_'], \
    name['x1'], name['x2'], name['x3'], \
    name['x4'], name['x5'], name['x6'], \
    name['x7'], \
    t_, \
     = cascade(y1_image, phi=A, is_train=False, reuse=False)

    if test_path == "/home/cjw/data/SET11/":
        y1_2, y2_2, y3_2, y4_2, y5_2, y6_2, y7_2, y8_2, \
        name['res1_2'], name['res2_2'], name['res3_2'], \
        name['res4_2'], name['res5_2'], name['res6_2'], \
        name['res7_2'], name['res8_2'], \
        name['x1_2'], name['x2_2'], name['x3_2'], \
        name['x4_2'], name['x5_2'], name['x6_2'], \
        name['x7_2'], \
        t_2, \
            = cascade(y2_image, phi=A, is_train=False, reuse=True)


    ###========================== RESTORE G =============================###
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(
        config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options))
    tl.layers.initialize_global_variables(sess)

    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_GRAY_1.npz', network=t_)

    ###======================= EVALUATION =============================###
    global sum, sum_s, sum_t
    sum = 0
    sum_s = 0
    sum_t = 0
    f1 = open("samples/cascade/%sblock/test/%s" % (num_stage, MR), "w")
    for imid in range(0, len(test_hr_imgs)):
        is_trans = False
        b_imgs_ = tl.prepro.threading_data(test_hr_imgs[imid:imid+1], fn=norm)
        size = b_imgs_.shape

        b_imgs_ = np.reshape(b_imgs_, [size[1], size[2]])

        if size[1] == 321:
            b_imgs_ = np.transpose(b_imgs_)
            is_trans = True
            temp_size = [1, size[2], size[1]]
            size = temp_size
        a = int(ma.ceil(size[1] / block_size))
        b = int(ma.ceil(size[2] / block_size))


        row_pad = block_size - size[1]%block_size
        col_pad = block_size - size[2]%block_size

        if (size[1] % block_size) & (size[2] % block_size) == 0:
            im_pad = b_imgs_
        else:
            im_pad = np.concatenate((b_imgs_, np.zeros((size[1], col_pad))), axis=1)
            im_pad = np.concatenate((im_pad, np.zeros((row_pad, size[2]+col_pad))), axis=0)

        size_p = im_pad.shape
        noise_img = im_pad

        global y_full
        y_full = np.zeros((1, a, b, size_y))
        for num_r in range(1, int(ma.ceil(size[1] / block_size)) + 1):
            for num_c in range(1, int(ma.ceil(size[2] / block_size)) + 1):
                img_block = noise_img[(num_r - 1) * block_size:num_r * block_size,
                            (num_c - 1) * block_size:num_c * block_size]
                img_block = np.reshape(img_block, [-1, block_size, block_size, 1])
                y_meas_ = sess.run(y_meas, feed_dict={t_block_image: img_block})
                y_meas_ = np.reshape(y_meas_, [-1, 1, 1, size_y])
                noise = np.random.normal(loc=0, scale=0, size=[1, 1, 1, size_y])
                y_meas_ = y_meas_ + noise
                y_full[:, (num_r - 1):num_r, (num_c - 1):num_c, :] = y_meas_

        y_fullimg = y_full
        print(y_fullimg.shape)
        start_time = time.time()

        '''choose one image to show the y, images of every stages and reconstruction image'''
        # if imid == 0:
        #     if config.TEST.hr_img_path == "/data/BSD68/":
        #         y1_, y2_, y3_, y4_, y5_, y6_, y7_, y8_,\
        #         name['res_block1'], name['res_block2'], name['res_block3'], name['res_block4'], \
        #         name['res_block5'], name['res_block6'], name['res_block7'], name['res_block8'], \
        #             = sess.run([y1, y2, y3, y4, y5, y6, y7, y8,
        #                         name.get('res1_').outputs, name.get('res2_').outputs, name.get('res3_').outputs,
        #                         name.get('res4_').outputs, name.get('res5_').outputs, name.get('res6_').outputs,
        #                         name.get('res7_').outputs, name.get('res8_').outputs], feed_dict={y1_image: y_fullimg})
        #     elif config.TEST.hr_img_path == "/data/SET11/":
        #         if size[1] == 256:
        #             y1_, y2_, y3_, y4_, y5_, y6_, y7_, y8_, \
        #             name['res_block1'], name['res_block2'], name['res_block3'], name['res_block4'], \
        #             name['res_block5'], name['res_block6'], name['res_block7'], name['res_block8'], \
        #                 = sess.run([y1, y2, y3, y4, y5, y6, y7, y8,
        #                             name.get('res1_').outputs, name.get('res2_').outputs, name.get('res3_').outputs,
        #                             name.get('res4_').outputs, name.get('res5_').outputs, name.get('res6_').outputs,
        #                             name.get('res7_').outputs, name.get('res8_').outputs], feed_dict={y1_image: y_fullimg})
        #         else:
        #             y1_, y2_, y3_, y4_, y5_, y6_, y7_, y8_, \
        #             name['res_block1'], name['res_block2'], name['res_block3'], name['res_block4'], \
        #             name['res_block5'], name['res_block6'], name['res_block7'], name['res_block8'], \
        #                 = sess.run([y1_2, y2_2, y3_2, y4_2, y5_2, y6_2, y7_2, y8_2,
        #                             name.get('res1_2').outputs, name.get('res2_2').outputs, name.get('res3_2').outputs,
        #                             name.get('res4_2').outputs, name.get('res5_2').outputs, name.get('res6_2').outputs,
        #                             name.get('res7_2').outputs, name.get('res8_2').outputs],
        #                            feed_dict={y1_image: y_fullimg})
        #
        #     # y_res = np.expand_dims(np.squeeze(y_fullimg[:, 1, 1, :]), axis=0)
        #     # y_res = np.concatenate((y_res, np.expand_dims(np.squeeze(y1_[:, 1, 1, :]), axis=0)), axis=0)
        #     # y_res = np.concatenate((y_res, np.expand_dims(np.squeeze(y2_[:, 1, 1, :]), axis=0)), axis=0)
        #     # y_res = np.concatenate((y_res, np.expand_dims(np.squeeze(y3_[:, 1, 1, :]), axis=0)), axis=0)
        #     # y_res = np.concatenate((y_res, np.expand_dims(np.squeeze(y4_[:, 1, 1, :]), axis=0)), axis=0)
        #     # y_res = np.concatenate((y_res, np.expand_dims(np.squeeze(y5_[:, 1, 1, :]), axis=0)), axis=0)
        #     # y_res = np.concatenate((y_res, np.expand_dims(np.squeeze(y6_[:, 1, 1, :]), axis=0)), axis=0)
        #     # y_res = np.concatenate((y_res, np.expand_dims(np.squeeze(y7_[:, 1, 1, :]), axis=0)), axis=0)
        #     # y_res = np.concatenate((y_res, np.expand_dims(np.squeeze(y8_[:, 1, 1, :]), axis=0)), axis=0)
        #     # sio.savemat(save_dir + '/y_res', {'y_res': y_res})
        #     for i in range(1, num_stage+1):
        #         name['resimg' + str(i)] = np.reshape(name.get('res_block%s' % i), [size_p[0], size_p[1]])
        #         res1 = name.get('resimg%s' % i)
        #         res1 = tl.prepro.threading_data(res1[:size[1], :size[2]], fn=inv_norm)
        #         save_image(res1.astype(np.uint8), save_dir + '/res%s.png' % i)
        #     img__ = name.get('resimg1')
        #     Y_plot = []
        #     for j in range(2, num_stage+2):
        #         img__1 = tl.prepro.threading_data(img__[:size[1], :size[2]], fn=inv_norm)
        #         psnr_add = compare_psnr(b_imgs_.astype(np.float32), img__[:size[1], :size[2]])
        #         ssim_a = ssim_c(X=b_imgs_.astype(np.float32), Y=img__[:size[1], :size[2]].astype(np.float32), multichannel=False)
        #         save_image(img__1.astype(np.uint8), save_dir + '/block_img%s_psnr%.4f_ssim%.4f.png' % ((j - 1), psnr_add, ssim_a))
        #         Y_plot.append(psnr_add)
        #         print("resimg%s's PSNR:%.8f" % (j - 1, psnr_add))
        #         if j < (num_stage+1):
        #             img__ = np.clip(img__ + name.get('resimg%s' % j), -1, 1)
        #         else:
        #             pass
        #
        #     X_plot = np.linspace(1, num_stage, num_stage)
        #     plt.plot(X_plot, Y_plot)
        #     plt.title("%sth%sMR" % (num, MR))
        #     plt.show()

        '''output images of every stages'''
        # if size[1] ==512:
        #     name['img1'], name['img2'], name['img3'], name['img4'], \
        #     name['img5'], name['img6'], name['img7'], img \
        #         = sess.run([name.get('x1_2').outputs, name.get('x2_2').outputs,
        #                     name.get('x3_2').outputs, name.get('x4_2').outputs,
        #                     name.get('x5_2').outputs, name.get('x6_2').outputs,
        #                     name.get('x7_2').outputs, t_2.outputs],
        #                    feed_dict={y2_image: y_fullimg})
        # else:
        #     name['img1'], name['img2'], name['img3'], name['img4'], \
        #     name['img5'], name['img6'], name['img7'], img \
        #         = sess.run([name.get('x1').outputs, name.get('x2').outputs,
        #                     name.get('x3').outputs, name.get('x4').outputs,
        #                     name.get('x5').outputs, name.get('x6').outputs,
        #                     name.get('x7').outputs, t_.outputs],
        #                    feed_dict={y1_image: y_fullimg})

        '''only output the reconstruction image'''
        if size[1] ==512:
            img = sess.run([t_2.outputs],
                           feed_dict={y2_image: y_fullimg})
        else:
            img = sess.run([t_.outputs],
                           feed_dict={y1_image: y_fullimg})
        print("took: %4.4fs" % (time.time() - start_time))
        sum_t += (time.time() - start_time)

        '''wirte the PSNR of every stage into file'''
        # for idx_img in range(1, num_stage):
        #     psnr1 = compare_psnr(b_imgs_.astype(np.float32), np.reshape(
        #         name.get('img%s' % idx_img), [size_p[0], size_p[1]])[:size[1], :size[2]])
        #     f1.write("%.4f " % psnr1)


        img = np.reshape(img, [size_p[0], size_p[1]])
        img = img[:size[1], :size[2]]

        psnr = compare_psnr(b_imgs_.astype(np.float32), img)

        f1.write("%.4f " % psnr)
        f1.write("\n")
        print("%s's PSNR:%.8f" % (test_hr_img_list[imid], psnr))
        sum += psnr

        if is_trans == True:
            img = np.transpose(img)
            b_imgs_ = np.transpose(b_imgs_)

        print("[*] save images")
        img = tl.prepro.threading_data(img, fn=inv_norm)
        save_image(img.astype(np.uint8), save_dir+'/%s_gen_psnr%.4f.png' % (test_hr_img_list[imid][:-4], psnr))
        b_imgs_ = tl.prepro.threading_data(b_imgs_, fn=inv_norm)
        save_image(b_imgs_.astype(np.uint8), save_dir+'/%s_hr.png' % test_hr_img_list[imid][:-4])

    f1.close()
    psnr_a = sum / len(test_hr_imgs)
    print("PSNR_AVERAGE:%.8f" % psnr_a)
    time_a = sum_t / len(test_hr_imgs)
    print("TIME_AVERAGE:%.8f" % time_a)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='evaluate_gray', help='evaluate_gray')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'evaluate_gray':
        evaluate()
    else:
        raise Exception("Unknow --mode")


