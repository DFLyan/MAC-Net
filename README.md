# Learning Memory Augmented Cascading Network for Compressed Sensing of Images
```
@inproceedings{chen2020learning,
  title={Learning memory augmented cascading network for compressed sensing of images},
  author={Chen, Jiwei and Sun, Yubao and Liu, Qingshan and Huang, Rui},
  booktitle={European Conference on Computer Vision},
  pages={513--529},
  year={2020},
  organization={Springer}
}
```

## DATA SET
We utilize 91 images which are the common usage in CS training. We crop these images into the size of 96*96. We test our model in three dataset: SET11, BSD68 and MICCAI 2013(MRI).
## TRAIN
* just 
```
$ python MAC_Net_train.py
```
## TEST
```
$ python MAC_Net_test.py
```

## Other things
Other details will be added.

# New
## If you want to simplify the generation of y from the image, you can modify the code:
### Firstly, reshape the measurement matrix:
```
phi = np.reshape(A, (block_size, block_size, 1, size_y))
```
### Secondly, use the function "tf.nn.conv2d" to realize the samling(measurement), for example:
```
y_meas = tf.nn.conv2d(t_target_image, A, (1, block_size, block_size, 1), padding='SAME')
```
### This can be modified in "train.py" and "model.py" where use the "for" to generate the y. It is time-consuming to use the "for". So if you use the new code, it will be faster. Because it will be processed in GPU instead of CPU.
