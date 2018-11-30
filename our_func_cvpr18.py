from __future__ import division
import os,time,cv2
import scipy.io as sio
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from numpy import *
import scipy.linalg
from copy import copy, deepcopy
from scipy import ndimage

def compIoU(im1, im2):
    im1_mask = (im1>0.5)
    im2_mask = (im2>0.5)
    iou = np.sum(im1_mask&im2_mask)/np.sum(im1_mask|im2_mask)
    return iou

def lrelu(x):
    return tf.maximum(x*0.2,x)

def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[0]//2, shape[1]//2
        for i in range(min(shape[2],shape[3])):
            array[cx, cy, i, i] = 1
        return tf.constant(array, dtype=dtype)
    return _initializer

def nm(x):
    w0=tf.Variable(1.0,name='w0')
    w1=tf.Variable(0.0,name='w1')
    return w0*x+w1*slim.batch_norm(x)

MEAN_VALUES = np.array([123.6800, 116.7790, 103.9390]).reshape((1,1,1,3))

def build_net(ntype,nin,nwb=None,name=None):
    if ntype=='conv':
        return tf.nn.relu(tf.nn.conv2d(nin,nwb[0],strides=[1,1,1,1],padding='SAME',name=name)+nwb[1])
    elif ntype=='pool':
        return tf.nn.avg_pool(nin,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def get_weight_bias(vgg_layers,i):
    weights=vgg_layers[i][0][0][2][0][0]
    weights=tf.constant(weights)
    bias=vgg_layers[i][0][0][2][0][1]
    bias=tf.constant(np.reshape(bias,(bias.size)))
    return weights,bias

def build_vgg19(input,reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    net={}
    vgg_rawnet=scipy.io.loadmat('Models/imagenet-vgg-verydeep-19.mat')
    vgg_layers=vgg_rawnet['layers'][0]
    net['input']=input-MEAN_VALUES
    net['conv1_1']=build_net('conv',net['input'],get_weight_bias(vgg_layers,0),name='vgg_conv1_1')
    net['conv1_2']=build_net('conv',net['conv1_1'],get_weight_bias(vgg_layers,2),name='vgg_conv1_2')
    net['pool1']=build_net('pool',net['conv1_2'])
    net['conv2_1']=build_net('conv',net['pool1'],get_weight_bias(vgg_layers,5),name='vgg_conv2_1')
    net['conv2_2']=build_net('conv',net['conv2_1'],get_weight_bias(vgg_layers,7),name='vgg_conv2_2')
    net['pool2']=build_net('pool',net['conv2_2'])
    net['conv3_1']=build_net('conv',net['pool2'],get_weight_bias(vgg_layers,10),name='vgg_conv3_1')
    net['conv3_2']=build_net('conv',net['conv3_1'],get_weight_bias(vgg_layers,12),name='vgg_conv3_2')
    net['conv3_3']=build_net('conv',net['conv3_2'],get_weight_bias(vgg_layers,14),name='vgg_conv3_3')
    net['conv3_4']=build_net('conv',net['conv3_3'],get_weight_bias(vgg_layers,16),name='vgg_conv3_4')
    net['pool3']=build_net('pool',net['conv3_4'])
    net['conv4_1']=build_net('conv',net['pool3'],get_weight_bias(vgg_layers,19),name='vgg_conv4_1')
    net['conv4_2']=build_net('conv',net['conv4_1'],get_weight_bias(vgg_layers,21),name='vgg_conv4_2')
    net['conv4_3']=build_net('conv',net['conv4_2'],get_weight_bias(vgg_layers,23),name='vgg_conv4_3')
    net['conv4_4']=build_net('conv',net['conv4_3'],get_weight_bias(vgg_layers,25),name='vgg_conv4_4')
    net['pool4']=build_net('pool',net['conv4_4'])
    net['conv5_1']=build_net('conv',net['pool4'],get_weight_bias(vgg_layers,28),name='vgg_conv5_1')
    net['conv5_2']=build_net('conv',net['conv5_1'],get_weight_bias(vgg_layers,30),name='vgg_conv5_2')
    #net['conv5_3']=build_net('conv',net['conv5_2'],get_weight_bias(vgg_layers,32),name='vgg_conv5_3')
    #net['conv5_4']=build_net('conv',net['conv5_3'],get_weight_bias(vgg_layers,34),name='vgg_conv5_4')
    #net['pool5']=build_net('pool',net['conv5_4'])
    return net

def build(input,sz):
    vgg19_features=build_vgg19(input[:,:,:,0:3])
    for layer_id in range(1,6):
        vgg19_f = vgg19_features['conv%d_2'%layer_id]
        input = tf.concat([input, tf.image.resize_bilinear(vgg19_f,sz)], axis=3)
    input = input/255.0
    net=slim.conv2d(input,64,[1,1],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv0')
    net=slim.conv2d(net,64,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv1')
    net=slim.conv2d(net,64,[3,3],rate=2,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv2')
    net=slim.conv2d(net,64,[3,3],rate=4,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv3')
    net=slim.conv2d(net,64,[3,3],rate=8,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv4')
    net=slim.conv2d(net,64,[3,3],rate=16,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv5')
    net=slim.conv2d(net,64,[3,3],rate=32,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv6')
    net=slim.conv2d(net,64,[3,3],rate=64,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv7')
    net=slim.conv2d(net,64,[3,3],rate=128,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv8')
    net=slim.conv2d(net,64,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv9')
    net=slim.conv2d(net,6,[1,1],rate=1,activation_fn=None,scope='g_conv_last')
    return tf.tanh(net)

def our_func(usrId, imIdx, im_path, cnt, pn, clk):

    if not os.path.isdir("res/%d/Ours/%05d" % (usrId, imIdx)):
        os.makedirs("res/%d/Ours/%05d/ints" % (usrId, imIdx))
        os.makedirs("res/%d/Ours/%05d/segs" % (usrId, imIdx))
        os.makedirs("res/%d/Ours/%05d/tmps" % (usrId, imIdx))

    sess=tf.Session()

    if cnt == 0 and imIdx == 0:
        global network,input,output,sz
        input = tf.placeholder(tf.float32, shape=[None, None, None, 7])
        output = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        sz = tf.placeholder(tf.int32, shape=[2])
        network=build(input,sz)

    saver = tf.train.Saver(var_list=[var for var in tf.trainable_variables() if var.name.startswith('g_')])
    sess.run(tf.initialize_all_variables())

    ckpt=tf.train.get_checkpoint_state("Models/ours_cvpr18")
    if ckpt:
        # print('loaded '+ckpt.model_checkpoint_path)
        saver.restore(sess,ckpt.model_checkpoint_path)


    input_image = cv2.imread(im_path, -1)
    iH, iW, _ = input_image.shape
    if cnt == 0:
        int_pos = np.uint8(255*np.ones([iH,iW]))
        int_neg = np.uint8(255*np.ones([iH,iW]))
        tmp_clk = cv2.imread(im_path, -1)
    else:
        int_pos = cv2.imread('res/%d/Ours/%05d/ints/pos_dt_%03d.png' % (usrId, imIdx, cnt - 1), -1)
        int_neg = cv2.imread('res/%d/Ours/%05d/ints/neg_dt_%03d.png' % (usrId, imIdx, cnt - 1), -1)
        tmp_clk = cv2.imread('res/%d/Ours/%05d/tmps/clk_%03d.png' % (usrId, imIdx, cnt - 1), -1)
    clk_pos = (int_pos==0)
    clk_neg = (int_neg==0)
    if pn == 1:
        clk_pos[clk.y,clk.x] = 1
        int_pos = ndimage.distance_transform_edt(1-clk_pos)
        int_pos = np.uint8(np.minimum(np.maximum(int_pos, 0.0), 255.0))
        cv2.imwrite('res/%d/Ours/%05d/ints/pos_dt_%03d.png' % (usrId, imIdx, cnt), int_pos)
        cv2.imwrite('res/%d/Ours/%05d/ints/neg_dt_%03d.png' % (usrId, imIdx, cnt), int_neg)
        cv2.circle(tmp_clk, (clk.x, clk.y), 5, (0, 255, 0), -1)
    else:
        clk_neg[clk.y,clk.x] = 1
        int_neg = ndimage.distance_transform_edt(1-clk_neg)
        int_neg = np.uint8(np.minimum(np.maximum(int_neg, 0.0), 255.0))
        cv2.imwrite('res/%d/Ours/%05d/ints/pos_dt_%03d.png' % (usrId, imIdx, cnt), int_pos)
        cv2.imwrite('res/%d/Ours/%05d/ints/neg_dt_%03d.png' % (usrId, imIdx, cnt), int_neg)
        cv2.circle(tmp_clk, (clk.x, clk.y), 5, (0, 0, 255), -1)
    input_pos_clks = deepcopy(int_pos)
    input_neg_clks = deepcopy(int_neg)
    input_pos_clks[int_pos != 0] = 255
    input_neg_clks[int_neg != 0] = 255
    input_ = np.expand_dims(np.float32(np.concatenate([input_image, np.expand_dims(int_pos, axis=2), np.expand_dims(int_neg, axis=2),
                                                      np.expand_dims(input_pos_clks, axis=2), np.expand_dims(input_neg_clks, axis=2)],axis=2)), axis=0)
    output_image = sess.run([network],feed_dict={input:input_,sz:[iH,iW]})
    output_image = np.minimum(np.maximum(output_image, 0.0), 1.0)
    output_image[np.where(output_image>0.5)]=1
    output_image[np.where(output_image<=0.5)]=0
    res_path = 'res/%d/Ours/%05d/segs/%03d.png' % (usrId, imIdx, cnt)
    segmask = np.uint8(output_image[0, 0, :, :, 0] * 255.0)

    cv2.imwrite(res_path, segmask)

    tmp_ol = cv2.imread(im_path, -1)
    tmp_ol[:,:,0] = 0.5*tmp_ol[:,:,0] + 0.5*segmask
    tmp_ol[:,:,1] = 0.5*tmp_ol[:,:,1] + 0.5*segmask
    tmp_ol[:,:,2] = 0.5*tmp_ol[:,:,2] + 0.5*segmask

    tmp_clk_path = 'res/%d/Ours/%05d/tmps/clk_%03d.png' % (usrId, imIdx, cnt)
    tmp_ol_path = 'res/%d/Ours/%05d/tmps/ol_%03d.png' % (usrId, imIdx, cnt)
    cv2.imwrite(tmp_clk_path, tmp_clk)
    cv2.imwrite(tmp_ol_path, tmp_ol)
