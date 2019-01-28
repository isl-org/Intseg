from __future__ import division
import os,time,cv2
import scipy.io as sio
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from numpy import *
import scipy.linalg
from copy import copy, deepcopy

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

def prepare_data():
    train_im_names = [line.rstrip() for line in open('./train.txt')]
    val_im_names = [line.rstrip() for line in open('./val.txt')]
    return train_im_names,val_im_names

config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)

im_path = "./img"
seg_path = "./inst"
train_im_names,val_im_names = prepare_data()
input=tf.placeholder(tf.float32,shape=[None,None,None,7])
output=tf.placeholder(tf.float32,shape=[None,None,None,1])
sz=tf.placeholder(tf.int32,shape=[2])
input_vgg=tf.placeholder(tf.float32,shape=[None,None,None,3])
network=build(input,sz)
vgg19_network=build_vgg19(input_vgg)

# L2 Loss
loss_d1=tf.reduce_mean(tf.square(tf.expand_dims(network[:,:,:,0],axis=3)-output))
loss_d2=tf.reduce_mean(tf.square(tf.expand_dims(network[:,:,:,1],axis=3)-output))
loss_d3=tf.reduce_mean(tf.square(tf.expand_dims(network[:,:,:,2],axis=3)-output))
loss_d4=tf.reduce_mean(tf.square(tf.expand_dims(network[:,:,:,3],axis=3)-output))
loss_d5=tf.reduce_mean(tf.square(tf.expand_dims(network[:,:,:,4],axis=3)-output))
loss_d6=tf.reduce_mean(tf.square(tf.expand_dims(network[:,:,:,5],axis=3)-output))
loss = tf.reduce_min([loss_d1, loss_d2, loss_d3, loss_d4, loss_d5, loss_d6]) + 0.0025*(32*loss_d1+16*loss_d2+8*loss_d3+4*loss_d4+2*loss_d5+1*loss_d6)

# L1 Loss
loss2_d1=tf.reduce_mean(tf.abs(tf.expand_dims(network[:,:,:,0],axis=3)-output))
loss2_d2=tf.reduce_mean(tf.abs(tf.expand_dims(network[:,:,:,1],axis=3)-output))
loss2_d3=tf.reduce_mean(tf.abs(tf.expand_dims(network[:,:,:,2],axis=3)-output))
loss2_d4=tf.reduce_mean(tf.abs(tf.expand_dims(network[:,:,:,3],axis=3)-output))
loss2_d5=tf.reduce_mean(tf.abs(tf.expand_dims(network[:,:,:,4],axis=3)-output))
loss2_d6=tf.reduce_mean(tf.abs(tf.expand_dims(network[:,:,:,5],axis=3)-output))
loss2 = tf.reduce_min([loss2_d1, loss2_d2, loss2_d3, loss2_d4, loss2_d5, loss2_d6]) + 0.0025*(32*loss2_d1+16*loss2_d2+8*loss2_d3+4*loss2_d4+2*loss2_d5+1*loss2_d6)

# IoU Loss
nw1 = tf.expand_dims(network[:,:,:,0],axis=3)
nw2 = tf.expand_dims(network[:,:,:,1],axis=3)
nw3 = tf.expand_dims(network[:,:,:,2],axis=3)
nw4 = tf.expand_dims(network[:,:,:,3],axis=3)
nw5 = tf.expand_dims(network[:,:,:,4],axis=3)
nw6 = tf.expand_dims(network[:,:,:,5],axis=3)
iou_d1 = 1-tf.reduce_mean(tf.multiply(nw1,output))/(tf.reduce_mean(tf.maximum(nw1,output))+1e-6)
iou_d2 = 1-tf.reduce_mean(tf.multiply(nw2,output))/(tf.reduce_mean(tf.maximum(nw2,output))+1e-6)
iou_d3 = 1-tf.reduce_mean(tf.multiply(nw3,output))/(tf.reduce_mean(tf.maximum(nw3,output))+1e-6)
iou_d4 = 1-tf.reduce_mean(tf.multiply(nw4,output))/(tf.reduce_mean(tf.maximum(nw4,output))+1e-6)
iou_d5 = 1-tf.reduce_mean(tf.multiply(nw5,output))/(tf.reduce_mean(tf.maximum(nw5,output))+1e-6)
iou_d6 = 1-tf.reduce_mean(tf.multiply(nw6,output))/(tf.reduce_mean(tf.maximum(nw6,output))+1e-6)
loss_iou = tf.reduce_min([iou_d1, iou_d2, iou_d3, iou_d4, iou_d5, iou_d6]) + 0.0025*(32*iou_d1+16*iou_d2+8*iou_d3+4*iou_d4+2*iou_d5+1*iou_d6)

# add positive/negative clicks as soft constraints
ct_mask = tf.cast(input[:,:,:,3],dtype=tf.bool) & tf.cast(input[:,:,:,4],dtype=tf.bool)
ct_mask = tf.tile(tf.expand_dims(~ct_mask,axis=3), [1,1,1,6])
ct_mask = tf.cast(ct_mask, dtype=tf.float32)
ct_mask /= tf.reduce_mean(ct_mask)
output_tile = tf.tile(output,[1,1,1,6])
ct_loss = tf.reduce_mean(tf.abs(network - output_tile) * ct_mask)

all_loss = loss_iou + ct_loss

opt=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(all_loss,var_list=[var for var in tf.trainable_variables() if var.name.startswith('g_')])

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.initialize_all_variables())
ckpt=tf.train.get_checkpoint_state("result64_vgg19_RDL6_IoU_dt_pt_ct_tanh")
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)

input_images=[None]*len(train_im_names)
output_masks=[None]*len(train_im_names)

# For displaying the losses
all=np.zeros(30000,dtype=float)
all2=np.zeros(30000,dtype=float)
all_iou=np.zeros(30000,dtype=float)
all_d1=np.zeros(30000,dtype=float)
all_d2=np.zeros(30000,dtype=float)
all_d3=np.zeros(30000,dtype=float)
all_d4=np.zeros(30000,dtype=float)
all_d5=np.zeros(30000,dtype=float)
all_d6=np.zeros(30000,dtype=float)

for epoch in range(1,101):
    if os.path.isdir("result64_vgg19_RDL6_IoU_dt_pt_ct_tanh/%04d"%epoch):
        continue
    cnt=0
    for id in np.random.permutation(len(train_im_names)):
    # for id in np.random.permutation(1):

        if input_images[id] is None:
            # The input image
            input_images[id] = cv2.imread(im_path + "/" + train_im_names[id]+".jpg",-1)
        if output_masks[id] is None:
            # The SBD Groundtruth mask
            mat_contents = sio.loadmat(seg_path + "/" + train_im_names[id] + ".mat")
            tmpstr = mat_contents['GTinst']
            tmpmat = tmpstr[0,0]
            output_masks[id] = tmpmat['Segmentation']
        output_mask = deepcopy(output_masks[id])
        output_mask[output_mask==255] = 0
        num_obj = output_mask.max()
        for obj_id in range(num_obj):
            st = time.time()
            # random clicks
            input_pos = cv2.imread("./train" + "/" + train_im_names[id] + "/ints/%03d_%03d_pos.png" % (obj_id + 1, np.random.randint(1, 16)),-1)
            input_neg = cv2.imread("./train" + "/" + train_im_names[id] + "/ints/%03d_%03d_neg.png" % (obj_id + 1, np.random.randint(1, 16)),-1)
            input_pos_clks = deepcopy(input_pos)
            input_neg_clks = deepcopy(input_neg)
            input_pos_clks[input_pos != 0] = 255
            input_neg_clks[input_neg != 0] = 255
            if np.sum(input_pos==0)==0:
                continue
            input_image=np.expand_dims(np.float32(np.concatenate(
                [input_images[id], np.expand_dims(input_pos,axis=2), np.expand_dims(input_neg,axis=2),
                 np.expand_dims(input_pos_clks,axis=2), np.expand_dims(input_neg_clks,axis=2)], axis=2)),axis=0)
            _,iH,iW,_=input_image.shape

            output_image = deepcopy(output_mask)
            output_image[output_mask != (obj_id+1)] = 0
            output_image[output_mask == (obj_id+1)] = 255
            output_image=np.expand_dims(np.expand_dims(np.float32(output_image),axis=0),axis=3)/255.0
            _,current,current2,current3,d1,d2,d3,d4,d5,d6=sess.run([opt,loss,loss2,loss_iou, iou_d1, iou_d2, iou_d3, iou_d4, iou_d5, iou_d6],feed_dict={input:input_image,sz:[iH,iW],output:output_image})
            all[cnt]=current*255.0*255.0 #squared in 255 range (remember the network takes [0,1]
            all2[cnt]=current2*255.0 #changed to 255 in error
            all_iou[cnt]=current3
            all_d1[cnt]=d1
            all_d2[cnt]=d2
            all_d3[cnt]=d3
            all_d4[cnt]=d4
            all_d5[cnt]=d5
            all_d6[cnt]=d6
            cnt+=1
            print("%d %d l2: %.4f l1: %.4f IoU: %.4f d1-6: %.4f %.4f %.4f %.4f %.4f %.4f time: %.4f %s"%(epoch,cnt,np.mean(all[np.where(all)]),np.mean(all2[np.where(all2)]),np.mean(all_iou[np.where(all_iou)]),np.mean(all_d1[np.where(all_d1)]),
                                                                                     np.mean(all_d2[np.where(all_d2)]),np.mean(all_d3[np.where(all_d3)]),np.mean(all_d4[np.where(all_d4)]), np.mean(all_d5[np.where(all_d5)]), np.mean(all_d6[np.where(all_d6)]),
                                                                                     time.time()-st,os.getcwd().split('/')[-2]))

    os.makedirs("result64_vgg19_RDL6_IoU_dt_pt_ct_tanh/%04d"%epoch)
    target=open("result64_vgg19_RDL6_IoU_dt_pt_ct_tanh/%04d/score.txt"%epoch,'w')
    target.write("%f\n%f\n%f"%(np.mean(all[np.where(all)]),np.mean(all2[np.where(all2)]),np.mean(all_iou[np.where(all_iou)])))
    target.close()

    saver.save(sess,"result64_vgg19_RDL6_IoU_dt_pt_ct_tanh/model.ckpt")
    saver.save(sess,"result64_vgg19_RDL6_IoU_dt_pt_ct_tanh/%04d/model.ckpt"%epoch)

    # validation
    all_test = np.zeros(100, dtype=float)
    all2_test = np.zeros(100, dtype=float)
    all_iou_test = np.zeros(100, dtype=float)
    target = open("result64_vgg19_RDL6_IoU_dt_pt_ct_tanh/%04d/test_score.txt" % epoch, 'w')

    for id in range(100):
        input_image = cv2.imread(im_path + "/" + val_im_names[id] + ".jpg", -1)
        input_pos = cv2.imread("./val" + "/" + val_im_names[id] + "/ints/%03d_%03d_pos.png" % (1, 1), -1)
        input_neg = cv2.imread("./val" + "/" + val_im_names[id] + "/ints/%03d_%03d_neg.png" % (1, 1), -1)
        input_pos_clks = deepcopy(input_pos)
        input_neg_clks = deepcopy(input_neg)
        input_pos_clks[input_pos != 0] = 255
        input_neg_clks[input_neg != 0] = 255
        output_gt = cv2.imread("./val" + "/" + val_im_names[id] + "/objs/%05d.png" % 1, -1)
        output_gt = np.expand_dims(np.expand_dims(np.float32(output_gt), axis=0), axis=3) / 255.0
        iH, iW, _ = input_image.shape
        input_image = np.expand_dims(np.float32(np.concatenate(
                [input_image, np.expand_dims(input_pos, axis=2), np.expand_dims(input_neg, axis=2),
                np.expand_dims(input_pos_clks, axis=2), np.expand_dims(input_neg_clks, axis=2)], axis=2)), axis=0)
        st=time.time()
        output_image, loss_test, loss2_test, iou_test = sess.run([network, loss, loss2, loss_iou],feed_dict={input:input_image,sz:[iH,iW],output: output_gt})
        all_test[id] = loss_test * 255.0 * 255.0
        all2_test[id] = loss2_test * 255
        all_iou_test[id] = iou_test
        target.write("%f %f %f\n" % (all_test[id], all2_test[id], all_iou_test[id]))
        print("%.3f"%(time.time()-st))
        output_image = np.minimum(np.maximum(output_image, 0.0), 1.0)
        for output_d in range(6):
            save_image = input_image[0, :, :, 0:3] / 255.0
            save_image[:, :, 0] = (save_image[:, :, 0] + 0.5 * output_image[0, :, :, output_d])
            save_image[:, :, 1] = (save_image[:, :, 1] + 0.5 * output_image[0, :, :, output_d])
            save_image[:, :, 2] = (save_image[:, :, 2] + 0.5 * output_image[0, :, :, output_d])
            save_image = np.minimum(np.maximum(save_image, 0.0), 1.0) * 255.0
            cv2.imwrite("result64_vgg19_RDL6_IoU_dt_pt_ct_tanh/%04d/%s_%02d_BW.png" % (epoch, val_im_names[id], output_d),
                    np.uint8(output_image[0, :, :, output_d] * 255.0))
            cv2.imwrite("result64_vgg19_RDL6_IoU_dt_pt_ct_tanh/%04d/%s_%02d.jpg" % (epoch, val_im_names[id], output_d),
                    np.uint8(save_image))
    target.write("Mean: %f %f %f\n" % (np.mean(all_test[np.where(all_test)]), np.mean(all2_test[np.where(all2_test)]), np.mean(all_iou_test[np.where(all_iou_test)])))
    target.close()
