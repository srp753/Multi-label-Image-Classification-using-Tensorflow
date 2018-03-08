#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 17:07:30 2018

@author: snigdha
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import sys
import os
import numpy as np
import tensorflow as tf
import argparse
#import os.path as osp
from PIL import Image
from functools import partial
from collections import defaultdict
import pickle

from eval import compute_map
#import models

tf.logging.set_verbosity(tf.logging.INFO)

CLASS_NAMES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]

def cnn_model_fn(features, labels, mode, num_classes=20):
    # Write this function
    # """Model function for CNN."""
    # Input Layer
    
    input_layer = tf.reshape(features["x"], [-1, 224, 224, 3])
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        flipped = tf.map_fn(lambda image: tf.image.random_flip_left_right(image),features["x"])
        cropped = tf.map_fn(lambda image:tf.random_crop(image,size=[224,224,3]),features["x"])
        
        fets = tf.concat([features["x"],flipped,cropped],axis = 0)
        #wts = tf.concat([features["w"],features["w"],features["w"]],axis = 0)
        lbls = tf.concat([labels,labels,labels],axis = 0)
        
        feats = tf.random_shuffle(fets,seed = features["x"].shape[0]*3)
        #wtgs = tf.random_shuffle(wts,seed = features["x"].shape[0]*3)
        lbels = tf.random_shuffle(lbls,seed = features["x"].shape[0]*3)
        
        features["x"]= feats
        input_layer = features["x"]
        labels = lbels
    
    tf.summary.image("Training_images",input_layer)
    
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[3,3],
        padding="same",
        strides = 1,
        activation=tf.nn.relu     
        )

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        kernel_size=[3,3],
        padding="same",
        strides = 1,
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Convolutional Layer #3 
    conv3 = tf.layers.conv2d(
        inputs=pool1,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        strides = 1,
        activation=tf.nn.relu)
    
    # Convolutional Layer #4 
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        strides = 1,
        activation=tf.nn.relu
        )
    
    #Pooling layer 2
    pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
    
    # Convolutional Layer #5 
    conv5 = tf.layers.conv2d(
        inputs=pool2,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        strides = 1,
        activation=tf.nn.relu,
        )
    
    # Convolutional Layer #6 
    conv6 = tf.layers.conv2d(
        inputs=conv5,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        strides = 1,
        activation=tf.nn.relu,
        )
    
    # Convolutional Layer #7 
    conv7 = tf.layers.conv2d(
        inputs=conv6,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        strides = 1,
        activation=tf.nn.relu,
        )
    
    #Pooling layer 3
    pool3 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=2)
    
    # Convolutional Layer #8 
    conv8 = tf.layers.conv2d(
        inputs=pool3,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        strides = 1,
        activation=tf.nn.relu
        )
    
    # Convolutional Layer #9
    conv9 = tf.layers.conv2d(
        inputs=conv8,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        strides = 1,
        activation=tf.nn.relu,
        )
    
    # Convolutional Layer #10
    conv10 = tf.layers.conv2d(
        inputs=conv9,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        strides = 1,
        activation=tf.nn.relu
        )
    
    #Pooling layer 4
    pool4 = tf.layers.max_pooling2d(inputs=conv10, pool_size=[2, 2], strides=2)
    
    
    # Convolutional Layer #11
    conv11 = tf.layers.conv2d(
        inputs=pool4,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        strides = 1,
        activation=tf.nn.relu
        )
    
    # Convolutional Layer #12
    conv12 = tf.layers.conv2d(
        inputs=conv11,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        strides = 1,
        activation=tf.nn.relu
        )
    
    # Convolutional Layer #13
    conv13 = tf.layers.conv2d(
        inputs=conv12,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        strides = 1,
        activation=tf.nn.relu
        )
    
    #Pooling layer 5
    pool5 = tf.layers.max_pooling2d(inputs=conv13, pool_size=[2, 2], strides=2)
    
    
    # Dense Layer
    pool5_flat = tf.contrib.layers.flatten(pool5)
    
    dense1 = tf.layers.dense(inputs=pool5_flat, units=4096,
                            activation=tf.nn.relu)                       
                            
    dropout1 = tf.layers.dropout(
        inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    dense2 = tf.layers.dense(inputs=dropout1, units=4096,
                            activation=tf.nn.relu)
                            
    dropout2 = tf.layers.dropout(
        inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    
    dense3 = tf.layers.dense(inputs=dropout2, units=1000,
                             activation = tf.nn.relu)
                             
    # Logits Layer
    logits = tf.layers.dense(inputs=dense3, units=20)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.sigmoid(logits, name="sigmoid_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    
    
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        
        loss = tf.identity(tf.losses.sigmoid_cross_entropy(
        labels, logits=logits), name='loss')
        
        decay_learning_rate = tf.train.exponential_decay(
                learning_rate = 0.001,
                global_step=tf.train.get_global_step(),
                decay_steps = 10000,
                decay_rate = 0.5,
                staircase = False,
                name = None)
        optimizer = tf.train.MomentumOptimizer(learning_rate=decay_learning_rate,
                                                         momentum = 0.9)

        tf.summary.scalar("decayed_learning_rate",decay_learning_rate)
        
        grads_and_vars= optimizer.compute_gradients(loss)
        
        for g, v in grads_and_vars:
            if g is not None:
                #print(format(v.name))
                tf.summary.histogram("{}/grad_histogram".format(v.name), g)
                
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)
        

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    
    
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a classifier in tensorflow!')
    parser.add_argument(
        'data_dir', type=str, default='data/VOC2007',
        help='Path to PASCAL data storage')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def _get_el(arr, i):
    try:
        return arr[i]
    except IndexError:
        return arr
    
from tensorflow.core.framework import summary_pb2
def summary_var(log_dir, name, val, step):
    writer = tf.summary.FileWriterCache.get(log_dir)
    summary_proto = summary_pb2.Summary()
    value = summary_proto.value.add()
    value.tag = name
    value.simple_value = float(val)
    writer.add_summary(summary_proto, step)
    writer.flush()

def load_pascal(data_dir, split='train'):
    
    """
    Function to read images from PASCAL data folder.
    Args:
        data_dir (str): Path to the VOC2007 directory.
        split (str): train/val/trainval split to use.
    Returns:
        images (np.ndarray): Return a np.float32 array of
            shape (N, H, W, 3), where H, W are 224px each,
            and each image is in RGB format.
        labels (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are active in that image.
    """
    
    
    sub_dir1 = '/ImageSets/Main/'
    sub_dir2 = '/JPEGImages/'
    f1 = open(data_dir+sub_dir1+"aeroplane"+"_"+split+".txt", 'r')

    img = []

    for line1 in f1:
        g1 = line1.strip().split(' ')
        img.append(g1[0])
    
    num =len(img)
    print("num",num)

    w = np.int32(np.zeros((num,20)))
    l = np.int32(np.zeros((num,20)))

    print("Entering the loop for weights and labels")

    cnt = 0
    for i in range(0,20):
    
        f2 = open(data_dir + '/ImageSets/Main/'+CLASS_NAMES[i]+'_'+split+'.txt')
        a1 = f2.read().split()
        t = a1[1::2]
        tt = np.int32(t)
        ttt = tt.reshape(1,num)
        w[:,cnt] = np.int32(np.abs(ttt))
        l[:,cnt] = ttt.clip(min = 0)
        cnt = cnt + 1
    

    labels = np.int32(l)
    weights = np.int32(w)
    print("Entering the loop for images")
    arr = []
    for j in img:
    
        im = Image.open(data_dir+sub_dir2+ j +'.jpg')
        im = im.resize((256, 256), Image.ANTIALIAS)
        arr.append(np.float32(im))

    image_ar = np.float32(arr)
    return (image_ar,labels,weights)  


def main():
    args = parse_args()
    # Load training and eval data
    train_data, train_labels, train_weights = load_pascal(
        args.data_dir, split='trainval')
    eval_data, eval_labels, eval_weights = load_pascal(
        args.data_dir, split='test')
    
        
    pascal_classifier = tf.estimator.Estimator(
        model_fn=partial(cnn_model_fn,
                         num_classes=train_labels.shape[1]),
        model_dir="pascal_vgg")
    tensors_to_log = {"loss": "loss"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=400)
    
    list22 = []
    for i in range(0,100):
        
        # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data, "w": train_weights},
        y=train_labels,
        batch_size=10,
        num_epochs=None,
        shuffle=True)
        
        pascal_classifier.train(
                input_fn=train_input_fn,
                steps=400,
                hooks=[logging_hook])
        
        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": eval_data, "w": eval_weights},
                y=eval_labels,
                num_epochs=1,
                shuffle=False)
        
        pred = list(pascal_classifier.predict(input_fn=eval_input_fn))
        pred = np.stack([p['probabilities'] for p in pred])
        rand_AP = compute_map(
                eval_labels, np.random.random(eval_labels.shape),
                eval_weights, average=None)
        print('Random AP: {} mAP'.format(np.mean(rand_AP)))
        gt_AP = compute_map(
                eval_labels, eval_labels, eval_weights, average=None)
        print('GT AP: {} mAP'.format(np.mean(gt_AP)))
        AP = compute_map(eval_labels, pred, eval_weights, average=None)
        print('Obtained {} mAP'.format(np.mean(AP)))
        print('per class:')
        for cid, cname in enumerate(CLASS_NAMES):
            print('{}: {}'.format(cname, _get_el(AP, cid)))
        list22.append(np.mean(AP))
        
        summary_var("pascal_vgg","mAP",np.mean(AP),i*400)
        
    with open('list22.pkl','wb') as fr2:
        pickle.dump(list22,fr2)    
    
            
        

if __name__ == "__main__":
    main()