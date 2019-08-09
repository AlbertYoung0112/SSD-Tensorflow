# -*- coding: utf-8 -*-

import tensorflow as tf
from nets import ssd_vgg_300
from tensorflow.python.framework import graph_util

slim = tf.contrib.slim

data_format = 'NHWC'

image_input = tf.placeholder(tf.float32, shape=(None, 300, 300, 3), name='ssd_input')
ssd_net = ssd_vgg_300.SSDNet()
output_pred_shapes = []
output_loc_shapes = []
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_input, is_training=False, reuse=False)
    for i in range(len(predictions)):
        output_pred_shapes.append(predictions[i].shape.as_list())
    for i in range(len(localisations)):
        output_loc_shapes.append(localisations[i].shape.as_list())
    outputs = predictions + localisations
    outputs_flatten = [tf.reshape(outputs[i], [1, -1]) for i in range(len(outputs))]
    outputs_cat = tf.concat(outputs_flatten, axis=1, name='ssd_output')

ckpt_filename = './checkpoints/ssd_300_vgg.ckpt'

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_filename)
    summary_writer = tf.summary.FileWriter('./freeze_log/', sess.graph)
    constant_graph = graph_util.convert_variables_to_constants(
        sess=sess, 
        input_graph_def=sess.graph_def, 
        output_node_names=['ssd_output']
        )
    with tf.gfile.FastGFile("SSD-TF.pb", mode='wb') as f:
        f.write(constant_graph.SerializeToString())
