import tensorflow as tf
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
from tf_grouping import group_point
import tf_util

def pre_load_checkpoint(checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        # print(" [*] Reading checkpoint from {}".format(ckpt.model_checkpoint_path))
        epoch_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        return epoch_step, ckpt.model_checkpoint_path
    else:
        return 0, None

def dgcnn(input, mlp, is_training, bn_decay, scope, bn=True):
    # input: [B, N, 1, C]

    k = 20

    input_squeeze = tf.squeeze(input, axis=2)  # [B, N, C]
    input_transpose = tf.transpose(input_squeeze, perm=[0,2,1])
    input_inner = tf.matmul(input_squeeze, input_transpose)
    input_inner = -2.0*input_inner
    input_square = tf.reduce_sum(tf.square(input_squeeze), axis=-1, keep_dims=True)
    input_square_transpose = tf.transpose(input_square, perm=[0, 2, 1])

    pairwise_dist = input_square + input_inner + input_square_transpose  # [B, N, N]

    neg_adj = -pairwise_dist
    _, nn_idx = tf.nn.top_k(neg_adj, k=k)  # [B, N, 20]

    input_neighbors = group_point(input_squeeze, nn_idx)  # [B, N, 20, C]
    input_tile = tf.tile(input, [1, 1, k, 1])  # [B, N, 20, C]

    dg_feature = tf.concat([input_tile, input_neighbors-input_tile], axis=-1)  # [B, N, 20, C*]

    with tf.variable_scope(scope) as sc:
        for i, num_out_channel in enumerate(mlp):
            dg_feature = tf_util.conv2d(dg_feature, num_out_channel, [1, 1], padding='VALID', stride=[1, 1],
                                        bn=bn, is_training=is_training, scope='conv%d' % (i), bn_decay=bn_decay)  # [B, N, 20, output_channel]

    net = tf.reduce_max(dg_feature, axis=-2, keep_dims=True)  # [B, N, 1, output_channel]

    return net

def cbam_module2(x, ratio=4, name='cbam',bn_decay=None, is_training=True, bn=True):
    channels = x.get_shape()[-1].value
    with tf.variable_scope(name):
        with tf.variable_scope('channel_attention'):
            x_gap = tf.reduce_mean(x, axis=1, keepdims=True)
            x_gap = tf.squeeze(x_gap,axis=[1,2])
            x_gap= tf_util.fully_connected(x_gap, channels//ratio, bn=bn, is_training=is_training,scope='fc1')
            x_gap= tf_util.fully_connected(x_gap, channels, bn=bn, is_training=is_training,scope='fc2',activation_fn=None)

        with tf.variable_scope('channel_attention', reuse=True):
            x_gmp = tf.reduce_max(x, axis=1, keepdims=True)
            x_gmp = tf.squeeze(x_gmp,axis=[1,2])
            x_gmp = tf_util.fully_connected(x_gmp, channels // ratio, bn=bn, is_training=is_training, scope='fc1')
            x_gmp = tf_util.fully_connected(x_gmp, channels, bn=bn, is_training=is_training, scope='fc2', activation_fn=None)

            scale = tf.reshape(x_gap + x_gmp, [-1, 1, 1, channels])
            scale = tf.nn.sigmoid(scale)

            x = x * scale

    return x

def normal_grouping_embedding(input, mlp, is_training, bn_decay, scope, local_size, bn=True, pooling='max'):
    '''
    :param input: [B, N, K, 3]
    :param is_training:
    :param bn_decay:
    :param scope:
    :param bn:
    :param pooling:
    :return: output: [B, N, 1, C']
    '''

    K = input.get_shape()[2].value
    center = input[:, :, 0, :]
    center = tf.expand_dims(center, axis=2)  # [B, N, 1, C]
    center_tile = tf.tile(center, [1, 1, K, 1])  # [B, N, K, C]
    input_concat = tf.concat([center_tile, input], axis=-1)  # [B, N, K, 2C]

    input_local = input_concat[:, :, 0:local_size, :]
    with tf.variable_scope(scope) as sc:
        for i, num_out_channel in enumerate(mlp):
            input_local = tf_util.conv2d(input_local, num_out_channel, [1,1],
                                          padding='VALID', stride=[1,1],
                                          bn=bn, is_training=is_training,
                                          scope='conv%d' % (i), bn_decay=bn_decay)
        if pooling=='max':
            output = tf.reduce_max(input_local, axis=2, keep_dims=True, name='maxpool')  # [B, N, 1, C']
        elif pooling=='avg':
            output = tf.reduce_mean(input_local, axis=2, keep_dims=True, name='avgpool')

        output = cbam_module2(output, is_training=is_training, bn_decay=bn_decay, name="cbam")

        return output

def feature_grouping_embedding(input, idx, mlp, is_training, bn_decay, scope, local_size, bn=True, pooling='max',size=50):
    '''
    :param input: [B, N, 1, C]
    :param idx: [B, N, 50]
    :param is_training:
    :param bn_decay:
    :param scope:
    :param bn:
    :param pooling:
    :return: output: [B, N, 1, C']
    '''

    input_tile = tf.tile(input, [1, 1, size, 1])  # [B, N, 50, C]

    # find local feature map according to topology indices
    input = tf.squeeze(input, axis=2)
    input_group = group_point(input, idx)  # [B, N, 50, C]
    input_concat = tf.concat([input_tile, input_group-input_tile], axis=-1)  # [B, N, 50, 2C]
    input_local = input_concat[:, :, 0:local_size, :]

    with tf.variable_scope(scope) as sc:
        for i, num_out_channel in enumerate(mlp):
            input_local = tf_util.conv2d(input_local, num_out_channel, [1,1],
                                          padding='VALID', stride=[1,1],
                                          bn=bn, is_training=is_training,
                                          scope='conv%d' % (i), bn_decay=bn_decay)

        if pooling=='max':
            output = tf.reduce_max(input_local, axis=2, keep_dims=True, name='maxpool')  # [B, N, 1, C']
        elif pooling=='avg':
            output = tf.reduce_mean(input_local, axis=2, keep_dims=True, name='avgpool')

        output = cbam_module2(output, is_training=is_training, bn_decay=bn_decay, name="cbam")

        return output

