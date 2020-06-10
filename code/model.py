import tensorflow as tf
import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/grouping'))

import model_util
import tf_util

def get_model(x, idx, is_training=True, bn_decay=None, sizes=[10, 30, 50]):
    '''
    :param x: [B, N, K, 3] --- [batch_size, patch_size, neighbor_size, vertex_normal]
    :param idx: [B, N, K]
    :param is_training:
    :param bn_decay:
    :return:
    '''
    input_normal = x[:, :, 0, :]
    input_normal = tf.expand_dims(input_normal, axis=2)

    net_noisy_1 = model_util.normal_grouping_embedding(x, mlp=[32, 32, 64], local_size=sizes[0],
                                                        is_training=is_training,
                                                        bn_decay=bn_decay, scope='layer_1')  # [B, N, 1, C]
    net_noisy_2 = model_util.feature_grouping_embedding(net_noisy_1, idx, mlp=[64, 64, 128],
                                                                   local_size=sizes[1], is_training=is_training,
                                                                   bn_decay=bn_decay, scope='layer_2',
                                                                   size=sizes[2])  # [B, N, 1, C]
    net_noisy_3 = model_util.feature_grouping_embedding(net_noisy_2, idx, mlp=[128, 128, 256],
                                                                   local_size=sizes[2], is_training=is_training,
                                                                   bn_decay=bn_decay, scope='layer_3',
                                                                   size=sizes[2])  # [B, N, 1, C]
    net_noisy = tf.concat([input_normal, net_noisy_1, net_noisy_2, net_noisy_3], axis=-1)  # [B, N, 1, 3C+3]
    net_noisy = tf_util.conv2d(net_noisy, 128, [1, 1], padding='VALID', stride=[1, 1],
                               bn=True, is_training=is_training, scope='layer',
                               bn_decay=bn_decay)  # [B, N, 1, 128]

    ########### denoise level 1 ###############
    res_coarse = model_util.dgcnn(net_noisy, [128, 128], is_training, bn_decay, 'res_coarse', bn=True)  # [B, N, 1, 128]

    net_clean_coarse = tf.subtract(net_noisy, res_coarse)  # [B, N, 1, 128]

    ## regress to three normal vectors
    normal_coarse = tf_util.conv2d(net_clean_coarse, 128, [1, 1], padding='VALID', stride=[1, 1], bn=True,
                                   is_training=is_training, bn_decay=bn_decay, scope='regress_coarse_1')
    normal_coarse = tf_util.conv2d(normal_coarse, 64, [1, 1], padding='VALID', stride=[1, 1], bn=True,
                                   is_training=is_training, bn_decay=bn_decay, scope='regress_coarse_2')
    normal_coarse = tf_util.conv2d(normal_coarse, 3, [1, 1], padding='VALID', stride=[1, 1],
                                   activation_fn=None, scope='regress_coarse_3')
    normal_coarse = tf.squeeze(normal_coarse, axis=2)  # [B, N, 3]

    ## magnitude of the normal vector should be 1
    normal_coarse = tf.nn.l2_normalize(normal_coarse, dim=2)
    ############################################

    ########### denoise level 2 ###############
    res_fine = model_util.dgcnn(net_clean_coarse, [128, 128], is_training, bn_decay, 'res_fine',
                                bn=True)  # [B, N, 1, 128]

    net_clean_fine = tf.subtract(net_clean_coarse, res_fine)  # [B, N, 1, 128]

    net_clean = tf.add(net_clean_coarse, net_clean_fine)

    ## regress to three normal vectors
    normal_fine = tf_util.conv2d(net_clean, 128, [1, 1], padding='VALID', stride=[1, 1], bn=True,
                                 is_training=is_training, bn_decay=bn_decay, scope='regress_fine_1')
    normal_fine = tf_util.conv2d(normal_fine, 64, [1, 1], padding='VALID', stride=[1, 1], bn=True,
                                 is_training=is_training, bn_decay=bn_decay, scope='regress_fine_2')
    normal_fine = tf_util.conv2d(normal_fine, 3, [1, 1], padding='VALID', stride=[1, 1],
                                 activation_fn=None, scope='regress_fine_3')
    normal_fine = tf.squeeze(normal_fine, axis=2)  # [B, N, 3]

    ## magnitude of the normal vector should be 1
    normal_fine = tf.nn.l2_normalize(normal_fine, dim=2)
    ############################################

    return tf.squeeze(res_coarse, axis=2), tf.squeeze(res_fine, axis=2), \
           normal_coarse, normal_fine, \
           tf.squeeze(net_clean_coarse, axis=2), tf.squeeze(net_clean_fine, axis=2)


def get_loss(normal_coarse, normal_fine, normal_gt, res_coarse, res_fine, alpha, reg_weight=0.05):
    dist_coarse = tf.reduce_sum(tf.square(normal_gt - normal_coarse), axis=-1)
    per_patch_loss_coarse = tf.reduce_mean(dist_coarse, axis=-1)
    per_batch_loss_coarse = tf.reduce_mean(per_patch_loss_coarse)

    ## enforce the residual as small as possible
    dist_res_coarse = tf.reduce_sum(tf.square(res_coarse), axis=-1)
    per_patch_loss_res_coarse = tf.reduce_mean(dist_res_coarse, axis=-1)
    per_batch_loss_res_coarse = tf.reduce_mean(per_patch_loss_res_coarse)

    loss_coarse = per_batch_loss_coarse + per_batch_loss_res_coarse * reg_weight

    dist_fine = tf.reduce_sum(tf.square(normal_gt - normal_fine), axis=-1)
    per_patch_loss_fine = tf.reduce_mean(dist_fine, axis=-1)
    per_batch_loss_fine = tf.reduce_mean(per_patch_loss_fine)

    ## enforce the residual as small as possible
    dist_res_fine = tf.reduce_sum(tf.square(res_fine), axis=-1)
    per_patch_loss_res_fine = tf.reduce_mean(dist_res_fine, axis=-1)
    per_batch_loss_res_fine = tf.reduce_mean(per_patch_loss_res_fine)

    loss_fine = per_batch_loss_fine + per_batch_loss_res_fine * reg_weight

    loss_total = alpha * loss_coarse + loss_fine

    return loss_coarse, loss_fine, loss_total

if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 800, 50, 3))
        idx = tf.zeros((32, 800, 50), dtype="int32")
        gt = tf.zeros((32, 800, 6))
        res_coarse, res_fine, normal_coarse, normal_fine, feature_coarse, feature_fine = get_model(
            inputs, idx)
        print np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print 'a'