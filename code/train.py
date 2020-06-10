import argparse
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
from tqdm import tqdm
import h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import model_util

def str2bool(x):
    return x.lower() in ('true')

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model')
parser.add_argument('--log_dir', default='log_synthetic', help='Log dir [default: log]')
parser.add_argument('--max_epoch', type=int, default=401, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=10, help='Batch Size during training [default: 32]')
parser.add_argument('--patch_size', type=int, default=800, help='Batch Size during training [default: 32]')
parser.add_argument('--neighbor_size', type=int, default=50, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=500000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for lr decay [default: 0.8]')
parser.add_argument('--restore', type=str2bool, default=False)

FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
PATCH_SIZE = FLAGS.patch_size
NEIGHBOR_SIZE = FLAGS.neighbor_size
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_INDEX)

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(BASE_DIR, FLAGS.model + '.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
os.system('cp train.py %s' % (LOG_DIR))  # bkp of train procedure

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

f_gt = h5py.File('../data/synthetic_gt_facet_800_50_idx_v1.h5')

gt_data = f_gt['data'][:]  # [6000, 800, 3]
f_gt.close()
f_noise = h5py.File('../data/synthetic_noisy_facet_800_50_idx_v1.h5')

input_data = f_noise['data'][:]  # [6000, 800, 10, 3]
input_idx = f_noise['index'][:]  # [6000, 800, 50]
f_noise.close()


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            input_data_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, PATCH_SIZE, 10, 3))
            input_idx_pl = tf.placeholder(tf.int32, shape=(BATCH_SIZE, PATCH_SIZE, NEIGHBOR_SIZE))
            gt_data_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, PATCH_SIZE, 3))
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            alpha = tf.train.piecewise_constant(batch, [30000, 60000, 90000],
                                                [1.0, 1.0, 1.0, 1.0])

            # Get model and loss
            res_coarse, res_fine, normal_coarse, normal_fine, _, _ = MODEL.get_model(
                input_data_pl,
                input_idx_pl, is_training_pl,
                bn_decay=bn_decay,
                sizes=[10, 30, 50])
            loss_coarse, loss_fine, loss_total = MODEL.get_loss(normal_coarse, normal_fine, gt_data_pl,
                                                                           res_coarse, res_fine, alpha)

            tf.summary.scalar('loss/loss_coarse', loss_coarse)
            tf.summary.scalar('loss/loss_fine', loss_fine)
            tf.summary.scalar('loss/loss_total', loss_total)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss_total, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=11)


        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        # merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                             sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        # sess.run(init)
        sess.run(init, {is_training_pl: True})

        restore_epoch, checkpoint_path = model_util.pre_load_checkpoint(LOG_DIR)
        global LOG_FOUT
        if restore_epoch == 0:
            LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
            LOG_FOUT.write(str(socket.gethostname()) + '\n')
            LOG_FOUT.write(str(FLAGS) + '\n')
        else:
            LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
            saver.restore(sess, checkpoint_path)

            print(checkpoint_path)

        ops = {'input_data_pl': input_data_pl,
               'input_idx_pl': input_idx_pl,
               'gt_data_pl': gt_data_pl,
               'is_training_pl': is_training_pl,
               'normal_coarse': normal_coarse,
               'normal_fine': normal_fine,
               'res_coarse': res_coarse,
               'res_fine': res_fine,
               'loss_coarse': loss_coarse,
               'loss_fine': loss_fine,
               'loss_total': loss_total,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in tqdm(range(restore_epoch+1, MAX_EPOCH)):
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)

            # Save the variables to disk.
            if epoch % 5 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), global_step=epoch)
                print("Model saved in file: %s" % save_path)


def nonuniform_sampling(num=4096, sample_num=1024):
    sample = set()
    loc = np.random.rand() * 0.8 + 0.1
    while (len(sample) < sample_num):
        a = int(np.random.normal(loc=loc, scale=0.3) * num)
        if a < 0 or a >= num:
            continue
        sample.add(a)

    return list(sample)

def rotate_point_cloud_and_gt(batch_data,batch_gt=None,z_rotated=True):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    B = batch_data.shape[0]
    N = batch_data.shape[1]

    for k in range(B):
        angles = np.random.uniform(size=(3)) * 2 * np.pi
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        if z_rotated:
            rotation_matrix = Rz
        else:
            rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))

        # rotation_angle = np.random.uniform(size=(3)) * 2 * np.pi
        # cosval = np.cos(rotation_angle)
        # sinval = np.sin(rotation_angle)
        # rotation_matrix = np.array([[cosval, 0, sinval],
        #                             [0, 1, 0],
        #                             [-sinval, 0, cosval]])

        batch_data[k,...] = np.dot(batch_data[k,...].reshape((-1, 3)), rotation_matrix).reshape([N,-1,3])

        if batch_gt is not None:
            batch_gt[k,...] = np.dot(batch_gt[k,...].reshape((-1, 3)), rotation_matrix).reshape([N,3])

    return batch_data,batch_gt

def jitter_perturbation_point_cloud(batch_data, sigma=0.005, clip=0.02):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, P, C = batch_data.shape
    # assert(clip > 0)
    # jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)

    clip = 2*sigma
    import scipy.stats as st
    jittered_data = np.clip(st.norm(0, sigma).rvs([B, N, P, C]), -1 * clip, clip)

    jittered_data += batch_data
    return jittered_data

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    global input_data, gt_data, input_idx

    file_size = input_data.shape[0]

    batch_num = file_size // BATCH_SIZE

    # Shuffle train files
    idx = np.arange(file_size)
    np.random.shuffle(idx)
    input_data = input_data[idx, ...]
    input_idx = input_idx[idx, ...]
    gt_data = gt_data[idx, ...]

    loss_coarse_sum = 0
    loss_fine_sum = 0
    loss_total_sum = 0

    for batch_idx in range(batch_num):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

        current_input_data = input_data[start_idx:end_idx, :, :, :]
        current_input_idx = input_idx[start_idx:end_idx, :, :]
        current_gt_data = gt_data[start_idx:end_idx, :, :]

        # Augment batched point clouds by rotation and jittering
        current_input_data, current_gt_data = rotate_point_cloud_and_gt(current_input_data, current_gt_data)
        current_input_data = jitter_perturbation_point_cloud(current_input_data)

        feed_dict = {ops['input_data_pl']: current_input_data,
                     ops['input_idx_pl']: current_input_idx,
                     ops['gt_data_pl']: current_gt_data,
                     ops['is_training_pl']: is_training, }
        summary, step, _, loss_coarse, loss_fine, loss_total = sess.run([ops['merged'], ops['step'],
                                                         ops['train_op'], ops['loss_coarse'],
                                                         ops['loss_fine'], ops['loss_total']],
                                                        feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        loss_coarse_sum += loss_coarse
        loss_fine_sum += loss_fine
        loss_total_sum += loss_total

    print('loss_coarse: %f' % (loss_coarse_sum / float(batch_num)))
    print('loss_fine: %f' % (loss_fine_sum / float(batch_num)))
    print('loss_total: %f' % (loss_total_sum / float(batch_num)))


if __name__ == "__main__":
    train()
