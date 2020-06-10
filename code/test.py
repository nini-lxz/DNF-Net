import tensorflow as tf
import numpy as np
import argparse
import h5py
import importlib
import os
import sys
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import model_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model')
parser.add_argument('--batch_size', type=int, default=40, help='Batch Size during training [default: 1]')
parser.add_argument('--patch_size', type=int, default=800, help='Batch Size during training [default: 1]')
parser.add_argument('--neighbor_size', type=int, default=50, help='Batch Size during training [default: 32]')
parser.add_argument('--log_dir', default='log/log_synthetic')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
PATCH_SIZE = FLAGS.patch_size
NEIGHBOR_SIZE = FLAGS.neighbor_size
GPU_INDEX = FLAGS.gpu
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_INDEX)

MODEL = importlib.import_module(FLAGS.model)  # import network module
LOG_DIR = FLAGS.log_dir


def evaluate(test_data, test_neighbor, with_index):

    with tf.device('/gpu:' + str(GPU_INDEX)):
        input_data_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, PATCH_SIZE, 10, 3))
        input_neighbor_pl = tf.placeholder(tf.int32, shape=(BATCH_SIZE, PATCH_SIZE, NEIGHBOR_SIZE))
        is_training_pl = tf.placeholder(tf.bool, shape=())

        _, _, _, pred_normal, _, _ = MODEL.get_model(input_data_pl, input_neighbor_pl, is_training_pl)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    # config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    _, restore_model_path = model_util.pre_load_checkpoint(LOG_DIR)
    saver.restore(sess, restore_model_path)
    print("Model restored: %s" % restore_model_path)

    ops = {'input_data_pl': input_data_pl,
           'input_neighbor_pl': input_neighbor_pl,
           'is_training_pl': is_training_pl,
           'pred_normal': pred_normal
           }

    if with_index == False:
        pred_normals = eval_one_epoch(sess, ops, test_data, test_neighbor)
    else:
        pred_normals = eval_one_epoch_idx(sess, ops, test_data, test_neighbor)

    sess.close()

    return pred_normals


def eval_one_epoch(sess, ops, test_data, test_neighbor):
    is_training = False
    file_size = test_data.shape[0]
    pred_normals = np.zeros(shape=(file_size, 3), dtype="float32")

    num_batches = file_size // BATCH_SIZE
    for batch_idx in tqdm(range(num_batches+1)):
        if batch_idx != num_batches:
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE
        else:
            start_idx = file_size - BATCH_SIZE
            end_idx = file_size
        current_test_data = test_data[start_idx:end_idx, :, :, :]
        current_test_neighbor = test_neighbor[start_idx:end_idx, :, :]

        feed_dict = {ops['input_data_pl']: current_test_data,
                     ops['input_neighbor_pl']: current_test_neighbor,
                     ops['is_training_pl']: is_training}
        pred_normal = sess.run([ops['pred_normal']], feed_dict=feed_dict)
        pred_normal = pred_normal[0]
        pred_normals[start_idx:end_idx, :] = pred_normal[:, 0, :]  # only use the center facet

    return pred_normals

def eval_one_epoch_idx(sess, ops, test_data, test_neighbor):
    is_training = False
    file_size = test_data.shape[0]
    pred_normals = np.zeros(shape=(file_size, 800, 3), dtype="float32")

    num_batches = file_size // BATCH_SIZE
    for batch_idx in tqdm(range(num_batches+1)):
        if batch_idx != num_batches:
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE
        else:
            start_idx = file_size - BATCH_SIZE
            end_idx = file_size

        current_test_data = test_data[start_idx:end_idx, :, :, :]
        current_test_neighbor = test_neighbor[start_idx:end_idx, :, :]

        feed_dict = {ops['input_data_pl']: current_test_data,
                     ops['input_neighbor_pl']: current_test_neighbor,
                     ops['is_training_pl']: is_training}
        pred_normal = sess.run([ops['pred_normal']], feed_dict=feed_dict)
        pred_normal = pred_normal[0]

        pred_normals[start_idx:end_idx, :, ] = pred_normal[:, :, :]

    return pred_normals

def normal_avg(pred_normals, test_idx):

    file_size = pred_normals.shape[0]
    pred_normals_new = np.zeros(shape=(file_size, 3), dtype="float32")

    for index in range(file_size):
        indices = np.where(test_idx==index)
        indices_x = indices[0]
        indices_y = indices[1]
        normals = pred_normals[indices_x, indices_y, :]  # [N, 3]
        normals_sum = np.sum(normals, axis=0)  # [3,]
        repeat_num = normals.shape[0]
        normals_avg = normals_sum / np.float(repeat_num)
        normals_avg_normalize = normals_avg / np.sqrt(np.sum(np.square(normals_avg)))

        pred_normals_new[index, :] = normals_avg_normalize

    return pred_normals_new

if __name__ == '__main__':

    test_name = 'carter100K_n1'
    with_index = False
    input_path = 'data/test/input/' + test_name + '.h5'
    f = h5py.File(input_path)
    test_data = f['data'][:]  # [B, 800, 10, 3]
    test_neighbor = f['index'][:]  #[B, 800, 50]

    with tf.Graph().as_default():
        pred_normals = evaluate(test_data, test_neighbor, with_index)   # True: [B, 800, 3]; False: [B, 3]

    if with_index == True:
        test_idx = f['face_index'][:]  # [B, 800]
        pred_normals = normal_avg(pred_normals, test_idx)

    # save to files
    output_path = 'data/test/output/' + test_name + "_NN.txt"
    np.savetxt(output_path, pred_normals)
