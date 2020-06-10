import numpy as np
import h5py
from tqdm import tqdm
import os

def patch_to_h5(test_data_folder, output_name, with_index=False):

    items = os.listdir(test_data_folder)
    facet_num = len(items)
    normal_data = np.zeros(shape=(facet_num, 800, 10, 3), dtype="float32")
    idx_data = np.zeros(shape=(facet_num, 800, 50), dtype="int32")

    if with_index == True:
        index_data = np.zeros(shape=(facet_num, 800), dtype="int32")

    for item in tqdm(range(facet_num)):
        txt_name = output_name + '_' + str(item+1) + '.txt'
        model_path = os.path.join(test_data_folder, txt_name)
        data = np.loadtxt(model_path)  # [800, 80]
        normal_data_patch = data[:, 0:30]
        normal_data_patch = np.reshape(normal_data_patch, newshape=(800, 10, 3))
        idx_data_patch = data[:, 30:80]
        normal_data[item, :, :, :] = normal_data_patch
        idx_data[item, :, :] = idx_data_patch
        if with_index == True:
            index = data[:, -1]
            index_data[item, :] = index

    output_file = '../../data/test/input/' + output_name + '.h5' # output path of h5 file
    f = h5py.File(output_file, 'w')
    f['data'] = normal_data
    f['index'] = idx_data
    if with_index==True:
        f['face_index'] = index_data
    f.close()

if __name__=='__main__':
    test_data_folder = '../../data/test/carter100K_n1/'  # this folder contains all the patch files
    output_name = 'carter100K_n1'
    patch_to_h5(test_data_folder, output_name, with_index=False)