from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import numpy as np
import h5py
import librosa
import librosa.display
import sys
import time
import pandas as pd
import re
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
set_session(sess)

SR = 44100 # [Hz]
max_len = 4.0
max_len = int(max_len)
n_fft = 2048
n_hop = 1024
n_mfcc = 48
len_raw = int(SR * max_len)
n_label = 19 # label category number 0~18

PATH_US = './testset'
save_HDF = './testset_hdf/'

def detect_index(name):
    index = name.split('.')
    return index[0]

dirlist = []
index = []
filename = []
for root, dirs, files in os.walk(PATH_US):
    for name in files:
        matchobj = re.findall('\d+', name)
        if (len(matchobj) == 2):
            filename.append(name)
            dirlist.append(os.path.join(root, name))
            index.append(detect_index(name))

df = pd.DataFrame({'classID': index, 'dir_filelist': dirlist, 'slice_filename': filename})
mfcc_shape = librosa.feature.mfcc(np.zeros(len_raw), SR, n_fft=n_fft, hop_length=n_hop, n_mfcc=n_mfcc).shape
n_mfcc_fr = mfcc_shape[1]
df['classID'] = df['classID'].astype(int)
n_data_all = df.shape[0]

def create_dataset_for(f_hdf, ds_name, num_data):
    if ds_name == 'mfcc':
        return f_hdf.create_dataset('mfcc', (num_data, n_mfcc, n_mfcc_fr), dtype='float32')
    else:
        print('ha? %s?' % ds_name)

def row_to_y(row_idx, row, dataset, label):
    y = label
    dataset[row_idx, y] = True

def create_dataset_for_test(f_hdf, ds_name, num_data):
    if ds_name == 'mfcc':
        return f_hdf.create_dataset('mfcc', (num_data, n_mfcc, n_mfcc_fr), dtype='float32')
    elif ds_name == 'y':
        return f_hdf.create_dataset('y', (num_data, n_label), dtype='bool')

def row_to_test(ds_name, row_idx, row, dataset, label):
    if ds_name == 'mfcc':
        row_to_mfcc_test(row_idx, row, dataset)
    elif ds_name == 'y':
        row_to_y(row_idx, row, dataset, label)

def row_to_mfcc_test(row_idx, row, dataset):
    '''
    row: row of dataframe of pandas
    dataset: a dataset of hdf file '''
    fname = row[2]
    src, sr = librosa.load(fname, SR)
    mfcc = librosa.feature.mfcc(src, sr, n_fft=n_fft,
                                hop_length=n_hop, n_mfcc=n_mfcc)
    dataset[row_idx, :, :min(n_mfcc_fr, mfcc.shape[1])] = mfcc[:, :n_mfcc_fr]

def set_to_hdf_test(hdf_filepath, df_subset, shfl_idxs, ds_name, label):
    assert len(df_subset) == len(shfl_idxs), 'data frame length != indices list'
    start_time = time.time()
    num_data = len(df_subset)
    if os.path.exists(hdf_filepath):
        mode = 'a'
    else:
        mode = 'w'
    print("------------------------Hello, world!----------------------")
    with h5py.File(hdf_filepath, mode) as f_hdf:
        dataset = create_dataset_for_test(f_hdf, ds_name, num_data)
        for row_idx, row in enumerate(df_subset.iloc[shfl_idxs].itertuples()):
#            print(ds_name, row_idx, row)
            row_to_test(ds_name, row_idx, row, dataset, label)
            if row_idx % 20 == 0:
                sys.stdout.write('\r%d/%d-th sample (%s) was written.' % (row_idx+1, num_data, ds_name))
    print('\n--- Done: It took %d seonds for %s, %s ---' %  (int(time.time() - start_time), ds_name, hdf_filepath.split('/')[-1]))

def main():
    for ds_name in ['y', 'mfcc']:
        cnt = -1
        for i in range(0, n_label):  # label class
            train_fr = df[df['classID'] == i]
            if (len(train_fr) == 0):
                continue
            else:
                cnt = cnt + 1
            np.random.seed(1337)
            train_shfl_idxs = np.random.permutation(len(train_fr))
            print(train_shfl_idxs)
            count = "mfcc_y_%ix%i_%i" % (n_mfcc, n_mfcc_fr, cnt)
            try:
                set_to_hdf_test(save_HDF + count + '.h5', train_fr, train_shfl_idxs, ds_name, cnt)
            except Exception as e:
                print(e)

if __name__ == "__main__":
    main()