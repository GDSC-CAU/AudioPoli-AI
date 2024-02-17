import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from capsulelayers import CapsuleLayer, PrimaryCap, Distance, Mask
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys; sys.argv=['']; del sys
import argparse
import h5py
import time
import re
import os

"""
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras import backend as K
from keras.utils import to_categorical
from capsulelayers import CapsuleLayer, PrimaryCap, Distance, Mask
from keras.backend.tensorflow_backend import set_session
import seaborn as sn
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import sys; sys.argv=['']; del sys
import argparse
import h5py
import time
import re
import os
"""
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
np.random.seed(1337)
"""
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)
"""
K.set_image_data_format('channels_last')

label = 16 # total label
height = 48 # Input image height
width = 173 # Input image width
SR = 44100 # [Hz] sampling rate
max_len = 4.0
max_len = int(max_len)
n_fft = 2048
n_hop = 1024
n_mfcc = 48
len_raw = int(SR * max_len)

model_save = './save_result/'
save_HDF = './testset_hdf/'
PATH_US = './testset'

# CapsNet margin loss
def margin_loss(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L, 1))

# predict
def test_one(model, data, args):
    x_test, y_test = data
    start = time.time()
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('\nPrediction response time : ', time.time() - start)
    print('Test accuracy:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0])

# preprocess data
def detect_index(name):
    index = name.split('.')[1].split('_')[0]
    if index == '강제추행(성범죄)':
        index = 1
    elif index == '강도범죄':
        index = 2
    elif index == '절도범죄':
        index = 3
    elif index == '폭력범죄':
        index = 4
    elif index == '화재':
        index = 5
    elif index == '갇힘':
        index = 6
    elif index == '응급의료':
        index = 7
    elif index == '전기사고':
        index = 8
    elif index == '가스사고':
        index = 9
    elif index == '낙상':
        index = 10
    elif index == '붕괴사고':
        index = 11
    elif index == '태풍-강풍':
        index = 12
    elif index == '지진':
        index = 13
    elif index == '도움요청':
        index = 14
    elif index == '실내':
        index = 15
    elif index == '실외':
        index = 16
    else:
        pass
    return index

def test_graph(model, data, args):
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
    ''' A:강제추행, B:강도범죄, C:절도범죄, D:폭력범죄, 
    E:화재, F:갇힘, G:응급의료, H:전기사고, 
    I:가스사고, J:낙상, K:붕괴사고, L:태풍/강풍, 
    M:지진, N:도움요청, O:실내, P:실외 
    '''
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=20)
    cm = confusion_matrix(np.argmax(y_test, 1), np.argmax(y_pred, 1))
    print('\nCategory Classification Report\n')
    print(classification_report(np.argmax(y_test, 1), np.argmax(y_pred, 1), target_names=labels))
    print('\nConfusion Matrix graph saved', model_save)
    return cm, labels

# read HDF5 file
def test_dataset():
    x_train_mfcc = []
    y_train_mfcc = []
    for i in range(0, label):  # 1~16 label
        for ds_name in ['mfcc', 'y']:
            if ds_name == 'mfcc':
                count = "mfcc_y_%ix%i_%i" % (height, width, i)
                mfcc = h5py.File(save_HDF + count + '.h5', 'r')
                x_train_mfcc.extend(mfcc[ds_name])
            if ds_name == 'y':
                count = "mfcc_y_%ix%i_%i" % (height, width, i)
                mfcc = h5py.File(save_HDF + count + '.h5', 'r')
                y_train_mfcc.extend(mfcc[ds_name])
    # reshape
    test_x = np.array(x_train_mfcc)
    test_y = np.array(y_train_mfcc)
    test_x = test_x.reshape(-1, height, width, 1)
    test_y = np.argmax(test_y, axis=1).reshape(-1)
    test_y = test_y[:, None]
    test_y = to_categorical(test_y.astype('float32'))
    return (test_x, test_y)

def main():
    parser = argparse.ArgumentParser(description="Capsule Network on Dataset.")
    parser.add_argument('--save_dir', default=model_save)
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load model
    eval_model = load_model('./eval_2',
                            custom_objects={'CapsuleLayer': CapsuleLayer, 'Mask': Mask, 'Distance': Distance,
                                            'PrimaryCab': PrimaryCap, 'margin_loss': margin_loss})
    eval_model.summary()

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

    # load test-dataset
    (x_test, y_test) = test_dataset()
    eval_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # eval_model.fit(x_test, y_test, batch_size=256, epoch=30)
    eval_model.save('./eval.h5')

    """
    # predict test-dataset
    test_one(model=eval_model, data=(x_test, y_test), args=args)
    # prediction result
    cm_test, labels = test_graph(model=eval_model, data=(x_test, y_test), args=args)
    # save predict values .csv
    cm_test.tofile(args.save_dir + '/cm_test.csv',',')
    # load predict values .csv
    test = np.loadtxt(open(args.save_dir+'/cm_test.csv', "rb"), delimiter=",", skiprows=0)
    # dataframe
    t = test.reshape(label,label)
    df_cm = pd.DataFrame(t, labels, labels)
    # heatmap
    plt.figure(figsize = (16,8))
    svn = sn.heatmap(df_cm, annot=True, cmap='Blues', annot_kws={"size": 14}, fmt='g', linewidths=.5)
    figure = svn.get_figure()
    figure.savefig(args.save_dir+'sv_conf.png', dpi=400)
    """

if __name__ == "__main__":
    main()