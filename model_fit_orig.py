import os

from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
import soundfile as sf

import glob
import json
import re
import gc

from sklearn import model_selection
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import Callback

os.environ["CUDA_VISIBLE_DEVICES"]='-1'

"""
import warnings
warnings.filterwarnings('ignore')
"""

@tf.function
def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

def load_wav_for_map(filename, label, fold):
  return load_wav_16k_mono(filename), label, fold

# applies the embedding extraction model to a wav data
def extract_embedding(wav_data, label, fold):
  ''' run YAMNet to extract embedding from the wav data '''
  scores, embeddings, spectrogram = yamnet_model(wav_data)
  num_embeddings = tf.shape(embeddings)[0]
  return (embeddings,
            tf.repeat(label, num_embeddings),
            tf.repeat(fold, num_embeddings))
    
class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        print("Cleaning garbage")
        tf.keras.backend.clear_session()

class MyModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
    
    def build_model(self):
        input = layers.Input(shape=self.input_shape)
        x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(input)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(input)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(self.num_classes)(x)
        return models.Model(inputs=input, outputs=x)

    def compile_model(self):
        self.model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           optimizer="adam",
                           metrics=['accuracy'])

    def fit_model(self, train_ds, val_ds, epochs=10, batch_size=32):
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                    patience=3,
                                                    restore_best_weights=True)
        """
        history = self.model.fit(train_ds,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=val_ds,
                                 callbacks=[ClearMemory(), callback],
                                 steps_per_epoch=folds.count()//batch_size
                                )
        """
        history = self.model.fit(train_ds,
                                 epochs=epochs,
                                 validation_data=val_ds,
                                 callbacks=[ClearMemory(), callback],
                                 use_multiprocessing=True,
                                 workers=20,
                                 steps_per_epoch=folds.count()//batch_size
                                )
        
        return history

class ReduceMeanLayer(tf.keras.layers.Layer):
    def __init__(self, axis=0, **kwargs):
        super(ReduceMeanLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, input):
        return tf.math.reduce_mean(input, axis=self.axis)

yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

base_train_dir = "/data/GDSC_AudioPoli/testset/Training/label/"
base_valid_dir = "/data/GDSC_AudioPoli/testset/Validation/label/"
base_train_audio = '/data/GDSC_AudioPoli/testset/Training/orig/'
base_valid_audio = '/data/GDSC_AudioPoli/testset/Validation/orig/'

train_json_files = glob.glob(os.path.join(base_train_dir, '*/*.json'))
valid_json_files = glob.glob(os.path.join(base_valid_dir, '*/*.json'))
train_orig_files = glob.glob(os.path.join(base_train_audio, '*/*.wav'), recursive=True)
valid_orig_files = glob.glob(os.path.join(base_valid_audio, '*/*.wav'), recursive=True)

output_csv = '/data/GDSC_AudioPoli/AudioPoli-AI/output.csv'
output_valid_csv = '/data/GDSC_AudioPoli/AudioPoli-AI/output_valid.csv'


pd_data = pd.read_csv(output_csv)
pd_data

output_dropna_csv = '/data/GDSC_AudioPoli/AudioPoli-AI/output_dropna.csv'
pd_data = pd.read_csv(output_dropna_csv)

my_classes = ['강제추행(성범죄)', '강도범죄', '절도범죄', '폭력범죄',
              '화재', '갇힘', '응급의료', '전기사고', '가스사고', '낙상', 
              '붕괴사고', '태풍-강풍', '지진', '도움요청', '실내', '실외'
             ]
map_class_to_id = {'강제추행(성범죄)':0, '강도범죄':1, '절도범죄':2, '폭력범죄':3,
              '화재':4, '갇힘':5, '응급의료':6, '전기사고':7, '가스사고':8, '낙상':9, 
              '붕괴사고':10, '태풍-강풍':11, '지진':12, '도움요청':13, '실내':14, '실외':15}

filtered_pd = pd_data[pd_data.category.isin(my_classes)]

class_id = filtered_pd['category'].apply(lambda name: map_class_to_id[name])
filtered_pd = filtered_pd.assign(category=class_id)

# KFold (n = 5)
filtered_pd['fold'] = -1
kf = model_selection.StratifiedKFold(n_splits = 16)
for fold, (trn_, val_) in enumerate(kf.split(X=filtered_pd, y=filtered_pd['category'])):
    filtered_pd.loc[val_, 'fold'] = fold

# filtered_pd['Fold'].unique()
    
filtered_pd.head(10)
filtered_pd['fold'].count()
# filtered_pd['fold'].unique()
# filtered_pd[filtered_pd.fold == 0]

filenames = filtered_pd['filename']
targets = filtered_pd['category']
folds = filtered_pd['fold']

main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets, folds))
main_ds = main_ds.map(load_wav_for_map)
main_ds = main_ds.map(extract_embedding).unbatch()

cached_ds = main_ds.cache()
train_ds = cached_ds.filter(lambda embedding, label, fold: fold < 14)
val_ds = cached_ds.filter(lambda embedding, label, fold: fold == 14)
test_ds = cached_ds.filter(lambda embedding, label, fold: fold == 15)

# remove the folds column now that it's not needed anymore
remove_fold_column = lambda embedding, label, fold: (embedding, label)

train_ds = train_ds.map(remove_fold_column)
val_ds = val_ds.map(remove_fold_column)
test_ds = test_ds.map(remove_fold_column)

train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)

saved_model_path = './audiopoli_proto'


input_shape = (1024,)
num_classes = len(my_classes)
my_model = MyModel(input_shape, num_classes)
my_model.compile_model()
history = my_model.fit_model(train_ds, val_ds, epochs=5)
my_model.save("seq_model.h5")

input_segment = tf.keras.layers.Input(shape=(), dtype=tf.float32, name='audio')
embedding_extraction_layer = hub.KerasLayer(yamnet_model_handle,
                                            trainable=False, name='yamnet')
_, embeddings_output, _ = embedding_extraction_layer(input_segment)
serving_outputs = my_model(embeddings_output)
serving_outputs = ReduceMeanLayer(axis=0, name='classifier')(serving_outputs)
serving_model = tf.keras.Model(input_segment, serving_outputs)
serving_model.save(saved_model_path, include_optimizer=False)
