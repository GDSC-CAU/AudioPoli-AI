import os

from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio

import glob
import json
import re
import soundfile as sf

from sklearn import model_selection
from tensorflow.keras import layers, models


yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

# Utility functions for loading audio files and making sure the sample rate is correct.

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

class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
class_names =list(pd.read_csv(class_map_path)['display_name'])

def process_json_file(json_file):
    target = re.split('[/._]', json_file)[-2]
    result = [file for file in train_orig_files \
              if "_{0}_".format(target) in file]
    result = result[0] if len(result) > 0 else None
    data = json.load(open(json_file))
    data = pd.json_normalize(data)
    annotations = data['annotations'][0]
    # print(annotations)
    data = []
    for annotation in annotations:
        item = {}
        item['filename'] = result
        item['target'] = target
        item['category'] = annotation['categories']['category_02']
        item['audio_type'] = annotation['audioType']
        data.append(item)

    # print("{0}\n{1}\n".format(json_file, result))
    # print("{0} {1}_{2}".format(target, re.split('[/.\[\]_]', result)[-4], re.split('[/._]', result)[-1]))
    return data

def load_wav_for_map(filename, label, fold):
  return load_wav_16k_mono(filename), label, fold

def extract_embedding(wav_data, label, fold):
  ''' run YAMNet to extract embedding from the wav data '''
  scores, embeddings, spectrogram = yamnet_model(wav_data)
  num_embeddings = tf.shape(embeddings)[0]
  return (embeddings,
            tf.repeat(label, num_embeddings),
            tf.repeat(fold, num_embeddings))




base_train_dir = "/data/GDSC_AudioPoli/testset/Training/label/"
base_valid_dir = "/data/GDSC_AudioPoli/testset/Validation/label/"
base_train_audio = '/data/GDSC_AudioPoli/testset/Training/orig/'
base_valid_audio = '/data/GDSC_AudioPoli/testset/Validation/orig/'

output_dropna_csv = '/data/GDSC_AudioPoli/AudioPoli-AI/output_dropna.csv'

# Find dataset directories via glob 
"""
train_json_files = glob.glob(os.path.join(base_train_dir, '*/*.json'))
valid_json_files = glob.glob(os.path.join(base_valid_dir, '*/*.json'))
train_orig_files = glob.glob(os.path.join(base_train_audio, '*/*.wav'), recursive=True)
valid_orig_files = glob.glob(os.path.join(base_valid_audio, '*/*.wav'), recursive=True)
"""

pd_data = pd.read_csv(output_dropna_csv)

my_classes = ['강제추행(성범죄)', '강도범죄', '절도범죄', '폭력범죄',
              '화재', '갇힘', '응급의료', '전기사고', '가스사고', '낙상', 
              '붕괴사고', '태풍-강풍', '지진', '도움요청', '실내', '실외'
             ]
map_class_to_id = {'강제추행(성범죄)':1, '강도범죄':2, '절도범죄':3, '폭력범죄':4,
              '화재':5, '갇힘':6, '응급의료':7, '전기사고':8, '가스사고':9, '낙상':10, 
              '붕괴사고':11, '태풍-강풍':12, '지진':13, '도움요청':14, '실내':15, '실외':16}

filtered_pd = pd_data[pd_data.category.isin(my_classes)]

class_id = filtered_pd['category'].apply(lambda name: map_class_to_id[name])
filtered_pd = filtered_pd.assign(category=class_id)

# KFold (n = 5)
filtered_pd['fold'] = -1
kf = model_selection.KFold(n_splits = 5)
for fold, (trn_, val_) in enumerate(kf.split(X=filtered_pd)):
    filtered_pd.loc[val_, 'fold'] = fold

# Audio File Embeddings

filenames = filtered_pd['filename']
targets = filtered_pd['category']
folds = filtered_pd['fold']

main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets, folds))
main_ds = main_ds.map(load_wav_for_map)
main_ds = main_ds.map(extract_embedding).unbatch()

# K-fold

cached_ds = main_ds.cache()
train_ds = cached_ds.filter(lambda embedding, label, fold: fold < 4)
val_ds = cached_ds.filter(lambda embedding, label, fold: fold == 4)
test_ds = cached_ds.filter(lambda embedding, label, fold: fold == 5)

# remove the folds column now that it's not needed anymore
remove_fold_column = lambda embedding, label, fold: (embedding, label)

train_ds = train_ds.map(remove_fold_column)
val_ds = val_ds.map(remove_fold_column)
test_ds = test_ds.map(remove_fold_column)

train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)

scores, embeddings, spectrogram = yamnet_model(testing_wav_data)
result = my_model(embeddings).numpy()

inferred_class = my_classes[result.mean(axis=0).argmax()]
print(f'The main sound is: {inferred_class}')
