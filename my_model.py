import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import my_utils
import my_nn
import argparse
import os
import my_preprocessing
import pickle
from PIL import Image

args_seqlen = None
args_channels = 128
args_num_attlayers = 4
args_show = False
args_writersource ='assets/writer6.png'
args_weights = 'weights/gen14/model_step100000.h5'
args_diffmode = 'new'

width = 1600
if args_writersource is None:
    assetdir = os.listdir('./assets')
    sourcename = './assets/' + assetdir[np.random.randint(0, len(assetdir))]
else: 
    sourcename = args_writersource

L = 30
tokenizer = my_utils.Tokenizer('data/my_translation.pkl')
beta_set = my_utils.get_beta_set(L=L)
alpha_set = tf.math.cumprod(1-beta_set)

C1 = args_channels
C2 = C1 * 3//2
C3 = C1 * 2
style_extractor = my_nn.StyleExtractor(96, width, 3)
model = my_nn.DiffusionWriter(num_layers=args_num_attlayers, c1=C1, c2=C2, c3=C3, d_emb=len(tokenizer.chars))

_stroke = tf.random.normal([1, 480, 2])
_text = tf.random.uniform([1, 40], dtype=tf.int32, maxval=len(tokenizer.chars))
_noise = tf.random.uniform([1, 1])
_style_vector = tf.random.normal([1, 14, 1280])
_ = model(_stroke, _text, _noise, _style_vector)
#we have to call the model on input first
model.load_weights(args_weights)

# writer_img = tf.expand_dims(preprocessing.read_img(sourcename, 96), 0)
org_img = my_preprocessing.read_img(sourcename, 96)
org_img = my_utils.pad_img(org_img, width, 96)
writer_img = tf.expand_dims(org_img, 0)

style_vector = style_extractor(writer_img)