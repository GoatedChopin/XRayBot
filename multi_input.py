import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def stat_scaler(tensor):
    means = tf.constant([46.90146271851587, 0.4350695683196575, 0.39966107741705315])
    std_devs = tf.constant([16.839922533380808, 0.4957683241322746, 0.4898308285442431])
    return (tensor - means) / std_devs


def build_mi_nn(pic_shape=(1024,1024,3), num_stats=3):
    # CNN
    input_pic = layers.Input(shape=pic_shape)
    x = MobileNet(input_shape=pic_shape, include_top=False)(input_pic)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10, activation='relu')(x)
    x = Model(inputs=input_pic, outputs=x)
    
    # stats
    input_stats = layers.Input(shape=(num_stats,))
    w = layers.Lambda(stat_scaler)(input_stats)
    w = layers.Dense(32, activation='relu')(w)
    w = layers.Dense(10, activation='relu')(w)
    w = Model(inputs=input_stats, outputs=w)
    
    # concat both layers
    combined = layers.concatenate([x.output, w.output])
    z = layers.Dense(13, activation='relu')(combined)
    model = Model(inputs=[x.input, w.input], outputs=z)
    
    model.compile(optimizer = "Adam", loss = "SparseCategoricalCrossEntropy", metrics=["acc"])
    return model

def convert_bw_rgb(pic_array):
    return np.stack((pic_array, pic_array, pic_array), axis=2)