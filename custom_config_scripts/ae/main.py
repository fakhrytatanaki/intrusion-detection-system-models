#--rocother---

import json
import sys
sys.path.append('../..') #for importing parent modules
import os
import importlib
from glob import glob
import ids_main.preprocessing as pproc
import optuna
from ids_main.perf import calc_auroc
from ids_main.cic_flow_meter_data_scripts import AutoencoderEvaluation
#--------


#--sklearn---
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,confusion_matrix,confusion_matrix
from ids_main.models.autoencoders.variational_autoencoder import VariationalAutoencoder
from ids_main.models.autoencoders.autoencoder import Autoencoder
from ids_main.models.autoencoders.common import reconstruction_error_funcs
from ids_main.common import train_nn_model
#--------

#--scipy---
import scipy
#--------

#--numpy---
import numpy as np
from numpy.random import shuffle,choice
#--------


#--tensorflow---
import tensorflow as tf
from tensorflow.keras import layers, losses,regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
#--------

#--pandas-and-dask---
import pandas as pd
import dask
import dask.dataframe as dd
import dask.array as da
import dask.bag as db
#--------

#--plotting--
import seaborn as sns
import matplotlib.pyplot as plt
#--------




with open('./optuna_config.json') as fp:
    proj_config = json.load(fp)


def ae_constructor(num_neurons_per_layer,cfg,ae_class,reg_constants_per_layer=None):

    if reg_constants_per_layer!=None:
        assert(len(reg_constants_per_layer)==len(num_neurons_per_layer))

    encoder_sub = tf.keras.Sequential([])
    decoder_sub = tf.keras.Sequential([])


    for i,num_neurons in enumerate(num_neurons_per_layer):
            if reg_constants_per_layer==None:
                encoder_sub.add(layers.Dense(num_neurons, activation='relu'))
            else:
                encoder_sub.add(layers.Dense(num_neurons, activation='relu',kernel_regularizer=regularizers.L2(reg_constants_per_layer[i])))


    for i,num_neurons in enumerate(reversed(num_neurons_per_layer)):
            if reg_constants_per_layer==None:
                decoder_sub.add(layers.Dense(num_neurons, activation='relu'))
            else:
                decoder_sub.add(layers.Dense(num_neurons, activation='relu',kernel_regularizer=regularizers.L2(reg_constants_per_layer[len(reg_constants_per_layer) - i - 1])))


    decoder_sub.add(layers.Dense(cfg['data_vec_size'], activation='relu'))

    ae = ae_class(encoder_sub,decoder_sub,cfg['data_vec_size'],cfg['latent_vec_size'])
    ae.compile(optimizer=Adam(learning_rate=cfg['learning_rate']),loss=losses.BinaryCrossentropy())

    return ae



if __name__=='__main__':

    autoencoder_config = {
        'epochs' : 15,
        'sub_sample' : {
            'enable':True,
            'size':500000,
        }
    }

    data_config = {
    'filter_features_using_rfc' : True,
    'num_best_features' : 0,
    'feature_scaling' : 'Min/Max',
    'excluded_cols' : ['Source Port','Destination Port','Protocol'],
    'excluded_cols_from_scaling' : []
   }

    cicids_full_fs = dd.read_parquet('../../parquet/inter_dataset/cicids_2017_fs.parquet')


