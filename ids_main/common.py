#--other---
import json
import sys
import importlib
from glob import glob
import os
#--------

#--sklearn---
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,confusion_matrix,confusion_matrix
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
from tensorflow.keras import layers, losses
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
import matplotlib.pyplot as plt
#--------
dask.config.set(scheduler='threads')

from .models.BinaryBayesClassifier import BinaryBayesClassifier
from .models.autoencoders.common import reconstruction_error_funcs


def _threshold_predictor(e : np.float32,thresh:np.float32):
    return 1 if e > thresh else 0

threshold_predictor  = np.vectorize(_threshold_predictor)


class CICIDSAutoencoderModel:
    def __init__(self,autoencoder,autoencoder_config,autoencoder_training_strategy):
        self.autoencoder = autoencoder
        self.autoencoder_config = autoencoder_config
        self.autoencoder_training_strategy = autoencoder_training_strategy
        self.threshold_value = 0
        self.reconstruction_errors = None

        rcf_name = self.autoencoder_config['reconstruction_error_func']
        self.rcf = reconstruction_error_funcs[rcf_name]

    def set_threshold(self,threshold_value):
        self.threshold_value = threshold_value

    def calculate_reconstruction_error(self,x):
        x_rec = self.autoencoder.predict(x,batch_size=655536)
        self.reconstruction_errors = self.rcf(x,x_rec) 


    def threshold_predictions(self,threshold_value=None):
        """
        Evaluates simple threshold prediction for the auto-encoder, if no threshold is given, then the last set (default) threshold in self.threshold_value is used instead
        """
        assert(type(self.reconstruction_errors)!=type(None))

        if not threshold_value:
            return threshold_predictor(self.reconstruction_errors,self.threshold_value)

        return threshold_predictor(self.reconstruction_errors,threshold_value)


    def fit(self,x):
        train_nn_model(
            model=self.autoencoder,
            model_config=self.autoencoder_config,
            data=x,
            training_strategy=self.autoencoder_training_strategy,
        )

    def predict(self,x):
        self.calculate_reconstruction_error(x)
        return self.threshold_predictions()




class CICIDSAutoencoderModelWithBinaryBayesClassifier(CICIDSAutoencoderModel):
    def __init__(self,autoencoder,autoencoder_config,autoencoder_training_strategy,binary_bayes_classifier):
        CICIDSAutoencoderModel.__init__(self,autoencoder,autoencoder_config,autoencoder_training_strategy)
        self.binary_bayes_classifier = binary_bayes_classifier


    def binary_bayes_classifier_predictions(self):
        assert(type(self.reconstruction_errors)!=type(None))
        return self.binary_bayes_classifier.predict(self.reconstruction_errors)

    def fit(self,x,y):
        x_benign = x[y==0]
        CICIDSAutoencoderModel.fit(self,x_benign)

        CICIDSAutoencoderModel.calculate_reconstruction_error(self,x)
        e = self.reconstruction_errors
        self.binary_bayes_classifier.fit(e,y)


    def predict(self,x):
        CICIDSAutoencoderModel.calculate_reconstruction_error(self,x)
        return self.binary_bayes_classifier_predictions()





def train_nn_model(model,model_config,data,training_strategy,data_transformation_strategy=lambda data:data,after_train_callback=None):

    data = data_transformation_strategy(data)
    training_data = data

    epochs = model_config['epochs']
    for i in range(epochs):

        if model_config['sub_sample']['enable']:
            print("sampling...")
            sub = choice(np.arange(len(data)), size=model_config['sub_sample']['size'], replace=False)
            training_data = data[sub]

        print(f"Epoch {i+1}/{epochs}")
        print("train...")
        training_strategy(training_data,model,model_config)
        
        if after_train_callback:
            ctx = {
                'epochs':i+1,
                'model':model,
                'model_config':model_config
                }
            after_train_callback(ctx)
        
    return model






    
