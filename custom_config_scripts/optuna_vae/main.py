#--other---
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



predictors = {
    'Simple Thresholding' : np.vectorize(lambda e,thresh:1 if e > thresh else 0)
}

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
    ae.compile(optimizer=Adam(learning_rate=cfg['learning_rate']))

    return ae


def get_optimized_ae(cfg,cicids_data,data_config,ae_class):

    cicids_benign = cicids_data[cicids_data.Label==0].drop('Label',axis=1).compute()
    cicids_all = cicids_data.drop('Label',axis=1).compute()

    y_true = cicids_data.Label.compute().to_numpy()

    print("[autoencoder opt] scaling data..")
    scaler = pproc.scalers[data_config['feature_scaling']]()
    scaler = scaler.fit(cicids_benign)
    cicids_benign_sc = scaler.transform(cicids_benign)
    cicids_all_sc = scaler.transform(cicids_all)

    _st = proj_config['study']
    ctx_path = os.path.join(_st['dir'],_st['ctx'])

    if os.path.exists(ctx_path):
        with open(ctx_path,'r') as fp:
            ctx = json.load(fp)
    else:
        ctx = {
                'best' : 0,
                'trial' : 1
              }


    def objective(trial : optuna.Trial,cfg,ctx,data,y_true) -> np.float64:

        data_vec_size = data.shape[1]
        num_neurons_per_layer = []
        reg_constants_per_layer = []
        num_neurons_last_layer = data_vec_size
        num_neurons_min = 4

        cfg['latent_vec_size'] = num_neurons_min
        cfg['data_vec_size']=data_vec_size
        cfg['epochs'] = trial.suggest_int('epochs', 10,100)

        num_layers = trial.suggest_int('num_layers',3,20)

        for l in range(num_layers):
            reg_constants_per_layer.append(np.exp(-trial.suggest_float(f'negative_ln_reg_constant_layer_{l}',2,5)))
            num_neurons_per_layer.append(trial.suggest_int(f'num_neurons_layer_{l}',num_neurons_min,max(num_neurons_min,num_neurons_last_layer)))
            num_neurons_last_layer = num_neurons_per_layer[-1]


        natural_log_learning_rate = -trial.suggest_float('negative_ln_learning_rate',4,9)

        cfg['learning_rate'] = np.exp(natural_log_learning_rate)
        re_func_name = trial.suggest_categorical("reconstruction_error_func",reconstruction_error_funcs.keys() )

        cfg['reconstruction_error_func']=re_func_name
        

        predictor = predictors['Simple Thresholding']


        ae = ae_constructor(num_neurons_per_layer,cfg,ae_class,reg_constants_per_layer)
        x = data

        plot_path = f"./plots/dists/trial_{ctx['trial']}.png"
        ev = AutoencoderEvaluation(x,y_true)
        ae = train_nn_model(
            model=ae,
            model_config=cfg,
            data=x,
            training_strategy=lambda data,model,_:model.fit(data,batch_size=65536),
            after_train_callback=lambda model_ctx:ev.evaluate_bayesian_inference(model_ctx,plot_path) if model_ctx['epochs']==cfg['epochs'] else None
            )


        res = ev.last_result['F1']

        ctx['trial']+=1
        if res > ctx['best']:
            ctx['best']=res
            ae.save(os.path.join(_st['dir'],_st['best_model']),'w')

            with open(os.path.join(_st['dir'],_st['best_model_cfg']),'w') as fp:
                json.dump(cfg, fp)

        with open(os.path.join(_st['dir'],_st['ctx']),'w') as fp:
            json.dump(ctx,fp)

        return res


    _objective = lambda trial : objective(
            trial,
            cfg,
            ctx,
            cicids_all_sc,
            y_true
            )

    _st = proj_config['study']

    _storage_uri=os.path.join(
            'sqlite:///',
            _st['dir'],
            _st['db'],
            )


    if os.path.exists(os.path.join(_st['dir'],_st['db'])):
        st = optuna.load_study(
                storage=_storage_uri,
                pruner=optuna.pruners.SuccessiveHalvingPruner(),
                sampler=optuna.samplers.NSGAIISampler(),
                study_name='ae_model'
             )
    else:
        print("creating new study...")
        st = optuna.create_study(
                direction='maximize',
                storage=_storage_uri,
                pruner=optuna.pruners.SuccessiveHalvingPruner(),
                sampler=optuna.samplers.NSGAIISampler(),
                study_name='ae_model'
             )

    st.optimize(_objective,n_trials=500)

    




if __name__=='__main__':

    autoencoder_config = {
        'epochs' : 15,
        'sub_sample' : {
            'enable':True,
            'size':900000,
        }
    }

    data_config = {
    'filter_features_using_rfc' : True,
    'num_best_features' : 0,
    'feature_scaling' : 'Standardisation',
    'excluded_cols' : ['Source Port','Destination Port','Protocol'],
    'excluded_cols_from_scaling' : []
   }

    cicids_full_fs = dd.read_parquet('../../parquet/inter_dataset/cicids_2017_fs.parquet')
    get_optimized_ae(autoencoder_config, cicids_full_fs,data_config,VariationalAutoencoder)


