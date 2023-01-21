import json
import sys
sys.path.append('../..') #for importing parent modules
import os
import importlib
from glob import glob
import scipy
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses,regularizers
import optuna
from ids_main.cic_flow_meter_data_scripts import AutoencoderEvaluation,CICFlowMeterDataLoader,binarize_attack_labels
from ids_main.models.autoencoders.variational_autoencoder import VariationalAutoencoder
from ids_main.models.autoencoders.autoencoder import Autoencoder
from ids_main.models.autoencoders.common import reconstruction_error_funcs
from ids_main.models.BinaryBayesClassifier import BinaryBayesClassifier
from ids_main.common import train_nn_model,CICIDSAutoencoderModel,CICIDSAutoencoderModelWithBinaryBayesClassifier

HELP_SCREEN= """
Usage : python main.py epochs=[number of epochs] scaler=[scaler name] sub_sample=[number of samples]

if sub_sample=0, then sub-sampling for auto-encoders will be disabled during training

Available Scalers : Normalisation, Min/Max, Standardisation 

Example : python main.py epochs=100 scaler=Normalisation sub_sample=50000
"""




with open('./optuna_config.json') as fp:
    proj_config = json.load(fp)


loss_funcs = {
        "BinaryCrossentropy":losses.BinaryCrossentropy,
        "MeanSquaredError":losses.MeanSquaredError,
        "Cosine" : lambda :losses.CosineSimilarity(axis=1)
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
                encoder_sub.add(layers.Dense(num_neurons, activation='relu',activity_regularizer=regularizers.L2(reg_constants_per_layer[i])))


    for i,num_neurons in enumerate(reversed(num_neurons_per_layer)):
            if reg_constants_per_layer==None:
                decoder_sub.add(layers.Dense(num_neurons, activation='relu'))
            else:
                decoder_sub.add(layers.Dense(num_neurons, activation='relu',kernel_regularizer=regularizers.L2(reg_constants_per_layer[len(reg_constants_per_layer) - i - 1])))


    decoder_sub.add(layers.Dense(cfg['data_vec_size'], activation='relu'))

    ae = ae_class(encoder_sub,decoder_sub,cfg['data_vec_size'],cfg['latent_vec_size'])
    ae.compile(
            optimizer='adam',
             loss=loss_funcs['MeanSquaredError']()
            )

    return ae


def get_optimized_ae(cfg,ae_class):


    _st = proj_config['study']
    ctx_path = os.path.join(_st['dir'],_st['ctx'])
    data_config = cfg['data_config']

    if os.path.exists(ctx_path):
        with open(ctx_path,'r') as fp:
            ctx = json.load(fp)
    else:
        ctx = {
                'best' : 0,
                'trial' : 0
              }

    dataset_a = proj_config['dataset_a']
    dataset_b = proj_config['dataset_b']

    dataloader_a = CICFlowMeterDataLoader(dataset_a,cfg)
    dataloader_b = CICFlowMeterDataLoader(dataset_b,cfg) if dataset_b else None

    dataloaders = (dataloader_a,dataloader_b)

    _objective = lambda trial : objective(
            trial,
            cfg,
            ctx,
            dataloaders
            )


    def objective(trial : optuna.Trial,cfg:dict,ctx:dict,dataloaders : tuple):

        print('preparing dataset A')
        dataloader_a =  dataloaders[0]
        x_train, x_test,y_train,y_test = dataloader_a.train_test_split()

        data_vec_size = x_train.shape[1]

        num_neurons_per_layer = []
        reg_constants_per_layer = []
        num_neurons_last_layer = data_vec_size
        num_neurons_min = trial.suggest_int('num_neurons_min',2,data_vec_size)

        re_func_name = 'Euclidean Distance'
        cfg['latent_vec_size'] = num_neurons_min-1
        cfg['data_vec_size']=data_vec_size

        num_layers = trial.suggest_int('num_layers',3,60)

        for l in range(num_layers):
            reg_constants_per_layer.append(np.exp(-trial.suggest_float(f'negative_ln_reg_constant_layer_{l}',2,5)))
            num_neurons_per_layer.append(trial.suggest_int(f'num_neurons_layer_{l}',num_neurons_min,num_neurons_last_layer))
            num_neurons_last_layer = num_neurons_per_layer[-1]

            if num_neurons_last_layer==num_neurons_min:
                break

        print(f'trying hidden_neurons:{num_neurons_per_layer},latent_vec_size : {cfg["latent_vec_size"]}')

        cfg['reconstruction_error_func']=re_func_name

        ae = ae_constructor(num_neurons_per_layer,cfg,ae_class)
        bbc = BinaryBayesClassifier(dist_bins=2000)
        model = CICIDSAutoencoderModelWithBinaryBayesClassifier(ae,cfg,lambda data,model,_:model.fit(data,batch_size=128),bbc)

        model.fit(x_train,binarize_attack_labels(y_train))

        intra_dataset_eval = AutoencoderEvaluation(model,x_test,y_test,res_dir=f"study/intra_dataset/trial_{ctx['trial']}")


        print('[Intra-Dataset Test] Primary results ')
        plot_desc=f"CICIDS Reconstruction Error Distribution\n Decoder : {data_vec_size} -> {num_neurons_per_layer} -> {num_neurons_min-1}"
        intra_dataset_eval.calculate_ae_outputs(plot_desc=plot_desc) 
        intra_dataset_eval.evaluate_bayesian_inference()
        r = intra_dataset_eval.evaluate_auroc()



        dataloader_b =  dataloaders[1]
        if dataloader_b:
            print('[Inter-Dataset Test]')
            _, _,y_train,y_test = dataloader_a.train_test_split(0.5)
            inter_dataset_eval = AutoencoderEvaluation(cicids_ae_model,x_test,y_test,res_dir=f"study/inter_dataset/trial_{ctx['trial']}")
            intra_dataset_eval.evaluate_bayesian_inference()
            inter_dataset_eval.calculate_ae_outputs() 

        res = r

        if res > ctx['best']:

            ctx['best']=res
            ae.save(os.path.join(_st['dir'],_st['best_model']),'w')

            with open(os.path.join(_st['dir'],_st['best_model_cfg']),'w') as fp:
                json.dump(cfg, fp)

        with open(os.path.join(_st['dir'],_st['ctx']),'w') as fp:
            json.dump(ctx,fp)

        ctx['trial']+=1
        return res



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

    st.optimize(_objective,n_trials=50)

    





if __name__=='__main__':
    

    default_args = {
           'epochs' : 20,
           'sub_sample' : 0,
           'scaler' : 'Min/Max',
           }

    args = sys.argv

    if len(args) > 1  and args[1]=='help':
        print(HELP_SCREEN)
        exit(0)

    for a in args[1:]:
        key,value = a.split('=')
        value_type = type(default_args[key]) 
        default_args[key]=value_type(value)


    autoencoder_config = {

        'epochs' : default_args['epochs'],
        'sub_sample' : {
            'enable':default_args['sub_sample'] > 0,
            'size':default_args['sub_sample'],
        },

        'data_config' : {
            'feature_scaling' : default_args['scaler'],
           }
    }


    get_optimized_ae(autoencoder_config,VariationalAutoencoder)
