import sys
import os
import numpy as np
from numpy.linalg import det
from numpy.random import shuffle,choice
from .preprocessing import scalers
from .models.histogram import HistogramDist
from .perf import summarize_performance,calc_auroc
from .preprocessing import scalers
from sklearn.metrics import confusion_matrix,roc_curve,roc_auc_score
import matplotlib.pyplot as plt
import dask.dataframe as dd
import pandas as pd
import dask
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split



#--------global-constants--------
G_PLOT_MARGIN = 3
"""
adds extra margins for reconstruction error distribution plots to make sure all values are visibile on the x-axis
"""
#--------------------------------


pandas_x_y_split = lambda df:(df.drop(columns=['Label']).values,df[['Label']].values)
is_attack = lambda df:~(df['Label'].str.lower()=='benign')
binarize_attack_labels = np.vectorize(lambda label:not label.lower()=='benign')

def plot_roc(ROC):
    fig,ax = plt.subplots(1,1)
    ax.plot(ROC[:,0],ROC[:,1])
    ax.plot([0,1],[0,1],linestyle='--')
    return fig,ax



def plot_cicids_attack_dists(rec_errs : np.ndarray ,attack_labels:np.ndarray,xbound=None,ybound=None) -> any:
    """
    Plots the reconstruction error distribution for every attack type
    """
    assert(attack_labels.shape[0]==rec_errs.shape[0])
    attacks = np.unique(attack_labels)
    num_attacks = len(attacks)

    dists = {} 

    for a in attacks:
        rec_err_segment = rec_errs[attack_labels==a]
        dists[a] = HistogramDist()
        dists[a].fit(rec_err_segment,num_bins=50)

    grid_length = int(np.ceil(np.sqrt(num_attacks)))
    fig,axes = plt.subplots(grid_length,grid_length)
    fig.set_size_inches(16,16)

    for i,ax in enumerate(axes.flatten()):

        if i < num_attacks:
            a = attacks[i]

            ax.set_xlabel(f'{a} ({dists[a].num_samples} samples)')
            ax.fill_between(dists[a].samples_range,dists[a].dist_prob)
            if xbound:
                ax.set_xbound(*xbound)
            if ybound:
                ax.set_ybound(*ybound)
        else:
            ax.remove()
        
    return fig,axes

def reconstruction_error_stats_per_attack(attack_labels:np.ndarray,rec_errs:np.ndarray):
    assert(attack_labels.shape[0]==rec_errs.shape[0])
    attacks = np.unique(attack_labels)
    stats_values = []

    for attack in attacks:
        rec_errs_for_an_attack = rec_errs[attack==attack_labels]
        stats_values.append({
            'Rec. Err. Mean':np.round(np.mean(rec_errs_for_an_attack),3),
            'Rec. Err. Std. Dev.': np.round(np.std(rec_errs_for_an_attack),3),
            'Attack Type':attack,
            'Samples':len(rec_errs_for_an_attack)
            })

    return pd.DataFrame(stats_values)#.set_index('Attack Type')



def perf_metrics_per_attack(attack_labels:np.ndarray,y_pred:np.ndarray):
    assert(attack_labels.shape[0]==y_pred.shape[0])


    attacks = np.unique(attack_labels)
    accuracy_values = []

    for attack in attacks:
        y_true = 0 if attack.lower()=='benign' else 1
        y_pred_seg = y_pred[attack_labels==attack]
        accuracy = f"{np.round(100*len(y_pred_seg[y_pred_seg==y_true])/len(y_pred_seg),2)}%"
        accuracy_values.append({'Accuracy':accuracy,'Attack Type':attack})

    return pd.DataFrame(accuracy_values)#.set_index('Attack Type')



def plot_cicids_dists_bin(dists,xbound=None,ybound=None):

    fig,ax = plt.subplots(1,len(dists.keys()))
    fig.set_size_inches(7*len(dists.keys()),10)
    fig.suptitle(f"""
    test
    """)
    for i,a in enumerate(dists.keys()):
        ax[i].set_xlabel(f'{a} : ({dists[a].num_samples} samples)')
        ax[i].fill_between(dists[a].samples_range,dists[a].dist_prob)

        if xbound:
            ax[i].set_xbound(*xbound)
        if ybound:
            ax[i].set_ybound(*ybound)
        
    return fig,ax


class AutoencoderEvaluation:
    """
    Class for evaluating autoencoders' performance using a validation dataset

    """
    def __init__(self,cicids_ae_model,x_test,y_test,res_dir,positive_label=1,negative_label=0,quantile_bounds=(0.01,0.99)):

        """
        params:-
            cicids_ae_model : Ready autoencoder-based model (class : CICIDSAutoencoderBasedModel)
            x_test : input data for testing,
            y_test : output data (labels) for testing,
            attack_labels : attack names,
            res_dir : base directory where evaluation results are stored,
            positive_label : Label value of positive classes (default : 1),
            negative_label : Label value of negative classes (default : 0),
            quantile_bounds : reconstruction error threshold values range in terms of quantiles (example: (0.1,0.9) specifies the minimum value as the biggest of the first 10% values and the max value as the biggest of the first 90% values)
        """

        print("[CICIDS Autoencoder Eval] preparing...")
        assert(len(x_test)==len(y_test))
        self.quantile_bounds = quantile_bounds
        self.x_test = x_test
        self.y_test = binarize_attack_labels(y_test)
        self.attack_labels = y_test
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.res_dir = res_dir
        self.model = cicids_ae_model

        if not os.path.exists(res_dir):
            os.makedirs(res_dir)



    def calculate_ae_outputs(self,res_name='dists',plot_desc="CICIDS Reconstruction Error Distributions"):
        """
        calculates the reconstruction error of the autoencoder
        """
        self.model.calculate_reconstruction_error(self.x_test)
        self.value_bounds = (np.quantile(self.model.reconstruction_errors,self.quantile_bounds[0]), np.quantile(self.model.reconstruction_errors,self.quantile_bounds[1]))
        self.plot_bounds = (self.value_bounds[0] - G_PLOT_MARGIN,self.value_bounds[1] + G_PLOT_MARGIN)
                            

        dists = {
                'Benign' : HistogramDist().fit(self.model.reconstruction_errors[self.y_test==0],num_bins=100),
                'Attack' : HistogramDist().fit(self.model.reconstruction_errors[self.y_test==1],num_bins=100)
                 }

        fig,_ = plot_cicids_dists_bin(dists,xbound=self.plot_bounds,ybound=(0,0.75))
        fig.suptitle(plot_desc)
        fig.savefig(os.path.join(self.res_dir,f'{res_name}.png'))
        plt.close()

        fig,_ = plot_cicids_attack_dists(self.model.reconstruction_errors,self.attack_labels,ybound=(0,0.75))
        fig.suptitle(plot_desc)
        fig.savefig(os.path.join(self.res_dir,f'{res_name}_attacks.png'))
        plt.close()



    def evaluate_bayesian_inference(self):
        """
        Evaluates the supervised bayesian binary classifier using the last reconstruction error values computed using ( self.calc_ae_outputs(self,ctx) ) 
        """
        y_pred = self.model.binary_bayes_classifier_predictions()
        res = summarize_performance(self.y_test, y_pred)
        pd.DataFrame([res]).to_csv(os.path.join(self.res_dir,'./bbc.csv'))
        self.calculate_and_save_cicids_attacks_info(y_pred,'cicids_bbc.csv')
        return res


    def calculate_and_save_cicids_attacks_info(self,y_pred,file_name):
        """
        Calculates accuracy per attack type and reconstruction error statistics per attack, saves the results as a csv file
        """
        stats = reconstruction_error_stats_per_attack(self.attack_labels,self.model.reconstruction_errors)
        acc_per_attack = perf_metrics_per_attack(self.attack_labels,y_pred)
        acc_per_attack.merge(stats).to_csv(os.path.join(self.res_dir,file_name),index=False)



    def evaluate_threshold_prediction(self,threshold_value=None):
        """
        calls self.model.threshold_predictions(self,threshold), then evaluates the performance
        params :-
        threshold_value : if None is specified (default) then the default threshold value of the model is used)
        returns :
        Tuple(
            y_pred : predictions of the model,
            cfm : Confusion Matrix,
            detection_metrics : metrics calculated using summarize_performance
            )
        """

        y_pred=self.model.threshold_predictions(threshold_value)
        cfm = confusion_matrix(self.y_test, y_pred,normalize='true')
        detection_metrics = summarize_performance(self.y_test,y_pred)
        return y_pred,cfm,detection_metrics



    def evaluate_auroc(self,thresh_divs=50,res_name='roc',plot_desc='CICIDS Simple Threshold Results',inter_dataset=False):
        """
        Evaluates performance using different thresholds and calculates the Receiver Operating Characteristic (ROC) Curve 
        sets the most optimal threshold (that gives best f1 score) as default threshold for the autoencoder model
        (True Positive Rate vs. False Positive Rate),
        returns : value of area under the ROC curve
        params :-
        thresh_divs : specifies the divisions (number of thresholds) between the smallest and the largest specified values
        res_name : file name of the saved results
        plot_desc : text descriptions embedded in the plotting image
        """

        if not inter_dataset:
            thresh_min,thresh_max = self.value_bounds
            thresh_vals = np.linspace(thresh_min,thresh_max,num = thresh_divs)
            cfms = [] #confusion matrices
            detection_metrics_values = [] 

            f1_best=0
            best_threshold=0 

            for i,thresh in enumerate(thresh_vals):
                y_pred,cfm,detection_metrics = self.evaluate_threshold_prediction(thresh)
                cfms.append(cfm)
                detection_metrics['Threshold']=thresh
                print(detection_metrics)
                detection_metrics_values.append(detection_metrics)
                self.model.set_threshold(thresh) #set the most optimal threshold for the model

                if detection_metrics['F1'] > f1_best:
                    f1_best = detection_metrics['F1'] 
                    best_threshold = thresh
                    self.calculate_and_save_cicids_attacks_info(y_pred,'cicids_autoencoder_thresh_f1_best.csv')


            self.model.set_threshold(best_threshold) #set the most optimal threshold for the model
            pd.DataFrame(detection_metrics_values).to_csv(os.path.join(self.res_dir,f'{res_name}.csv'),index=False)
        else:
            print(f"thresh : {self.model.threshold_value}")
            y_pred,cfm,detection_metrics = self.evaluate_threshold_prediction()
            pd.DataFrame([detection_metrics]).to_csv(os.path.join(self.res_dir,f'ae_inter.csv'),index=False)
            self.calculate_and_save_cicids_attacks_info(y_pred,'cicids_ae_inter.csv')

        fpr,tpr,_ = roc_curve(self.y_test,self.model.reconstruction_errors)

        roc =  np.concatenate((fpr.reshape(-1,1),tpr.reshape(-1,1)),axis=1)
        auroc = roc_auc_score(self.y_test,self.model.reconstruction_errors)
        fig,ax = plot_roc(roc)
        fig.suptitle(f"AUROC={np.round(auroc,3)}")
        fig.savefig(os.path.join(self.res_dir,f'{res_name}.png'))

        plt.close()
        return auroc


        
class CICFlowMeterDataLoader:
        """
        Loads and manages CICIDS data
        """
        def __init__(self,file_path:str,neural_network_cfg:dict,existing_scaler=None):
            """
            Loads CICIDS data
            params :-
            file_path : CICIDS data parquet file path, data must be numeric and have binary labels (benign as 0 and attack as 1)
            neural_network_cfg : dict specifying the configurations of the neural network and data
            """

            self.df = dd.read_parquet(file_path)
            self.cfg = neural_network_cfg
            self.scaler_name = self.cfg['data_config']['feature_scaling']
            self.data_size = self.df.compute().shape[0]
            self.scaler=existing_scaler

        def train_test_split(self,train_size=0.8,smote=False):
            x = self.df.drop(columns=['Label']).values.compute()
            y = self.df.compute().Label.values



            x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=train_size)

            if smote:
                print("performing SMOTE sampling technique on trainig data")
                smoter = SMOTE()
                x_train,y_train = smoter.fit_resample(x_train,y_train)

            if not self.scaler:
                self.scaler = scalers[self.scaler_name]()
                self.scaler.fit(x_train)

            x_train = self.scaler.transform(x_train)
            x_test = self.scaler.transform(x_test)


            return x_train,x_test,y_train,y_test 







        def autoencoder_train_test_split_sample_with_controlled_ratio(self,n:int,benign_to_attack_ratio : float =1,benign_train_proportion=0.8) -> tuple[np.ndarray,np.ndarray]:
            """
            sample data with replacement such that the class ratio of benign to attack is controlled (default is 50% benign and 50% attack), useful for 
            dealing with class imbalances

            params :-
            n : total sample size combined (benign + attack)
            benign_ratio :the ratio of benign-class data to attack-class data (default is 1)
            benign_train_proportion : how much of benign data is used for training (default is 0.8)


            returns :-
            Tuple(
                x_train : sampled benign input data for training,
                x_test : training data consisting of both benign and attack
                y_test : sampled label data corresponding to x_test
            )
            """


            n_benign = int ( n//(1+benign_to_attack_ratio) ) 
            n_benign_train_inds = np.zeros(n_benign,dtype=bool)
            n_benign_train_inds[:int(n_benign*(benign_train_proportion))]=True 
            n_benign_test_inds = ~n_benign_train_inds

            n_attack = n - n_benign

            d_benign = self.df[~is_attack(self.df)].compute().sample(n=n_benign,replace=True)

            d_benign_train = d_benign[n_benign_train_inds]
            d_benign_test = d_benign[n_benign_test_inds]

            d_attack = self.df[is_attack(self.df)].compute().sample(n=n_attack,replace=True)

            sample = pd.concat(
                    (d_benign_test,d_attack)
                    )

            x_train = d_benign_train.drop(columns=['Label']).values

            if not self.scaler:
                self.scaler = scalers[self.scaler_name]()
                self.scaler.fit(x_train)


            x_train = self.scaler.transform(x_train)
            x_test = self.scaler.transform(sample.drop(columns=['Label']).values)
            y_test = sample['Label'].values

            return x_train,x_test,y_test


        def autoencoder_train_test_split(self,benign_train_proportion=0.8) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
            """
            performs train/test split on CICIDS data for the autoencoder then scales the data with respect to the benign training data 
            params:-
            benign_train : proportion of benign data being used for training
            returns:-

            Tuple(
                x_train : benign training data,
                x_test : testing data that includes attack data and the remainder of benign data ,
                y_test : testing data labels for validation
            )
            """

            benign_total = self.df[~is_attack(self.df)].compute()
            benign_total_size=benign_total.shape[0]

            benign_train_size = int(np.floor(benign_train_proportion*benign_total_size))

            benign_train_inds=np.zeros(benign_total_size,dtype=bool)
            benign_train_inds[:benign_train_size]=True
            benign_train = benign_total[benign_train_inds].drop(columns=['Label'])

            benign_test = benign_total[~benign_train_inds]
            attack = self.df[is_attack(self.df)].compute()
            test = pd.concat((benign_test,attack))

            x_train = benign_train.values
            x_train = benign_train.values
            x_test = test.drop(columns=['Label']).values
            y_test = test['Label'].values


            if not self.scaler:
                self.scaler = scalers[self.scaler_name]()
                self.scaler.fit(x_train)

            x_train = self.scaler.transform(x_train)
            x_test = self.scaler.transform(x_test)

            return x_train,x_test,y_test



            



            
            











