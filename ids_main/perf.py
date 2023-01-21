import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,confusion_matrix

def summarize_performance(y_true,y_pred):
    
    perf = {
        'Accuracy' : accuracy_score(y_true,y_pred),
        'Precision' : precision_score(y_true,y_pred,pos_label=1),
        'Recall' : recall_score(y_true,y_pred,pos_label=1),
        'F1' : f1_score(y_true,y_pred,pos_label=1),
    }
    
    return perf

def plot_summarized_results(results,x_metric,ax=None):
    fig = None
    if ax:
        fig,ax = plt.subplots(1,1)
        fig.set_size_inches(16,12)
        fig.set_dpi(150)
    
    sns.lineplot(
        label='Accuracy',
        ax=ax,
        data=results,
        x=x_metric,
        y='Accuracy',
    )

    sns.lineplot(
        label='F1',
        ax=ax,
        data=results,
        x=x_metric,
        y='F1',
    )

    sns.lineplot(

        label='Precision',
        ax=ax,
        data=results,
        x=x_metric,
        y='Precision',
    )

    sns.lineplot(
        label='Recall',
        ax=ax,
        data=results,
        x=x_metric,
        y='Recall',
    )


    ax.set_ylabel('Score')
    return fig,ax



def calc_auroc(confusion_matrices):

    ROC = []

    for cfm in confusion_matrices:
        tn, fp, fn, tp = cfm.ravel()
        true_positive_rate = tp/(tp+fn)
        false_positive_rate = fp/(tn+fp)
        ROC.append([false_positive_rate,true_positive_rate])

    ROC.sort(key=lambda tup:tup[0]) 

    AUROC=0 #area under ROC curve calculation
    x_prev = None
    for i,tpr_fpr in enumerate(ROC):

        x,y = tpr_fpr

        if i > 0:
            dx = x - x_prev
            AUROC+=y*dx
        x_prev = x
    return AUROC,np.array(ROC)
