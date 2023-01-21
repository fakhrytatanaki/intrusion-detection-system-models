from .histogram import HistogramDist
import numpy as np

def _bayesian_infer(x,prob_true,dist_true,dist_false,outlier_label):
    prob_false = 1 - prob_true
    evidence = dist_true.probability_where_value_belongs(x)*prob_true + dist_false.probability_where_value_belongs(x)*prob_false 

    if np.isclose(evidence,0): #value is probably an outlier (outside the histogram bounds)
        return outlier_label

    posterior_prob_true = (dist_true.probability_where_value_belongs(x)*prob_true)/(evidence)
    posterior_prob_false = (dist_false.probability_where_value_belongs(x)*prob_false)/(evidence)
    return (1 if posterior_prob_true > posterior_prob_false else 0)


_bayesian_infer_vec = np.vectorize(_bayesian_infer,excluded=['prob_true','dist_true','dist_false','outlier_label'])

class BinaryBayesClassifier:
    def __init__(self,dist_bins=200,quantile_bounds=(0,1),outlier_label=1):
        self.quantile_bounds = quantile_bounds
        self.outlier_label = outlier_label
        self.dist_bins = dist_bins
        pass

    def fit(self,x_train: np.ndarray,y_train:np.ndarray):
        x_true = x_train[y_train==1]
        x_false = x_train[y_train==0]

        self.dist_true = HistogramDist(quantile_bounds=self.quantile_bounds)
        self.dist_false = HistogramDist(quantile_bounds=self.quantile_bounds)

        self.dist_true.fit(x_true,num_bins=self.dist_bins)
        self.dist_false.fit(x_false,num_bins=self.dist_bins)

        self.prob_true = len(x_true)/len(x_train)
        self.prob_false = 1 - self.prob_true
        return self


    def predict(self,x):
        return _bayesian_infer_vec(x,self.prob_true,self.dist_true,self.dist_false,self.outlier_label)






