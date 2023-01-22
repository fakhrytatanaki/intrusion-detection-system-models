import dask
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler

binarize_attack_labels = np.vectorize(lambda label:not label.lower()=='benign')


class OneHotEncoder:
    def __init__(self,data,negative_one_as_cold=False):
        if negative_one_as_cold:
            self.ohe_lut = 2*np.identity(data.max()+1) - np.ones(data.max()+1)
        else:
            self.ohe_lut = np.identity(data.max()+1)

    def transform(self,data):
        return np.apply_along_axis(lambda x:self.ohe_lut[x],0, data)

    def inverse_transform(self,data):
        return np.apply_along_axis(lambda x:np.argmax(x),1, data)




scalers : dict[str,object]= {
    'Normalisation' : Normalizer,
    'Standardisation' : StandardScaler,
    'MinMax' : MinMaxScaler
}



@dask.delayed
def find_zero_var_cols(x):
    x_stddev = x.std()
    return x_stddev[np.isclose(x_stddev,0)]

def get_n_best_features_rfc(x,y,n_best,rfc_n_est=10):
    rfc = RandomForestClassifier(n_estimators=rfc_n_est)
    rfc = rfc.fit(x,y)
    features_by_importance = np.sort(
            np.array(
                [tuple(i) for i in np.column_stack((rfc.feature_importances_,rfc.feature_names_in_))],
                dtype=[('k',np.float64),('v',rfc.feature_names_in_.dtype)]
                )
            )

    best_features = features_by_importance[-n_best:]
    print("best_features : ",best_features)

    return [t[1] for t in best_features]


def feature_selection(data,n_best=0,excluded_cols=[]):
    labels = data.Label.compute()
    data_features_in = data.drop(columns=['Label']) 
    selected_cols = data_features_in.columns[data_features_in.dtypes.values=='float64'] #select only columns with numeric attributes
    zero_variance_cols = find_zero_var_cols(data_features_in[selected_cols]).compute()
    print(f'zero variance : {zero_variance_cols}')
    selected_cols = [c for c in selected_cols if not (c in zero_variance_cols) and not (c in excluded_cols) ]  #exclude columns with zero variance
    print(f'selected columns : {selected_cols}')
    print(f'index : {data.index}')
    
    if n_best > 0:
        data_features_in = data_features_in[selected_cols]
        selected_cols = get_n_best_features_rfc(data_features_in,labels,n_best,rfc_n_est=40) #feature selection using RFC

    selected_cols.append('Label')
    return selected_cols
