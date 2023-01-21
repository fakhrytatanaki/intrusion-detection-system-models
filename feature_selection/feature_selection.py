import numpy as np
import sys
sys.path.append('..')
from ids_main.preprocessing import feature_selection
import dask.dataframe as dd
import json



with open('feature_selection_conf.json','r') as fp:
    feature_selection_conf = json.load(fp)


# Full and Clean CICIDS Datasets
cicids_2017_full = dd.read_parquet(feature_selection_conf['cicids_a_full']) 
cicids_2018_full = dd.read_parquet(feature_selection_conf['cicids_b_full']) 


sel = feature_selection(cicids_2017_full,feature_selection_conf['num_best_features'],feature_selection_conf['excluded_cols'])

cicids_2017_full[sel].compute().to_parquet(feature_selection_conf['cicids_a_subset'])
cicids_2018_full[sel].compute().to_parquet(feature_selection_conf['cicids_b_subset'])


