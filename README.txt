--Intrusion Detection System project--
Implemented for the thesis titled 
"Improving the Data Generalisability of Network-based Intrusion Detection Systems"
by Fakhry Hussein Tatanaki

--For macOS and GNU/Linux users--
The dependencies can be installed using the following script "thesis_project_setup.sh"
by typing "sh ./thesis_project_setup.sh" in the terminal, make sure the terminal shell is 
running in the same directory as the project
---------------------------------

alternatively, the following can be used to install the dependencies
(important note : the project requires python verison 3)

1-setting up a virtual environment (optional but recommended) :

--For macOS and GNU/Linux users--
---------------------------------
python -m venv venv
source venv/bin/activate
---------------------------------


--------For Windows users--------
---------------------------------
python -m venv venv
venv\Scripts\Activate.ps1
---------------------------------

2- Installing dependencies using pip
---------------------------------
python -m pip -U pip "dask[complete]" pyarrow pandas numpy matplotlib optuna imblearn tensorflow
---------------------------------

Feature Selection
_______________________________________________________
Since the CICIDS Datasets are very large, the script had to be run separately.
The script is located at the directory : ./feature_selection 

when the script "feature_selection.py" is run, it performs feature selection on the datasets using Random Forest Classifier (RFC) and removes 
columns incompatible with the model and zero variance columns

The feature selection can be configured using a JSON file './feature_selection/feature_selection_conf.json' with
the following format

Note : The datasets should be in parquet format

{
  "num_best_features": (number of best features to be selected using RFC, if it is set to zero, then no RFC feature_selection is done),
  "excluded_cols": (other columns removed from the data),
  "cicids_a_full": (location of the first CICIDS dataset, which will be used for feature selection),
  "cicids_b_full": (location of the second CICIDS dataset, which will have the same columns as the first),
  "cicids_a_subset": (destination of the feature selection result of first dataset),
  "cicids_b_subset": (destination of the feature selection result of the second dataset),
}


Models
_______________________________________________________
./ids_main contains the core implementations of the project
./optuna_scripts contains the scripts that test the models using hyper-parameter tuning and calculates the performance results

for example : ./optuna_scripts/autoencoder_bbc
Tests the Auto-encoder model using thresholding and Bayes Classifier 

each model folder like autoencoder_bbc has a JSON File ./optuna_config.json
with the following format

{
  "dataset_a" : (Location of the parquet dataset processed using feature selection script),
  "dataset_b" : (Location of the second dataset to use for inter-dataset validation, can be left empty to skip inter-dataset validation),
  "study" : { 
    "dir" : (location of the directory where the optuna results are saved, including metrics and plots),
    "db" : (location of the optuna database that maintains the results for hyper-parameter tuning (inside "dir")),
    "best_model" : (location of the auto-encoder model with best results (inside "dir")),
    "best_model_cfg" : (location of the best JSON configuration file for the auto-encoder model with best results (inside "dir")),
    "ctx" : (location of json file for maintaining some info about the hyper-parameter tuning (inside "dir")),
  }
}

to run 
____________________________________________________
cd ./optuna_scripts/autoencoder_bbc/
python main.py
____________________________________________________
extra arguments can be supplied such as which scaler to use, run "python main.py help" for more info
