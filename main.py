"""
Created on Thu Feb 27 17:49:58 2020

@author: MGE

# Instructions:

Please follow the steps below, before running the script:


1.  Navigate to Runtime>Change runtime type and select GPU as Hardware accelerator 
2.   Download this as .zip 
1.   Choose `Files` from the side menu
2.   Upload the downloaded .zip file
3.   Run the following cells in order
"""


#!pip install biopython

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x
import tensorflow as tf
print(tf.__version__)
import os
import numpy as np
import pandas as pd
import evaluation as evaal
import preprocessing as prp
import functions as func
from datetime import datetime
from tensorflow import keras
#print(keras.__version__)
from keras.preprocessing.sequence import pad_sequences

"""### Preprocessing"""

#@title Please choose the algorithm that you want to use: { form-width: "250px", display-mode: "both" }
algorithm = "SVM" #@param ["SVM", "BiLSTM"]



# ========== Importing the data

X, y = prp.import_data(algorithm)

# =========================== Preprocessing ===============================

# ========== Ordinal encoding

amino_codes = ['0', 'A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y']
non_amino_letters = ['B', 'J', 'O', 'U', 'X', 'Z']

amino_mapping = prp.create_mapping(amino_codes)
 
X['mapped_seq'] = prp.integer_encoding(X['seq'], amino_mapping) 

# ========== Sequence padding

X_pad = pad_sequences(X['mapped_seq'], maxlen=3800, padding='post', truncating='post')

# ===================== Dimensionality reduction

# =========== RFE
#reduced_X = prp.dim_reduction_RFE(X_pad, y, 100)
# If you are using new dataset, run the above line and comment the lines in the below
# Otherwise, in order to save time the optimal features have been saved in a file and will be read from the file (features_100.csv)

feat_support = pd.read_csv('features_100.csv')

X = pd.DataFrame(data=X_pad)
reduced_X = pd.DataFrame()
c = 0
for index, r in feat_support.iterrows():
    if(r[0] == True):
        reduced_X.loc[:,c] = X.iloc[:, index]
        c+=1

if (algorithm == "BiLSTM"):
    # Add the generated data (in the same order of their labels as they were added in preprocessing.py) at the end of X
    reduced_X = prp.add_synthetics("dataset/cytoplasmiccytoplasmicmembrane_synthetic(cyt&cm).txt", reduced_X)
    reduced_X = prp.add_synthetics("dataset/periplasmiccytoplasmicmembrane_synthetic(per_cm&per).txt", reduced_X)
    reduced_X = prp.add_synthetics("dataset/outermembraneextracellular_synthetic(om_ext&ext).txt", reduced_X)

"""### Machine Learning models"""

#@title Please choose the run type: { form-width: "250px" }
run_type = "10_fold_regular" #@param ["10_fold_regular", "10_fold_grand_mean"]
#@markdown * 10_fold_grand_mean will run 10_fold_regular for 30 times.
# ========================= Machine learning models ======================

if (algorithm == "SVM"):
    # ========== 1. Support Vector Machine (SVM) ==========
    
    # ==========  Hyperparameter tuning    
    #svm_params = evaal.SVM_tuning(reduced_X, y)
    svm_params = {'C': 50, 'gamma': 0.0001, 'kernel': 'rbf'}
    
    # ==========  Model evaluation using 10-fold cross-validation
#    evaal.svm_eval(reduced_X, y, svm_params, run_type)
    evaal.svm_eval(reduced_X, y, svm_params, run_type)

elif (algorithm == "BiLSTM"):
    # ========== 2. Deep learning ==========
    
    # ========== One-hot encoding    
    X_ohe, y_ohe = prp.one_hot_encoding(reduced_X, y)    
    
    # ========== Bidirectional Long short-term memory networks (Bi-LSTMs)
    # ========== Model evaluation using 10-fold Cross-validation
#    evaal.lstm_eval(reduced_X, y_ohe, run_type)
    evaal.lstm_eval(reduced_X, y_ohe, run_type)

func.beeep()
