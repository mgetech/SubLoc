# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:47:59 2020

@author: MGE
"""

import os
import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
#print(keras.__version__)
from keras.preprocessing.sequence import pad_sequences


def import_FastaSeq(addrFasta):
    
    X = [seq.seq for seq in SeqIO.parse(addrFasta, 'fasta')]
    
    X = pd.DataFrame(data=X,   columns = ['seq'])
    
    amino_codes = ['0', 'A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y']
    non_amino_letters = ['B', 'J', 'O', 'U', 'X', 'Z']
    
    amino_mapping = create_mapping(amino_codes)
    
     
    X['mapped_seq'] = integer_encoding(X['seq'], amino_mapping) 
    
    # ========== Sequence padding
    
    X_pad = pad_sequences(X['mapped_seq'], maxlen=3800, padding='post', truncating='post')
    
    feat_support = pd.read_csv('save/features_100.csv')
    
    X = pd.DataFrame(data=X_pad)
    reduced_X = pd.DataFrame()
    c = 0
    for index, r in feat_support.iterrows():
        if(r[0] == True):
            reduced_X.loc[:,c] = X.iloc[:, index]
            c+=1    
    
    return reduced_X

def one_hot_encoding(X, y):
    from tensorflow.keras.utils import to_categorical
    # One hot encoding of sequences
    X_ohe = to_categorical(X)    
    from sklearn.preprocessing import LabelEncoder
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    # convert integers to dummy variables (i.e. one hot encoded)
    y_ohe = to_categorical(encoded_Y)
    return X_ohe, y_ohe




def GAN_input(addr, X):
    file = open(addr,"w")
    for index, r in X.iterrows():
        for cell in r:
            file.write(str(cell)+" ")       
        file.write("\n")
    
    file.flush()
    file.close()





def create_mapping(codes):
    mapping = {}
    for index, val in enumerate(codes):
        mapping[val] = index
    
    return mapping

def integer_encoding(data, mapping):
  
    encode_list = []
    for row in data.values:
        row = str(row)
        row = row.replace('X', '0')
        row = row.replace('U', '0')
        row = row.replace('O', '0')
        row = row.replace('B', 'N')
        row = row.replace('Z', 'Q')
        row = row.replace('J', 'L')
        row = list(row)        
        row_encode = []
        for code in row:
            row_encode.append(mapping.get(code, 0))
        encode_list.append(np.array(row_encode))
  
    return encode_list





