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

import functions as func
from datetime import datetime


def import_data(algorithm):
    # cytoplasmic
    cyto = [seq.seq for seq in SeqIO.parse("dataset/cytoplasmic-3_0.txt", 'fasta')]
    
    # cytoplasmic/cytoplasmicmembrane
    cyt_cm = [seq.seq for seq in SeqIO.parse("dataset/cytoplasmiccytoplasmicmembrane-3_0.txt", 'fasta')]
    
    # cytoplasmicmembrane
    cm = [seq.seq for seq in SeqIO.parse("dataset/cytoplasmicmembrane-3_0.txt", 'fasta')]
    
    # periplasmiccytoplasmicmembrane
    per_cm = [seq.seq for seq in SeqIO.parse("dataset/periplasmiccytoplasmicmembrane-3_0.txt", 'fasta')]
    
    # periplasmic
    per = [seq.seq for seq in SeqIO.parse("dataset/periplasmic-3_0.txt", 'fasta')]
    
    # periplasmicoutermembrane
    per_om = [seq.seq for seq in SeqIO.parse("dataset/periplasmicoutermembrane-3_0.txt", 'fasta')]
    
    # outermembrane
    om = [seq.seq for seq in SeqIO.parse("dataset/outermembrane-3_0.txt", 'fasta')]
    
    # outermembraneextracellular
    om_ext = [seq.seq for seq in SeqIO.parse("dataset/outermembraneextracellular-3_0.txt", 'fasta')]
    
    # extracellular
    ext = [seq.seq for seq in SeqIO.parse("dataset/extracellular-3_0.txt", 'fasta')]
    
        
    cyto_labels = np.ones(len(cyto), dtype=int)
    cyt_cm_labels = np.full(len(cyt_cm), 2, dtype=int)
    cm_labels = np.full(len(cm), 3, dtype=int)
    per_cm_labels = np.full(len(per_cm), 4, dtype=int)
    per_labels = np.full(len(per), 5, dtype=int)
    per_om_labels = np.full(len(per_om), 6, dtype=int)
    om_labels = np.full(len(om), 7, dtype=int)
    om_ext_labels = np.full(len(om_ext), 8, dtype=int)
    ext_labels = np.full(len(ext), 9, dtype=int)
        
    if (algorithm == "SVM"):
        X = np.concatenate((cyto, cyt_cm, cm, per_cm, per, per_om, om, om_ext, ext), axis=0)
        y = np.concatenate((cyto_labels, cyt_cm_labels, cm_labels, per_cm_labels,
                            per_labels, per_om_labels, om_labels, om_ext_labels, ext_labels), axis=0)
    elif (algorithm == "BiLSTM"):
        cyto_cm_syn_labels = open("dataset/cytoplasmiccytoplasmicmembrane_synthetic(cyt&cm).txt","r").readlines() 
        cyto_cm_syn_labels = np.full(len(cyto_cm_syn_labels), 2, dtype=int)
    
        per_cm_syn_labels = open("dataset/periplasmiccytoplasmicmembrane_synthetic(per_cm&per).txt","r").readlines() 
        per_cm_syn_labels = np.full(len(per_cm_syn_labels), 4, dtype=int)
    
    #    per_om_syn_labels = open("dataset/periplasmicoutermembrane_synthetic(om&per_om).txt","r").readlines() 
    #    per_om_syn_labels = np.full(len(per_om_syn_labels), 6, dtype=int)
    
        om_ext_syn_labels = open("dataset/outermembraneextracellular_synthetic(om_ext&ext).txt","r").readlines() 
        om_ext_syn_labels = np.full(len(om_ext_syn_labels), 8, dtype=int)
        
        
        X = np.concatenate((cyto, cyt_cm, cm, per_cm, per, per_om, om, om_ext, ext), axis=0)
        y = np.concatenate((cyto_labels, cyt_cm_labels, cm_labels, per_cm_labels,
                            per_labels, per_om_labels, om_labels, om_ext_labels, ext_labels,
                            cyto_cm_syn_labels, per_cm_syn_labels, om_ext_syn_labels), axis=0)
        
    X_df = pd.DataFrame(data=X,   columns = ['seq'])
    
    return X_df, y

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

def feat_scaling(X):
    # Feature Scaling
    sc_X = StandardScaler()
    X_scaled = sc_X.fit_transform(X)
    return X_scaled




def add_synthetics(addr, X):
    with open(addr) as f:
        mm = [list(x.split(" ")) for x in f]
    ft = [line[0:100] for line in mm[:]]
    syn = []
    for row in ft:
        syn.append([int(float(i)) for i in row] )
    
    X = X.append(pd.DataFrame(syn), ignore_index=True)
    return X

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


def dim_reduction_mRMR(df, k):
    import pymrmr
    s = datetime.now()
    reduced = pymrmr.mRMR(df, 'MIQ', k)
    func.execution_time(s)
    # finish Beep sound
    func.beeep()
    # below is how we can prepare the data to use mRMR    
    #ordinal_df = pd.DataFrame(pd.np.column_stack([y,X_pad]))
    #ordinal_df.rename(columns={0:'class'}, inplace=True)
    #for i in range(1,1001):
    #    ordinal_df.rename(columns={i:'Feat'+str(i)}, inplace=True)
    #reduced_ordf = prp.dim_reduction_mRMR(ordinal_df)    
    return reduced

def dim_reduction_PCA(X, k):
    from sklearn.decomposition import PCA    
    pca = PCA(n_components=k)
    reduced_X = pca.fit_transform(X)
    print(pca.explained_variance_ratio_)
    return reduced_X

def dim_reduction_SKBest(X, y, k):
    from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
    selector = SelectKBest(score_func=mutual_info_classif,k=k)
    KBest2_f_classif = selector.fit(X, y)
    best=X.columns[selector.get_support(indices=True)].tolist()
    return best

def dim_reduction_RFE(X, y, k):

    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import RFE    
    s = datetime.now()    
    model = ExtraTreesClassifier()
    rfe = RFE(model, k)
    fit = rfe.fit(X, y)
    print(f"Num Features: {fit.n_features_}")
    print(f"Selected Features: {fit.support_}")
    print(f"Feature Ranking: {fit.ranking_}")    
    feat_support = pd.DataFrame(data=fit.support_)
    feat_support['ranking'] = fit.ranking_    
    # saving the extracted features
    out_filename = os.path.join('features_100.csv')
    feat_support.to_csv(out_filename, index=False)  
    
   
    X = pd.DataFrame(data=X)
    reduced_X = pd.DataFrame()
    c = 0
    for index, r in feat_support.iterrows():
        if(r[0] == True):
            reduced_X.loc[:,c] = X.iloc[:, index]
            c+=1

    return reduced_X            
    



def dim_reduction_ForwardSelection(X, y, k):
    from sklearn import svm    
    from mlxtend.feature_selection import SequentialFeatureSelector as SFS
    from sklearn.metrics import make_scorer
    from sklearn.metrics import f1_score   
    s = datetime.now()    
    acc_scorer = make_scorer(f1_score, average= 'macro')    
    sfs = SFS(svm.SVC(), k_features=k, forward=True, floating=False, verbose=2, scoring=acc_scorer, cv=0)    
    # Perform SFS
    sfs = sfs.fit(X, y)    
    sfs.k_feature_names_    
    feat_cols = list(sfs.k_feature_idx_)
    print(feat_cols)    
    func.execution_time(s)
    # finish Beep sound
    func.beeep()
    return feat_cols            







