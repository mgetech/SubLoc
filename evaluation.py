# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 14:27:44 2020

@author: MGE
"""
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score,precision_score, recall_score
import random
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from inspect import signature
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import column_or_1d
import functions as func
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
#print(keras.__version__)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv1D, Add, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, GlobalMaxPooling1D
from tensorflow.compat.v1.keras.layers import CuDNNLSTM

############# SVM ALGORITHM ##############

####### SVM hyperparameter tuning
def SVM_tuning(X, y):
    
    s = datetime.now()
    from sklearn.multiclass import OneVsRestClassifier
    clf = svm.SVC(verbose = 2)
    
    # Choose some parameter combinations to try
    parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                         'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                        {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                         'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                        {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
                       ]
    
    acc_scorer = make_scorer(f1_score, average= 'macro')
#    acc_scorer = make_scorer(average_precision_score)
    
    from sklearn import preprocessing
#    y = preprocessing.label_binarize(y, classes=[1, 2, 3,4,5,6,7,8,9])

    #grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_obj = GridSearchCV(clf, parameters, cv=3, scoring=acc_scorer, n_jobs=4)
    grid_obj = grid_obj.fit(X, y)
    
    # Set the clf to the best combination of parameters
    clf = grid_obj.best_estimator_
    print(grid_obj.best_params_)
    
    func.execution_time(s)
    # finish Beep sound
    func.beeep()
    
    return grid_obj.best_params_
    

def crossValidation(X, y, n_folds):
    dtS = list()
    dtX = list(X)
    dty = list(y)
    fSize = int(len(X) / n_folds)
    #random.seed(4)
    for i in range(n_folds):
        fold_X = list()
        fold_y = list()
        
        while len(fold_X) < fSize:
            
            i = random.randrange(len(dtX))
            fold_X.append(dtX.pop(i))
            fold_y.append(dty.pop(i))
            
        dtS.append([fold_X,fold_y])
        
    return dtS


def prec_recall(y_real, y_proba, title):
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    
    average_precision = average_precision_score(y_real, y_proba)
    
#    print('Average precision-recall score: {0:0.2f}'.format(average_precision))

    
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.figure(figsize = (10,7))    
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title+'Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision))
    
    
    
    
################ 10-fold cross-validation RandomForest ######

def svm_10fold(X, y, params):
    cmSum = [[0,0],
             [0,0]] 
    acc = 0
    bAcc = 0
    fc = 0
    fc2 = 0
    y_real = []
    y_proba = []
    perc = 0
    rec = 0

    folds = crossValidation(np.asarray(X), np.asarray(y), 10)
    c = 0
    for fold in folds:
        X_train = list()
        y_train = list()
        for f_t in folds:
            X_train.append(f_t[0])
            y_train.append(f_t[1])
    
        X_train.pop(c)
        X_train = sum(X_train, [])
        y_train.pop(c)
        y_train = sum(y_train, [])

        X_test = list()
        y_test = list()    
        for row in fold[0]:
            X_test.append(row)
    
        for row in fold[1]:
            y_test.append(row)
            
        
        c=c+1
    
        
        
        clf = svm.SVC(**params)

        clf.fit(X_train,y_train)
    
        # prediction on test set
        y_pred = clf.predict(X_test)
        
        y_real.append(y_test)
        y_proba.append(y_pred)

        
#        cm = confusion_matrix(y_test, y_pred)
#        cmSum = [[cmSum[i][j] + cm[i][j]  for j in range
#        (len(cm[0]))] for i in range(len(cm))] 
        
        acc = acc + (accuracy_score(y_test, y_pred)*100) 
        bAcc = bAcc + (accuracy_score(y_train, clf.predict(X_train))*100) 


        fc = fc + f1_score(y_test, y_pred, average="macro")
        fc2 = fc2 + f1_score(y_test, y_pred, average="micro")
        
        perc = perc + precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = rec + recall_score(y_test, y_pred, average="macro")
        
        
    
        
    precision = perc/10
    recall = rec/10
            
    testingAccuracy = acc/10
    trainingAccuracy = bAcc/10
    Fscore = fc/10
    Fscore2 = fc2/10

    return cmSum, trainingAccuracy, testingAccuracy, Fscore, y_real, y_proba, Fscore2, precision,recall 



def svm_eval(X, y, params, runtype):
    
    s = datetime.now()

    svm_params = params
    
    if (runtype == "10_fold_regular"):
        print("Regular 10-fold cross-validation SVM")  
        svm = svm_10fold(X, y,svm_params)        
        
        print(f"Testing Accuracy: {round(svm[2],3)}%")
        print(f"F-score(macro): {round(svm[3],3)}")
        print(f"F-score(micro): {round(svm[6],3)}")
        print(f"Precision: {round(svm[7],3)}")
        print(f"Recall: {round(svm[8],3)}")
    
    
    elif (runtype == "10_fold_grand_mean"):
 
        print("Grand mean 10-fold cross-validation SVM")  
    #   grand mean
        
        acc = 0
        fc1 = 0
        fc2 = 0
        perc = 0
        rec = 0
        for i in range(0,30):
            print(f"Iteration {i+1}/30")
            svm = svm_10fold(X, y,svm_params)
            acc = acc + svm[2]
            fc1 = fc1 + svm[3]
            fc2 = fc2 + svm[6]
            perc = perc + svm[7]
            rec = rec + svm[8]
        
        print("Grand mean of 10-fold cross-validation SVM (30 runs)")         
        print(f"Grand mean Accuracy: {round(acc/30,3)}%")
        print(f"Grand mean F-score(macro): {round(fc1/30,3)}")
        print(f"Grand mean F-score(micro): {round(fc2/30,3)}")
        print(f"Grand mean Precision: {round(perc/30,3)}")
        print(f"Grand mean Recall: {round(rec/30,3)}")


    func.execution_time(s)
    



def neuralNet_10fold(X, y, ep, batch, netType, es):
    cmSum = [[0,0,0], 
            [0,0,0], 
            [0,0,0]] 
    acc = 0
    fc = 0
    fc2 = 0
    y_real = []
    y_proba = []
    perc = 0
    rec = 0

    folds = crossValidation(np.asarray(X), np.asarray(y), 10)
    c = 0
    for fold in folds:
        X_train = list()
        y_train = list()
        for f_t in folds:
            X_train.append(f_t[0])
            y_train.append(f_t[1])
    
        X_train.pop(c)
        X_train = sum(X_train, [])
        X_train = np.asarray(X_train)
        y_train.pop(c)
        y_train = sum(y_train, [])
        y_train = np.asarray(y_train)

        X_test = list()
        y_test = list()    
        for row in fold[0]:
            X_test.append(row)
    
        X_test = np.asarray(X_test)
    
        for row in fold[1]:
            y_test.append(row)
        y_test = np.asarray(y_test)
        
        c=c+1
        

        if (netType == 'lstm'):            
            clf = lstm_model()
        elif (netType == 'PortCNN'):            
            clf = PortCNN_model()
        
        clf.fit(X_train, y_train,epochs=ep, batch_size=batch,
            validation_data=(X_test, y_test),
            callbacks=[es])
        
        display_model_score(clf,
                [X_train, y_train],
                
                [X_test, y_test],
                256)

#        clf.fit(X_train, y_train, epochs=ep, batch_size=batch)
        #test_loss, test_acc = clf.evaluate(X_test, y_test)

#        predictions = clf.predict(X_test)
        y_pred = np.argmax(clf.predict(X_test),axis=1)
#        y_test = y_test.values.tolist()
        y_test2 = [np.argmax(y1, axis=None, out=None) for y1 in y_test]  
        
        
#        y_pred = clf.predict_classes(X_test)
#        y_test = y_test.values.tolist()
#        y_test = [np.argmax(y1, axis=None, out=None) for y1 in y_test]
        
        
        y_real.append([x+1 for x in y_test2])
        y_proba.append([x+1 for x in y_pred.tolist()])
        

        
#        cm = confusion_matrix(y_test2, y_pred)
#        cmSum = [[cmSum[i][j] + cm[i][j]  for j in range
#        (len(cm[0]))] for i in range(len(cm))] 
        
        acc = acc + (accuracy_score(y_test2, y_pred)*100) 
        
#        y_train = y_train.values.tolist()
#        y_train = [np.argmax(y1, axis=None, out=None) for y1 in y_train]
#        bAcc = bAcc + (accuracy_score(y_train, clf.predict_classes(X_train))*100) 
        
        fc = fc + f1_score(y_test2, y_pred, average="macro")
        fc2 = fc2 + f1_score(y_test2, y_pred, average="micro")
        
        perc = perc + precision_score(y_test2, y_pred, average="macro")
        rec = rec + recall_score(y_test2, y_pred, average="macro")
        
        
    
        
    precision = perc/10
    recall = rec/10
            
    testingAccuracy = acc/10
    Fscore = fc/10
    Fscore2 = fc2/10

    return testingAccuracy, Fscore, y_real, y_proba, Fscore2, precision,recall 
        
                




# Utility function: Display model score(Loss & Accuracy) across all sets.
def display_model_score(model, train, test, batch_size):

    train_score = model.evaluate(train[0], train[1], batch_size=batch_size, verbose=1)
    print('Train loss: ', train_score[0])
    print('Train accuracy: ', train_score[1])
    print('-'*70)
    
    test_score = model.evaluate(test[0], test[1], batch_size=batch_size, verbose=1)
    print('Test loss: ', test_score[0])
    print('Test accuracy: ', test_score[1])


from keras import backend as K
def precision(y_true, y_pred):
    """Precision metric.
    
    Only computes a batch-wise average of precision.
    
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


    
def lstm_model():    
    x_input = Input(shape=(100,))
    emb = Embedding(21, 128, input_length=100)(x_input)
    bi_rnn = Bidirectional(CuDNNLSTM(64, kernel_regularizer=l2(0.00001),
                                     recurrent_regularizer=l2(0.00001),
                                     bias_regularizer=l2(0.00001)))(emb)
    x1 = Dropout(0.3)(bi_rnn)    
    # softmax classifier
    x_output = Dense(9, activation='softmax')(x1)    
    model1 = Model(inputs=x_input, outputs=x_output)
#    model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[precision])
    return model1



def lstm_eval(X, y, runtype):
    s = datetime.now()
    
    es = EarlyStopping(monitor='val_loss', patience=3, verbose=1)  
    
    if (runtype == "10_fold_regular"):
        print("Regular 10-fold cross-validation Bi-LSTM")  

        # 10-fold cross-validation
        lstm = neuralNet_10fold(X, y,ep=50, batch=256, netType= 'lstm', es=es)
        
        print(f"Testing Accuracy: {round(lstm[0],3)}%")
        print(f"F-score(macro): {round(lstm[1],3)}")
        print(f"F-score(micro): {round(lstm[4],3)}")
        print(f"Precision: {round(lstm[5],3)}")
        print(f"Recall: {round(lstm[6],3)}")
    
    
    elif (runtype == "10_fold_grand_mean"):
        print("Grand mean 10-fold cross-validation Bi-LSTM")  
        # grand mean 10-fold cross-validation
        
        acc = 0
        fc1 = 0
        fc2 = 0
        perc = 0
        rec = 0
        for i in range(0,30):
            print(f"Iteration {i+1}/30")
            lstm = neuralNet_10fold(X, y,ep=50, batch=256, netType= 'lstm', es=es)
            acc = acc + lstm[0]
            fc1 = fc1 + lstm[1]
            fc2 = fc2 + lstm[4]
            perc = perc + lstm[5]
            rec = rec + lstm[6]
    
        print("Grand mean of 10-fold cross-validation BiLSTM (30 runs)")         
        print(f"Grand mean Accuracy: {round(acc/30,3)}%")
        print(f"Grand mean F-score(macro): {round(fc1/30,3)}")
        print(f"Grand mean F-score(micro): {round(fc2/30,3)}")
        print(f"Grand mean Precision: {round(perc/30,3)}")
        print(f"Grand mean Recall: {round(rec/30,3)}")

    
    func.execution_time(s)


