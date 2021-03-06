#!/usr/bin/env python
# coding: utf-8

# In[ ]: 


"""
Created on 09/09/2019

@author: Yannick Bottino
"""


import xgboost as xgb

import pandas as pd
import numpy as np

from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import preprocessing
import data
from time import time
import metrics
import parameters

import shap
import matplotlib.pyplot as plt





#------------------ XGB parameters -----------------------
params = parameters.xgb_params

num_boost_round = parameters.num_boost_round
early_stopping_rounds = parameters.early_stopping_rounds



# ------------------- Model Interpretation ----------------

def SHAP_Analysis(model, X, y, feature_names):
    """
    Function that takes 3 arguments for plotting SHAP Analysis of the model:
    - model : The model we want to explain
    - X : The data features used for training
    - y : The data target used for training
    - features_names : the names of the training features
    
    This functions plots the summary plot    
    """
    

    explainer = shap.TreeExplainer(model)
    print('Calculating shap values for SHAP Analysis. \nThis may take few minutes due to the high amount of records...')
    shap_values = explainer.shap_values(X = X, y = y)
    
    shap.summary_plot(shap_values , X , feature_names)
    plt.show()
    
       
    
    

#---------------- Training methods -----------------------

def train_validate_model(df):
    #Get data
    X, feature_names = preprocessing.training_set_preprocessing.preco_features(df)
    y, _ = preprocessing.training_set_preprocessing.preco_target(df)
    
    
    #split train, validation and testing data
    #cette partie pourra être remplacée par un CV de la librairie sklearn à posteriori
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.15, random_state=38)
    
    _,features = preprocessing.training_set_preprocessing.preco_features(df)



    #Reconstructing data Matrix for Model training
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    
    
    htuning = input('Do you want to perform hyperparameter tuning ? (Y/n)')
    if htuning == 'Y':
        zerzafgrz = 0
    else:
        #Model training
        print("Training a XGBoost model")
        gbm = xgb.train(parameters.xgb_params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=50)
        

    print("\nPerformance on  training set")
    dtest = xgb.DMatrix(X_train[features])
    test_probs = gbm.predict(dtrain)
    error_test = metrics.rmspe(y_train.Ventes.values, test_probs)
    R_squarred = r2_score(y_train.Ventes.values, test_probs)
    adj_2 = metrics.adjusted_r2(feature_names,y_train.Ventes.values, test_probs)
    print('RMSPE: {:.6f}'.format(error_test))
    print('R Squarred: {:.6f}'.format(R_squarred))
    print('R Squarred (adj): {:.6f}'.format(adj_2))
    
    print("\nPerformance on Validation set")
    yhat = gbm.predict(xgb.DMatrix(X_valid[features]))
    error = metrics.rmspe(y_valid.Ventes.values, yhat)
    R_squarred = r2_score(y_valid.Ventes.values, yhat)
    adj_2 = metrics.adjusted_r2(feature_names,y_valid.Ventes.values, yhat)
    print('RMSPE: {:.6f}'.format(error_test))
    print('R Squarred: {:.6f}'.format(R_squarred))
    print('R Squarred (adj): {:.6f}'.format(adj_2))
    
    

    print("\nPerformance on test set")
    dtest = xgb.DMatrix(X_test[features])
    test_probs = gbm.predict(dtest)
    error_test = metrics.rmspe(y_test.Ventes.values, test_probs)
    R_squarred = r2_score(y_test.Ventes.values, test_probs)
    adj_2 = metrics.adjusted_r2(feature_names,y_test.Ventes.values, test_probs)
    print('RMSPE: {:.6f}'.format(error_test))
    print('R Squarred: {:.6f}'.format(R_squarred))
    print('R Squarred (adj): {:.6f}'.format(adj_2))
    
    #------------------- Plot feature importances ----------------
    xgb.plot_importance(booster = gbm, show_values = False, importance_type = 'gain')
    plt.show()
    
    # ------------------- Perform SHAP Analysis on training data --------------------
    SHAP_Analysis(gbm, X_train, y_train, feature_names)

    return gbm, watchlist




    
def train_deploy_model(df):
    
    #Get data
    X, _ = preprocessing.training_set_preprocessing.preco_features(df)
    y, _ = preprocessing.training_set_preprocessing.preco_target(df)
    
    
    #split train, validation and testing data
    #cette partie pourra être remplacée par un CV de la librairie sklearn à posteriori
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.1, random_state=42)
    
    _,features = preprocessing.training_set_preprocessing.preco_features(df)



    #Reconstructing data Matrix for Model training
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    
  
    
    #Model training
    print("Training a XGBoost model")
    
    start_time = time()
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=50)
  
    fitting_time = time()-start_time

    print('Total fitting time = {:0.2f}s.'.format(fitting_time))
    
    #Save Trained Model
    dump(gbm, 'trained_XBG.joblib')

