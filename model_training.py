# -*- coding: utf-8 -*-
"""
Created on 09/09/2019

@author: Yannick Bottino
"""


import xgboost as xgb
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split
import preprocessing
import data
from time import time
import metrics





#------------------ XGB parameters -----------------------
params = {"objective": "reg:linear",
          "booster" : "gbtree",
          "eta": 0.03,
          "max_depth": 6,
          "subsample": 0.8,
          "colsample_bytree": 0.7,
          "silent": 1,
          "seed": 300,
          "lambda" : 0.3
          }

num_boost_round = 10
early_stopping_rounds = 100



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
    
    
    #Model training
    print("Training a XGBoost model")
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
    

    print("Performance on  training set")
    dtest = xgb.DMatrix(X_train[features])
    test_probs = gbm.predict(dtrain)
    error_test = metrics.rmspe(y_train.Ventes.values, test_probs)
    print('RMSPE: {:.6f}'.format(error_test))
    
    print("Performance on Validation set")
    yhat = gbm.predict(xgb.DMatrix(X_valid[features]))
    error = metrics.rmspe(y_valid.Ventes.values, yhat)
    print('RMSPE: {:.6f}'.format(error))

    print("Performance on test set")
    dtest = xgb.DMatrix(X_test[features])
    test_probs = gbm.predict(dtest)
    error_test = metrics.rmspe(y_test.Ventes.values, test_probs)
    print('RMSPE: {:.6f}'.format(error_test))
    
    xgb.plot_importance(model = gbm, max_num_features=50, height=0.8)

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
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
  
    fitting_time = time()-start_time

    print('Total fitting time = {:0.2f}s.'.format(fitting_time))
    
    #Save Trained Model
    dump(gbm, 'trained_XBG.joblib')
