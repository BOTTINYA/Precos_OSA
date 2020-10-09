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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score

import preprocessing
import data
from time import time
import metrics
import parameters
import functions

import shap
import matplotlib.pyplot as plt





#------------------ XGB parameters -----------------------
params = parameters.xgb_params


# ------------------- Model Interpretation ----------------

def SHAP_Analysis(model, X, y, feature_names):
    """
    Function that takes 4 arguments for plotting SHAP Analysis of the model:
    - model : The model we want to explain
    - X : The data features used for training
    - y : The data target used for training
    - features_names : the names of the training features
    
    This functions plots the summary plot    
    """
    

    explainer = shap.TreeExplainer(model)
    print('Calculating shap values for SHAP Analysis. \nThis may take a moment due to the high amount of records...')
    shap_values = explainer.shap_values(X = X, y = y)
    
    shap.summary_plot(shap_values , X , feature_names)
    plt.show()
    
       
    
    

#---------------- Training methods -----------------------

def train_validate_model(df,enseigne):
    
    """
    Cette fonction est une fonctions qui sert à faire la validation du modèle. 
    Elle propose de faire un hypermarameter tuning automatique, entraine l'algo et trace les resultats du comportement avec une analyse SHAP.
    """
    
    
    #Get data
    X, feature_names = preprocessing.training_set_preprocessing.preco_features(df)
    y, _ = preprocessing.training_set_preprocessing.preco_target(df)
    
    
    #split train, validation and testing data
    #cette partie pourra être remplacée par un CV de la librairie sklearn à posteriori
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.33, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.15, random_state=38)
    
    _,features = preprocessing.training_set_preprocessing.preco_features(df)



    #Reconstructing data Matrix for Model training
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    
    htuning = 'n'
    #htuning = input('Voulez-vous faire un tuning automatique des Hyperparamètres du modèle ? (Y/n)' )
    
    if htuning.upper() == 'Y':
        #Model tuning
        print("Performing Grid Search on the model")
        gbm = xgb.XGBRegressor()

        reg = GridSearchCV(gbm, param_grid = parameters.xgb_grid, cv = 3, verbose = 1, n_jobs = -1)
        reg.fit(X_train,y_train)
        
        print('\nBest found Parameters for model are :')
        print(reg.best_params_)
        
        #Model training
        gbm = xgb.train(reg.best_params_, dtrain, num_boost_round = parameters.num_boost_round, evals=watchlist, early_stopping_rounds=parameters.early_stopping_rounds, verbose_eval=50)
        functions.save_obj(reg.best_params_, 'xgb_params_'+enseigne )
        print('Hyperparameters saved in local directory as "xgb_params_<enseigne>.pkl". They are loaded in parameters.py for ulterior use')
        print('To save a trained model with the found parameters, re-run main.py, retrain a model for deployment.')
        
    else:
        #Model training
        print("Training a XGBoost model")
        gbm = xgb.train(parameters.xgb_params, dtrain, num_boost_round = parameters.num_boost_round, evals=watchlist, early_stopping_rounds=parameters.early_stopping_rounds, verbose_eval=50)
    
    
    

    print("\nPerformance on  training set")
    dtest = xgb.DMatrix(X_train[features])
    test_probs = np.exp(gbm.predict(dtrain)) - 1
    error_test = metrics.rmspe(np.exp(y_train.VentesUC_log_transformed.values)-1, test_probs)
    R_squarred = r2_score(np.exp(y_train.VentesUC_log_transformed.values)-1, test_probs)
    adj_2 = metrics.adjusted_r2(feature_names,np.exp(y_train.VentesUC_log_transformed.values)-1, test_probs)
    print('RMSPE: {:.6f}'.format(error_test))
    print('R Squarred: {:.6f}'.format(R_squarred))
    print('R Squarred (adj): {:.6f}'.format(adj_2))
    
    
    
    print("\nPerformance on Validation set")
    yhat = np.exp(gbm.predict(xgb.DMatrix(X_valid[features])))-1
    error = metrics.rmspe(np.exp(y_valid.VentesUC_log_transformed.values) - 1, yhat)
    R_squarred = r2_score(np.exp(y_valid.VentesUC_log_transformed.values) - 1, yhat)
    adj_2 = metrics.adjusted_r2(feature_names,np.exp(y_valid.VentesUC_log_transformed.values) - 1, yhat)
    print('RMSPE: {:.6f}'.format(error_test))
    print('R Squarred: {:.6f}'.format(R_squarred))
    print('R Squarred (adj): {:.6f}'.format(adj_2))
    
    

    print("\nPerformance on test set")
    dtest = xgb.DMatrix(X_test[features])
    test_probs = np.exp(gbm.predict(dtest)) - 1
    error_test = metrics.rmspe(np.exp(y_test.VentesUC_log_transformed.values) - 1, test_probs)
    R_squarred = r2_score(np.exp(y_test.VentesUC_log_transformed.values) - 1, test_probs)
    adj_2 = metrics.adjusted_r2(feature_names,np.exp(y_test.VentesUC_log_transformed.values) - 1, test_probs)
    print('RMSPE: {:.6f}'.format(error_test))
    print('R Squarred: {:.6f}'.format(R_squarred))
    print('R Squarred (adj): {:.6f}'.format(adj_2))
    
    
    # Evaluation metrics
    #------------------- Plot feature importances ----------------
    xgb.plot_importance(booster = gbm, show_values = False, importance_type = 'gain')
    plt.show()
      
    
    
    # ------------------- Perform SHAP Analysis on training data --------------------
    X_shap, _, y_shap, _ = train_test_split(X_train, y_train, test_size = 0.95)  #reduction du nombre de lignes a analyser
    SHAP_Analysis(gbm, X_shap, y_shap, feature_names)
    

    return gbm, watchlist




    
def train_deploy_model(df,enseigne):
    """
    Cette fonction récupère les hyperparametres enrigistré dans le local directory pour l'enseigne concernée
    et entraine et suvegarde le modèle de prédiction pour réaliser les prédictions à postériori
    """
    
    #Get data
    X, _ = preprocessing.training_set_preprocessing.preco_features(df)
    y, _ = preprocessing.training_set_preprocessing.preco_target(df)
    
    
    #split train, validation and testing data
    #cette partie pourra être remplacée par un CV de la librairie sklearn à posteriori
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state=42)
    
    _,features = preprocessing.training_set_preprocessing.preco_features(df)



    #Reconstructing data Matrix for Model training
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    
  
    
    #Model training
    print("Training a XGBoost model")
    
    start_time = time()
    gbm = xgb.train(params, dtrain, num_boost_round = parameters.num_boost_round, evals=watchlist, early_stopping_rounds=parameters.early_stopping_rounds, verbose_eval=50)
  
    fitting_time = time()-start_time

    print('Total fitting time = {:0.2f}s.'.format(fitting_time))
    
    #Save Trained Model
    dump(gbm, 'trained_XBG_'+enseigne+'.joblib')
    
    print('Model saved in local directory as "trained_XGB_<enseigne>.joblib"')

