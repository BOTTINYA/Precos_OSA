#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Created on 09/09/2019

@author: Yannick Bottino
"""

import data
import model_training
import preprocessing
import predictions
import exportation

import joblib

import pandas as pd
import numpy as np



# --------------- Ce que l'on souhaite faire avec le modèle ----------------------
Training_of_model = input('Do you want to re-train the model ? (Y/n)')                                                          #Veut-on ré-entrainer le modèle ?
For_Deployment = 'n'
if Training_of_model == 'Y':
    For_Deployment = input('Do you wish to re-train the model for deployment or testing ? (Y for deployment/ n for testing)')   #Si nous ré-entrainons le modèle, veut-on faire l'étape de validation ou souhaitons-nous le l'entrainer sur toutes les données pour le déployer ensuite ?

# -------------- Identification de la table dans laquelle on va exporter les données -----------------
dataset_id = 'precos'
table_name = 'last_preco'



# ------------------ Entrainement du modèle -------------------------
if Training_of_model == 'Y':
    """Dans un premier temps, on verifie si on souhaite ré-entrainer le modèle, 
       et si oui, est-ce dans le but de le déployer ou de tester des paramètres en vue de validation.
    """
    
      
    #Data extraction for training model
    df = data.data_extraction.BDD_Promo('csv')
    #Data Cleaning
    df = preprocessing.training_set_preprocessing.training_set_cleaning(df)
    
    
    
    if For_Deployment is 'n':      
        """Souhaite-t-on faire l'étape de validation ?"""
        
        model, watchlist = model_training.train_validate_model(df)
    
    else:
        """ On va entrainer le modèle en vue de déploiement, puis faire les prédictions sur celui-ci.
        """
        model_training.train_deploy_model(df)
        
        #Load trained model
        gbm = joblib.load('trained_XBG.joblib')
    
        #Get data to perform prediction on
        F = data.data_extraction.Forecast('csv')
    
        #Get training features
        df_train = data.data_extraction.BDD_Promo('csv')
        _,features = preprocessing.training_set_preprocessing.preco_features(df_train)
    
        #Perform predictions
        y_pred = predictions.perform_predictions(F,features,gbm)
        y_pred = np.round_(y_pred, decimals = 0)
        
        df_pred = pd.DataFrame(data = y_pred, columns = ['Pre_commandes'])    #construction du DataFrame pour concatenation des données de préco
    
        #Construction du DataFrame final des preco
        Precos = pd.concat([F[features],df_pred], axis = 1)
    
#        exportation.BigQuery_exportation(Precos, dataset_id, table_name)
        
        
# --------------------- On souhaite utiliser le modèle sauvegardé -------------------
else:
    #Load trained model
    gbm = joblib.load('trained_XBG.joblib')
    
    #Get data to perform prediction on
    F = data.data_extraction.Forecast('csv')
    
    #Get training features
    df_train = data.data_extraction.BDD_Promo('csv')
    _,features = preprocessing.training_set_preprocessing.preco_features(df_train)
    
    #Perform predictions
    y_pred = predictions.perform_predictions(F,features,gbm)
    y_pred = np.round_(y_pred, decimals = 0)
    
    df_pred = pd.DataFrame(data = y_pred, columns = ['Pre_commandes'])    #construction du DataFrame pour concatenation des données de préco
    
    #Construction du DataFrame final des preco
    Precos = pd.concat([F[features],df_pred], axis = 1)
    
#    exportation.BigQuery_exportation(Precos, dataset_id, table_name)

