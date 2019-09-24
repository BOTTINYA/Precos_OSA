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
import settings

import joblib

import pandas as pd
import numpy as np


# *****************************************************************
# ------------------------ Program parameters ----------------------
# *****************************************************************


enseigne = settings.enseigne

# --------------- Ce que l'on souhaite faire avec le modèle ----------------------
Training_of_model = input('Do you want to re-train the model ? (Y/n)')                                                          #Veut-on ré-entrainer le modèle ?
For_Deployment = 'n'
if Training_of_model == 'Y':
    For_Deployment = input('Do you wish to re-train the model for deployment or testing ? (Y for deployment/ n for testing)')   #Si nous ré-entrainons le modèle, veut-on faire l'étape de validation ou souhaitons-nous le l'entrainer sur toutes les données pour le déployer ensuite ?

    
# -------------- Identification de la table dans laquelle on va exporter les données -----------------
bigquery_dataset_name = 'osa-2019.precos'
bigquery_table_name = 'last_precos_raw_'+enseigne

# -------------- Identification du bucket dans lequel on va exporter les données -----------------
bucket_name = '<bucket_name>'
file_destination_name = 'last_precos_raw_'+enseigne+'.csv'



# *********************************************************************
# ------------------------ Program script -----------------------------
# *********************************************************************



# ------------------ Entrainement du modèle -------------------------
if Training_of_model == 'Y':
    """
    Dans un premier temps, on verifie si on souhaite ré-entrainer le modèle, 
    et si oui, est-ce dans le but de le déployer ou de tester des paramètres en vue de validation.
    """
    
      
    #Data extraction for training model
    df = data.data_extraction.BDD_Promo('BigQuery', enseigne)
    #Data Cleaning
    df_clean = preprocessing.training_set_preprocessing.training_set_cleaning(df)
    
    #Data Encoding
    print('\nPerforming training data encoding')
    df_encoded = preprocessing.training_set_preprocessing.feature_encoding(df_clean)
    print('Data encoding finished\n')
    
    
    
    if For_Deployment == 'n':      
        """Souhaite-t-on faire l'étape de validation ?"""
        model, watchlist = model_training.train_validate_model(df_encoded, enseigne)
    
    else:
        """ On va entrainer le modèle en vue de déploiement, puis faire les prédictions sur celui-ci.
        """
        model_training.train_deploy_model(df_encoded,enseigne)
        
        
# --------------------- On souhaite utiliser le modèle sauvegardé -------------------
else:
    #Load trained model
    gbm = joblib.load('trained_XBG_'+enseigne+'.joblib')
    print('Predictions will be performed using trained_XGB_'+enseigne+'.joblib regression model')
    
    #Get data to perform prediction on
    F = data.data_extraction.Forecast('csv')
    
    #Get training features
    #Data extraction for used to train model
    df = data.data_extraction.BDD_Promo('BigQuery', enseigne)
    #Data Cleaning
    df_clean = preprocessing.training_set_preprocessing.training_set_cleaning(df)
    
    #Data Encoding
    df_encoded = preprocessing.training_set_preprocessing.feature_encoding(df_clean)
    
    
    
    
    _,features = preprocessing.training_set_preprocessing.preco_features(df_train)
    
    #Perform predictions
    y_pred = predictions.perform_predictions(F,features,gbm)
    y_pred = np.round_(y_pred, decimals = 0)
    
    df_pred = pd.DataFrame(data = y_pred, columns = ['Preconisations_Ventes'])    #construction du DataFrame pour concatenation des données de préco
    
    #Construction du DataFrame final des preco
    Precos = pd.concat([F[features],df_pred], axis = 1)
    Precos = Precos.dropna()
    
    exportation.BigQuery_exportation(Precos, bigquery_dataset_name, bigquery_table_name)
    exportation.export_forecast_to_GCS(Precos, bucket_name, file_destination_name)