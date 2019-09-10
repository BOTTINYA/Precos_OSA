# -*- coding: utf-8 -*-
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



# --------------- Ce que l'on souhaite faire avec le mod�le ----------------------
Training_of_model = True      #Veut-on r�-entrainer le mod�le ?
For_Deployement = False    #Si nous r�-entrainons le mod�le, veut-on faire l'�tape de validation ou souhaitons-nous le l'entrainer sur toutes les donn�es pour le d�ployer ensuite ?

# -------------- Identification de la table dans laquelle on va exporter les donn�es -----------------
dataset_id = 'precos'
table_name = 'last_preco'



# ------------------ Entrainement du mod�le -------------------------
if Training_of_model is True:
    """Dans un premier temps, on verifie si on souhaite r�-entrainer le mod�le, 
       et si oui, est-ce dans le but de le d�ployer ou de tester des param�tres en vue de validation.
    """
    
      
    #Data extraction for training model
    df = data.data_extraction.BDD_Promo('csv')
    #Data Cleaning
    df = preprocessing.training_set_preprocessing.training_set_cleaning(df)
    
    
    
    if For_Deployement is False:      
        """Souhaite-t-on faire l'�tape de validation ?"""
        
        model, watchlist = model_training.train_validate_model(df)
    
    else:
    """ On va entrainer le mod�le en vue de d�ploiement, puis faire les pr�dictions sur celui-ci. """
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
        
        df_pred = pd.DataFrame(data = y_pred, columns = ['Pre_commandes'])    #construction du DataFrame pour concatenation des donn�es de pr�co
    
        #Construction du DataFrame final des preco
        Precos = pd.concat([F[features],df_pred], axis = 1)
    
        exportation.BigQuery_exportation(Precos, dataset_id, table_name)
        
        
# --------------------- On souhaite utiliser le mod�le sauvegard� -------------------
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
    
    df_pred = pd.DataFrame(data = y_pred, columns = ['Pre_commandes'])    #construction du DataFrame pour concatenation des donn�es de pr�co
    
    #Construction du DataFrame final des preco
    Precos = pd.concat([F[features],df_pred], axis = 1)
    
    exportation.BigQuery_exportation(Precos, dataset_id, table_name)