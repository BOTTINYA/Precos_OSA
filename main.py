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
import string

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# *****************************************************************
# ------------------------ Program parameters ----------------------
# *****************************************************************


enseigne = settings.enseigne

# --------------- Ce que l'on souhaite faire avec le modèle ----------------------
Training_of_model = input('Souhaitez-vous ré-entrainer le modèle ? (Y/n)')                                                          #Veut-on ré-entrainer le modèle ?
For_Deployment = 'n'
if Training_of_model.upper() == 'Y':
    For_Deployment = input('Souhaitez-vous ré-entrainer le modèle pour déploiement ou phase de test ? (Y pour deploiement/ n pour phase de test)')   #Si nous ré-entrainons le modèle, veut-on faire l'étape de validation ou souhaitons-nous le l'entrainer sur toutes les données pour le déployer ensuite ?

    
# -------------- Identification de la table dans laquelle on va exporter les données -----------------
#bigquery_dataset_name = 'osa-2019.Performance_Promos'
#bigquery_table_name = 'last_precos_raw_'+enseigne+'_test'

# -------------- Identification du bucket dans lequel on va exporter les données -----------------
#bucket_name = 'osa_data_bucket'
#file_destination_name = 'last_precos_raw_'+enseigne+'.csv'



# *********************************************************************
# ------------------------ Program script -----------------------------
# *********************************************************************



# ------------------ Entrainement du modèle -------------------------
if Training_of_model.upper() == 'Y':
    """
    Dans un premier temps, on verifie si on souhaite ré-entrainer le modèle, 
    et si oui, est-ce dans le but de le déployer ou de tester des paramètres en vue de validation.
    """
    
      
    #Data extraction for training model
    df = data.data_extraction.BDD_Promo('BigQuery', enseigne)
    #Data Cleaning
    df_clean, identification_columns = preprocessing.training_set_preprocessing.training_set_cleaning(df)      #identification_columns est une liste des colonnes qui ne seront pas utilisées par l'algo pour les
                                                                                                               #prédictions mais qui sont nécessaires pour l'identification du mag et 
                                                                                                               #de l'EAN sur lequel on va faire la préco
    
    #Data Transforamtion pour normalisation des colonnes Skewed
    df_transform = preprocessing.training_set_preprocessing.data_forward_transform(df_clean)
    
    #Data Encoding
    print('\nPerforming training data encoding')
    df_encoded = preprocessing.training_set_preprocessing.feature_encoding(df_transform)
    print('Data encoding finished\n')
    
    
    
    #Selecting only training columns from DataFrame
    transformed_columns = [col for col in df_encoded if 'transformed' in col]
    training_columns = list(df_encoded.columns)
    
    for x in df_encoded.columns:
        if set([x+'_log_transformed']) & set(transformed_columns) != set():
            training_columns.pop(training_columns.index(x))

    final_training_columns = np.setdiff1d(training_columns,identification_columns)
    
    
    
    df_encoded = df_encoded.dropna()
    
    
    
    if For_Deployment.upper() == 'N':      
        """Souhaite-t-on faire l'étape de validation ?"""
        
       #Afficher la distribution des données
        print('Affichage de la distribution des données\n')
        fig = plt.figure
        fig(figsize =(20,30))
        i=0
        for col in final_training_columns:
            plt.subplot(7, 4, i + 1)
            plt.title(col)
            sns.distplot(df_encoded[[col]],  kde_kws={'bw':0.3})
            i+=1

        plt.tight_layout
        plt.show()
        print('\n')
        
    
        #Afficher la matrice de correlation des colonnes
        print('Affichage de la corrélation des colonnes\n')
        X_corr = df_encoded[final_training_columns].corr()
        mask = np.zeros_like(X_corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        fig = plt.figure(figsize = (12,12))

        sns.heatmap(X_corr, center=0, square = True, mask = mask)
        plt.show()
        
        print('\n')
    
        # Entrainer le modèle pour validation et interpretation
        model, watchlist = model_training.train_validate_model(df_encoded[final_training_columns], enseigne)
    
    else:
        """ On va entrainer le modèle en vue de déploiement, puis faire les prédictions sur celui-ci.
        """
        model_training.train_deploy_model(df_encoded[final_training_columns],enseigne)
        
        
# --------------------- On souhaite utiliser le modèle sauvegardé -------------------
else:
    #Load trained model
    gbm = joblib.load('trained_XBG_'+enseigne+'.joblib')
    print('Predictions will be performed using trained_XGB_'+enseigne+'.joblib regression model')
    
    #Get data to perform prediction on
    F, nom_ope = data.data_extraction.Forecast('BigQuery', enseigne)
    

    #Data Cleaning
    F_clean, identification_columns = preprocessing.training_set_preprocessing.training_set_cleaning(F)
    
    #Data Transforamtion pour normalisation des colonnes Skewed
    F_transform = preprocessing.training_set_preprocessing.data_forward_transform(F_clean)
    
    #Data Encoding
    print('\nPerforming training data encoding')
    F_encoded = preprocessing.training_set_preprocessing.feature_encoding(F_transform)
    print('Data encoding finished\n')

    
    
    #Selecting only training columns from DataFrame
    transformed_columns = [col for col in F_encoded if 'transformed' in col]
    training_columns = list(F_encoded.columns)
    
    for x in F_encoded.columns:
        if set([x+'_log_transformed']) & set(transformed_columns) != set():
            training_columns.pop(training_columns.index(x))

            
    _,target_column = preprocessing.training_set_preprocessing.preco_target(F_encoded)
    
    final_training_columns = np.setdiff1d(np.setdiff1d(training_columns,identification_columns),target_column)

    
    
    
    #Perform predictions
    y_pred = predictions.perform_predictions(F_encoded,final_training_columns,gbm)
    y_pred = np.int_(np.round_(y_pred, decimals = 0))
    
    df_pred = pd.DataFrame(data = y_pred, columns = ['PreconisationVentesUC'])    #construction du DataFrame pour concatenation des données de préco
    
    
    
    #Construction du DataFrame des preco
    #identification_columns.append('VentesUC')          #Je n'tutilise cette ligne de code uniquement pour tester les perf du code en phase de test
    
    Forecast = pd.concat([F_encoded[identification_columns],df_pred],axis = 1)

    
    Forecast.to_csv('../Precos_OSA/data/Precos_'+nom_ope+'.csv')
    #exportation.BigQuery_exportation(Precos, bigquery_dataset_name, bigquery_table_name)
    #exportation.export_forecast_to_GCS(Precos, bucket_name, file_destination_name)