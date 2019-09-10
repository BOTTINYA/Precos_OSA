#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import xgboost as xgb
import pandas as pd
import numpy as np
import operator

import matplotlib.pyplot as plt

from google.cloud import bigquery
import google.datalab.bigquery as bq



class data_extraction:
    def BDD_Promo(data_source):
        #Fonction qui va aller chercher les données d'entrainement pour entrainement du modèle.
        #L'argument de cette fonction est source_data.
        #       source_data peut prendre 2 valeurs: 'csv' ou 'BigQuery'.
        #              "csv" pointe vers le fichier '../Test_OSA_GCP/data/BDD Promo.csv'
        #              "BigQuery" va lancer le script qui va récuperer les données depuis la table BigQuery associée
        
        if data_source == 'csv':
            #Charge la BDD Promo à partir du csv
            df = pd.read_csv('../code for production/data/BDD_Promo.csv', sep=';')
        elif data_source == 'BigQuery':
            raise ValueError('Le code pour sourcing de la BDD Promo depuis BigQuery n est pas encore écrit. Veuillez utiliser csv en argument de l objet data_extraction pour le moment.')
        else:
            raise ValueError('Veuillez utiliser csv ou BigQuery en argument de l objet data_extraction')
        
        return df
    
    
    def Forecast(data_source):
        #Fonction qui va aller chercher les données pour prédictions.
        #L'argument de cette fonction est source_data.
        #       source_data peut prendre 2 valeurs: 'csv' ou 'BigQuery'.
        #              "csv" pointe vers le fichier '../Test_OSA_GCP/data/Forecast.csv'
        #              "BigQuery" va lancer le script qui va récuperer les données depuis la table BigQuery associée
        
        if data_source == 'csv':
            #Charge la BDD Promo à partir du csv
            df = pd.read_csv('../code for production/data/Forecast.csv', sep=';')
        elif data_source == 'BigQuery':
            raise ValueError('Le code pour sourcing de la BDD Promo depuis BigQuery n est pas encore écrit. Veuillez utiliser csv en argument de l objet data_extraction pour le moment.')
        else:
            raise ValueError('Veuillez utiliser csv ou BigQuery en argument de l objet data_extraction')
        
        return df

