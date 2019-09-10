# -*- coding: utf-8 -*-
"""
Created on 09/09/2019

@author: Yannick Bottino
"""


import xgboost as xgb
import pandas as pd
import numpy as np
import operator

import matplotlib.pyplot as plt

from google.cloud import bigquery
import google.datalab.bigquery as bq



class data_extraction:
    def BDD_Promo(data_source):
        #Fonction qui va aller chercher les donn�es d'entrainement pour entrainement du mod�le.
        #L'argument de cette fonction est source_data.
        #       source_data peut prendre 2 valeurs: 'csv' ou 'BigQuery'.
        #              "csv" pointe vers le fichier '../Test_OSA_GCP/data/BDD Promo.csv'
        #              "BigQuery" va lancer le script qui va r�cuperer les donn�es depuis la table BigQuery associ�e
        
        if data_source == 'csv':
            #Charge la BDD Promo � partir du csv
            df = pd.read_csv('../code for production/data/BDD_Promo.csv', sep=';')
        elif data_source == 'BigQuery':
            raise ValueError('Le code pour sourcing de la BDD Promo depuis BigQuery n est pas encore �crit. Veuillez utiliser csv en argument de l objet data_extraction pour le moment.')
        else:
            raise ValueError('Veuillez utiliser csv ou BigQuery en argument de l objet data_extraction')
        
        return df
    
    
    def Forecast(data_source):
        #Fonction qui va aller chercher les donn�es pour pr�dictions.
        #L'argument de cette fonction est source_data.
        #       source_data peut prendre 2 valeurs: 'csv' ou 'BigQuery'.
        #              "csv" pointe vers le fichier '../Test_OSA_GCP/data/Forecast.csv'
        #              "BigQuery" va lancer le script qui va r�cuperer les donn�es depuis la table BigQuery associ�e
        
        if data_source == 'csv':
            #Charge la BDD Promo � partir du csv
            df = pd.read_csv('../code for production/data/Forecast.csv', sep=';')
        elif data_source == 'BigQuery':
            raise ValueError('Le code pour sourcing de la BDD Promo depuis BigQuery n est pas encore �crit. Veuillez utiliser csv en argument de l objet data_extraction pour le moment.')
        else:
            raise ValueError('Veuillez utiliser csv ou BigQuery en argument de l objet data_extraction')
        
        return df
        

        
       