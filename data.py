#!/usr/bin/env python
# coding: utf-8

# In[ ]: 


import xgboost as xgb
import pandas as pd
import numpy as np
import operator
import settings

from time import time

import matplotlib.pyplot as plt

from google.cloud import bigquery
import google.datalab.bigquery as bq


client = bigquery.Client()



class data_extraction:
    def BDD_Promo(data_source,enseigne):
        #Fonction qui va aller chercher les données d'entrainement pour entrainement du modèle.
        #L'argument de cette fonction est source_data.
        #       source_data peut prendre 2 valeurs: 'csv' ou 'BigQuery'.
        #              "csv" pointe vers le fichier '../Test_OSA_GCP/data/BDD Promo.csv'
        #              "BigQuery" va lancer le script qui va récuperer les données depuis la table BigQuery associée
        
        if data_source == 'csv':
            
            print('\nloading training data from csv file...')
            #Charge la BDD Promo à partir du csv
            df = pd.read_csv('../Precos_OSA/data/BDD_Promos_V2.csv', sep=',')
            df = df.drop(['Unnamed: 0'], axis = 1)
            
        elif data_source == 'BigQuery':
            print('\nQuerying BigQuery for training data...')

            sql = """
            SELECT DISTINCT
                * 
            FROM 
                `osa-2019.Performance_Promos.HistoriquePromoPourTrainingAlgo` 
            WHERE Enseigne LIKE UPPER('"""+enseigne+"""')""" 
            

            start_time = time()

            df = client.query(sql).to_dataframe()         #Interrogation de BigQuery 
            
            df.to_csv('../Precos_OSA/data/BDD_Promos_V2.csv')

            print('Querying and loading time = {:0.2f} s '.format(time() - start_time))
            print('Request finished\n')
            
        else:
            raise ValueError('Veuillez utiliser csv ou BigQuery en argument de l objet BDD_Promo')
        
        return df
    
    
    def Forecast(data_source, enseigne):
        #Fonction qui va aller chercher les données pour prédictions.
        #L'argument de cette fonction est source_data.
        #       source_data peut prendre 2 valeurs: 'csv' ou 'BigQuery'.
        #              "csv" pointe vers le fichier '../Test_OSA_GCP/data/Forecast.csv'
        #              "BigQuery" va lancer le script qui va récuperer les données depuis la table BigQuery associée
        
        if data_source == 'csv':
            print('loading data to predict from csv file...')
            #Charge la BDD Promo à partir du csv
            df = pd.read_csv('../Precos_OSA/data/Forecast.csv', sep=';')
        elif data_source == 'BigQuery':
            nom_ope = input("Sur quelle OP voulez-vous réaliser des précos ? (Entrer le nom de l'OP à l'identique de Recas CAS)")
            date_debut_conso = input("Quelle est la date de début de conso de l'OP ? (Entrer la date au format AAAA-MM-JJ)")
            
            print('\nQuerying BigQuery for prediction data...')

            sql = """
            SELECT DISTINCT
                * 
            FROM 
                `osa-2019.Performance_Promos.HistoriquePromoPourTrainingAlgo` 
            WHERE 
                NomOpe LIKE '"""+nom_ope+"""'
                AND DateDebutConso = '"""+date_debut_conso+"""'
                AND Enseigne LIKE UPPER('"""+enseigne+"""')""" 
            

            start_time = time()

            df = client.query(sql).to_dataframe()         #Interrogation de BigQuery 
            
            df.to_csv('../Precos_OSA/data/Forecast.csv')

            print('Querying and loading time = {:0.2f} s '.format(time() - start_time))
            print('Request finished\n')
        else:
            raise ValueError('Veuillez utiliser csv ou BigQuery en argument de l objet data_extraction')
        
        return df, nom_ope

