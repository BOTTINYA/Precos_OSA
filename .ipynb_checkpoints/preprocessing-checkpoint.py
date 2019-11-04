#!/usr/bin/env python
# coding: utf-8

# In[ ]: 


"""
Created on 09/09/2019

@author: Yannick Bottino
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



#Ce script va filtrer les lignes sur lesquelles nous voulons entrainer l'algorithme.
#Nous n'allons considerer que les codes pour lesquels nous avons des ventes

class training_set_preprocessing:
    #Use only Sales bigger than zero. Simplifies calculation of rmspe
    
    
    def training_set_cleaning(df):
        
        columns = list(df)
        
        columns_to_drop = ['Annee','EngagementUC','DateDebutConso','DateFinConso']
        for col in columns_to_drop:
            df = df.drop([col], axis = 1)
        
        for col in (set(columns) - set(columns_to_drop)):                        #On vire les colonnes ou toutes les valeurs de la colonne sont nulles
            if df[col].isnull().all() == True:
                df = df.drop([col], axis = 1)
            else:
                pass
            
        return df
    
    def preco_features(df):
        features = df.drop(['ventes'], axis=1)
        feature_names = list(features)
        return features, feature_names
    
    def preco_target(df):
        target = pd.DataFrame(df['ventes'])
        target_name = list(target)
        return target, target_name

    
    def feature_encoding(df):
        """
        Cette fonction réalise l'encodage des colonnes catégoriques ou non selon differentes stratégies:
        - Pour les colonnes catégoriques très spécifiques et à grand nombre de niveaux (code SAP, nom magasin, ...), la fonction va soit faire un label encoding, soit supprimer la colonne afin d'utiliser d'autres colonnes
        - Pour les colonnes catégoriques à nombre de niveau plus faible (<15), la fonction va réaliser un One Hot Encoding
        - Pour certaines colonnes du choix de l'utilisateur, une stratégie plus spécifique peut être définie
        
        La fonction renvoie le dataframe encodé.
        """
        categorical_columns = list(df.select_dtypes(include=['category','object']))
                   
            
        specific_columns = []        
        
        data = df.copy()
        for col in categorical_columns:
            if col in specific_columns:
                pass
            elif len(data[col].unique()) > 15:
                data = data.drop([col], axis=1)
            else:
                data = pd.concat([data,pd.get_dummies(data[col],prefix=col)],axis=1).drop([col],axis=1)
        return data

    
    
class prediction_data_preprocessing:
    pass