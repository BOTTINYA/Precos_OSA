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



class training_set_preprocessing:
    
    
    def training_set_cleaning(df):
        
        print('Cleaning Data...')
        columns = df.columns
        
        
        
        columns_to_drop = ['Confirmation',
                           'Annee', 
                           'NomOpe', 
                           'SemaineDebut',
                           'DateDebutConso', 
                           'DateFinConso', 
                           'Enseigne',
                           'DirectionRegionale', 
                           'ZoneCVR', 
                           'SecteurCM', 
                           'NomMagasin',
                           'CodeMagasin', 
                           'CodeSAPProduit',
                           'EANProduit', 
                           'NomProduit',
                           'Mecanique',
                           'MaxVentesEANPasseeEnUC',
                           'UmbrellaBrand']
        
        if 'AUCHAN' in df.Enseigne.unique():
            columns_to_drop.append('CodeLot')
        
        for col in columns_to_drop:
            #df = df.drop([col], axis = 1)
            pass
        
        for col in (set(columns) - set(columns_to_drop)):                        #On vire les colonnes ou toutes les valeurs de la colonne sont nulles
            if df[col].isnull().all() == True:
                df = df.drop([col], axis = 1)
            else:
                pass
        
        #df = df.dropna()
        print('Data cleansing done\n')
        
        return df, columns_to_drop
    
    
    
    
    def data_forward_transform(df):
        print('Transforming skewed columns for Normal distribution approximation...')
        columns = df.columns
        columns_to_log_transform = ['CAMagasin',
                                    'BaselineOSA',
                                    'DureeEnJoursDepuisLancement',
                                    'TotalVentesMarqueUC',
                                    'VentesTotalesUG',
                                    'VentesTotalesProductBrandEnUC',
                                    'PreviVol',
                                    'PreviUCRetouche',
                                    'IndiceMagPromophile',
                                    'NBCodesJoues',
                                    'NBJours',
                                    'TauxDeDegradation',
                                    'NBProductBrandJoues',
                                    'VentesUC']
        
        for col in (set(columns_to_log_transform) & set (columns)):
            df[col+'_log_transformed'] = np.log(df[col]+1)
        return df
        
        
        
        
    def data_back_transform(df):
        transformed_cols = [col for col in df.columns if '_log_transform' in col]
        
        for col in transformed_cols:
            df[col.replace("_log_transformed","")] = np.exp(df[col]) - 1
            df = df.drop('col', axis = 1)
            pass
        print ( 'Data transformation done\n')
        
        return df
    
    
    
    
    def preco_features(df):
        
        target_col = [col for col in list(df) if 'VentesUC' in col]
        
        
        features = df.drop([target_col[0]], axis=1)
        feature_names = list(features)
        
        return features, feature_names
    
    def preco_target(df):
        target_col = [col for col in list(df) if 'VentesUC_log_transformed' in col]
        
        target = pd.DataFrame(df[target_col])
        target_name = target_col
        
        return target, target_name

    
    
    
    
    def feature_encoding(df):
        """
        Cette fonction réalise l'encodage des colonnes catégoriques ou non selon differentes stratégies:
        - Pour les colonnes catégoriques très spécifiques et à grand nombre de niveaux (code SAP, nom magasin, ...), la fonction va soit faire un label encoding, soit supprimer la colonne afin d'utiliser d'autres colonnes
        - Pour les colonnes catégoriques à nombre de niveau plus faible (<10), la fonction va réaliser un One Hot Encoding
        - Pour certaines colonnes du choix de l'utilisateur, une stratégie plus spécifique peut être définie
        
        La fonction renvoie le dataframe encodé.
        """
        print('Feature encoding...')
        
        
        categorical_columns = list(df.select_dtypes(include=['category','object']))
                   
            
        specific_columns = []        
        
        data = df.copy()
        
        for col in categorical_columns:
            if col in specific_columns:
                pass
            elif len(data[col].unique()) > 3:
                pass
                #data = data.drop([col], axis=1)
            else:
                pass
                #data = pd.concat([data,pd.get_dummies(data[col],prefix=col)],axis=1).drop([col],axis=1)
                
        print('Feature encoding done\n')
        
        return data