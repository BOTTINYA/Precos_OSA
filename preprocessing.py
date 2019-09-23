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
    #Use only Sales bigger then zero. Simplifies calculation of rmspe
    def training_set_cleaning(df):
        data = df[df['ventes'] > 0]
        return data
    
    def preco_features(df):
        features = df.drop(['ventes'], axis=1)
        feature_names = list(features)
        return features, feature_names
    
    def preco_target(df):
        target = pd.DataFrame(df['ventes'])
        target_name = list(target)
        return target, target_name

    
#    def feature_encoding(df):
    
