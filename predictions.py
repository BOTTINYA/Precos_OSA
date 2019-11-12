#!/usr/bin/env python
# coding: utf-8

# In[ ]: 


"""
Created on 09/09/2019

@author: Yannick Bottino
"""


import xgboost as xgb
import numpy as np
import pandas as pd
import math



def perform_predictions(data,features,model):
    """Cette fonction prend en argument:
        - F : DataFrame des lignes à prédire
        - features : le nom des colonnes du dataset d'entrainement
        - model : le model sur lequel on doit faire les prédictions
    """
    print('\nPerforming predictions from the trained model')
    dforecast = xgb.DMatrix(data[features])
    
    target_col = [col for col in list(data) if 'VentesUC' in col]
    
    
    if 'transformed' in target_col[0]:
        forecast_transformed = model.predict(dforecast)
        #perform backtransform of prediction values
        forecast = np.exp(forecast_transformed)-1
    else:
        forecast = model.predict(dforecast)
    
    print('\nPredictions finished')
    return forecast

