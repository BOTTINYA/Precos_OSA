# -*- coding: utf-8 -*-
"""
Created on 09/09/2019

@author: Yannick Bottino
"""


import xgboost as xgb
import numpy as np
import pandas as pd



def perform_predictions(data,features,model):
    """Cette fonction prend en argument:
        - F : DataFrame des lignes à prédire
        - features : le nom des colonnes du dataset d'entrainement
        - model : le model sur lequel on doit faire les prédictions
    """
    
    dforecast = xgb.DMatrix(data[features])
    forecast = model.predict(dforecast)
    return forecast