#!/usr/bin/env python
# coding: utf-8

# In[ ]: 


"""
Created on 09/09/2019

@author: Yannick Bottino
"""

import functions
import settings

settings.init()

enseigne = settings.enseigne


#------------------ XGB parameters -----------------------
xgb_params = functions.load_obj('xgb_params_'+enseigne)
#xgb_params = functions.load_obj('xgb_params')


# ---- Grille des hyperparamètres utilisés pour le hyperparameter tuning ------
xgb_grid = {'max_depth':[7, 9, 11], 
            'learning_rate':[0.1],
            'n_estimators':[250, 400] ,
            'verbosity':[1], 
            'silent':[0], 
            'objective':['reg:squarederror' ],  #Cet objectif est l'objectif RMSE classique pour des regressions
            'booster':['gbtree'],
            'n_jobs':[-1], 
            'nthread':[-1], 
            'gamma':[0.05],  
            'reg_alpha':[1], 
            'reg_lambda':[1, 1.5],
            'importance_type':['gain']}


# ----- Paramètres de boosting du XGB ------
num_boost_round = 200
early_stopping_rounds = 10