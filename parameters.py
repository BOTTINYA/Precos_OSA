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
            'reg_alpha':[0.5 ,1], 
            'reg_lambda':[0.7, 1, 1.5],
            'importance_type':['gain']}

num_boost_round = 500
early_stopping_rounds = 10