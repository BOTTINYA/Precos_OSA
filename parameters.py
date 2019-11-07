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


xgb_grid = {'max_depth':[7,9], 
            #'learning_rate':[0.1, 0.3],
            'n_estimators':[100,250] ,
            'verbosity':[1], 
            'silent':[0], 
            'objective':['reg:squarederror'],
            'booster':['gbtree'],
            'n_jobs':[-1], 
            'nthread':[-1], 
            'gamma':[0, 0.1],  
            'reg_alpha':[0, 0.1 , 0.5], 
            'reg_lambda':[0,0.5,1, 1.3],
            'importance_type':['gain']}

num_boost_round = 50
early_stopping_rounds = 50