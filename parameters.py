#!/usr/bin/env python
# coding: utf-8

# In[ ]: 


"""
Created on 09/09/2019

@author: Yannick Bottino
"""

import functions


#------------------ XGB parameters -----------------------
xgb_params = functions.load_obj('xgb_params')

xgb_grid = {'max_depth':[7,9], 
            'learning_rate':[0.1, 0.3],
            'n_estimators':[100,250,500] ,
            'verbosity':[1], 
            'silent':[0], 
            'objective':['reg:squarederror'],
            'booster':['gbtree'],
            'n_jobs':[-1], 
            'nthread':[-1], 
            'gamma':[0],  
            'reg_alpha':[0], 
            'reg_lambda':[1],
            'importance_type':['gain']}

num_boost_round = 500
early_stopping_rounds = 30