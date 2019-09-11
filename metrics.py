#!/usr/bin/env python
# coding: utf-8

# In[ ]: 


"""
Created on 09/09/2019

@author: Yannick Bottino
"""

import os
import numpy as np
import operator
from sklearn.metrics import r2_score



#define RMSPE for XGBOOST

def rmspe(y, yhat):
    yhat[yhat<0]=0
    return np.nanmean(np.absolute(y-yhat))
    
#    np.minimum(np.absolute(y-yhat)/yhat, 1)
    
def rmspe_xg(yhat, y):
    y = y.get_label()
    #y = y.get_label()
    yhat = yhat
    #yhat = yhat
    return "rmspe", rmspe(y,yhat)

def adjusted_r2(feature_names, y_true, y_pred):
    n = len(y_true)
    p = len(feature_names)
    r2 = r2_score(y_true,y_pred)
    adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)
    return adj_r2
