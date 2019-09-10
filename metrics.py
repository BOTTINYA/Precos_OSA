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

