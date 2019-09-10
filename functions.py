#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Created on 09/09/2019

@author: Yannick Bottino
"""

import os
    
    
    
# Gather some features
def build_features(features, data):
    # Remplace NaNs by 0
    data.fillna(0, inplace=True)
    # Use some properties directly
    features.extend(data.columns)
    return data

