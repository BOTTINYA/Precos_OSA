#!/usr/bin/env python
# coding: utf-8

# In[ ]: 


"""
Created on 09/09/2019

@author: Yannick Bottino
"""

import os
import pickle




def save_obj(obj, name ):
    """
    Fonction qui récupère un objet quel qu'il soit et le converti en obj pkl avec le nom souhaité
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    """
    Fonction qui récupère un objet converti en pkl
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)