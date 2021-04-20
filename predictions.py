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
    
    forecast_transformed = model.predict(dforecast)
    
    #perform backtransform of prediction values
    forecast = np.exp(forecast_transformed)-1

    
    print('\nPredictions finished')
    return forecast


def f_15(x):
    return round(x*1.15,0)

def f_7_5(x):
    return round(x*1.075,0)


def boost_magasins_auchan (data):
    """
    Cette fonction considère une liste de magasins issus d'une analyse des performances promo sur Auchan.
    On va booster 'à la main' cette liste de magasins sur lesquels nous avons tendance à sous-engager. 
    Le facteur de boost à été défini par Jeremy comme de 15 % pour une liste de 29 mags, de 7.5% sur les top 30 mag en CA
    """
        
    magasins_sous_engages_auchan = ["AUCHAN MARTIGUES 24",
                                    "AUCHAN VELIZY 48",
                                    "AUCHAN NOYELLES 7",
                                    "AUCHAN MACON 150",
                                    "AUCHAN AUBAGNE 27",
                                    "AUCHAN TOULOUSE 70",
                                    "AUCHAN LE PONTET 14",
                                    "AUCHAN OLIVET 19",
                                    "AUCHAN CHASSENEUIL 140",
                                    "AUCHAN RONCQ 2",
                                    "AUCHAN BAGNOLET 54",
                                    "AUCHAN SETE 163",
                                    "AUCHAN PLAISIR 16",
                                    "AUCHAN MERIADECK 29",
                                    "AUCHAN DUNKERQUE 11",
                                    "AUCHAN DOMERAT 152",
                                    "AUCHAN NEUILLY 137",
                                    "AUCHAN MARSEILLE 67",
                                    "AUCHAN CAMBRAI 26",
                                    "AUCHAN LOUVROIL 10",
                                    "AUCHAN ISSY 66",
                                    "AUCHAN MAUREPAS 65",
                                    "AUCHAN MONTIVILLIERS 104",
                                    "AUCHAN VILLARS 45",
                                    "AUCHAN MANOSQUE 160",
                                    "AUCHAN TAVERNY 107",
                                    "AUCHAN CERGY 49",
                                    "AUCHAN MEAUX 058",
                                    "AUCHAN ARRAS 69",
                                    "AUCHAN TOURS NORD 108"]
    
    top_30_magasins_auchan = ["AUCHAN OLIVET 19",
                                "AUCHAN NOYELLES 7",
                                "AUCHAN MARTIGUES 24",
                                "AUCHAN MEAUX 058",
                                "AUCHAN ARRAS 69",
                                "AUCHAN RONCQ 2",
                                "AUCHAN DOMERAT 152",
                                "AUCHAN NEUILLY 137",
                                "AUCHAN MONTIVILLIERS 104",
                                "AUCHAN VILLARS 45",
                                "AUCHAN SETE 163",
                                "AUCHAN MARSEILLE 67",
                                "AUCHAN CERGY 49",
                                "AUCHAN LE PONTET 14",
                                "AUCHAN MAUREPAS 65",
                                "AUCHAN AUBAGNE 27",
                                "AUCHAN TOULOUSE 70",
                                "AUCHAN TOURS NORD 108",
                                "AUCHAN VELIZY 48",
                                "AUCHAN MERIADECK 29",
                                "AUCHAN CAMBRAI 26",
                                "AUCHAN MANOSQUE 160",
                                "AUCHAN MACON 150",
                                "AUCHAN TAVERNY 107",
                                "AUCHAN ISSY 66",
                                "AUCHAN LOUVROIL 10",
                                "AUCHAN DUNKERQUE 11",
                                "AUCHAN PLAISIR 16",
                                "AUCHAN CHASSENEUIL 140",
                                "AUCHAN BAGNOLET 54"]
    
    liste_top_30_hors_mag_sous_engages = set(top_30_magasins_auchan) - set(magasins_sous_engages_auchan)
    
    if 'AUCHAN' in data.Enseigne.unique():

        
        data['PreconisationVentesUC'] = np.where(data['NomMagasinMicroStrat'].isin(magasins_sous_engages_auchan), data['PreconisationVentesUC'].apply(f_15), data['PreconisationVentesUC'])
        
        data['PreconisationVentesUC'] = np.where(data['NomMagasinMicroStrat'].isin(liste_top_30_hors_mag_sous_engages), data['PreconisationVentesUC'].apply(f_7_5), data['PreconisationVentesUC'])
        
    else:
        pass
    
    return data
    
    
    