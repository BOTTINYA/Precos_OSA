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
        
    magasins_sous_engages_auchan = ['10 - AUCHAN LOUVROIL',
                                    '104 - AUCHAN MONTIVILLIERS',
                                    '107 - AUCHAN TAVERNY',
                                    '108 - AUCHAN TOURS NORD',
                                    '11 - AUCHAN GRANDE SYNTHE',
                                    '137 - AUCHAN NEUILLY SUR MARNE',
                                    '14 - AUCHAN LE PONTET',
                                    '140 - AUCHAN CHASSENEUIL DU POITOU',
                                    '150 - AUCHAN MACON',
                                    '152 - AUCHAN DOMERAT',
                                    '16 - AUCHAN PLAISIR',
                                    '160 - AUCHAN MANOSQUE',
                                    '163 - AUCHAN SETE',
                                    '19 - AUCHAN OLIVET',
                                    '2 - AUCHAN RONCQ',
                                    '24 - AUCHAN MARTIGUES',
                                    '26 - AUCHAN ESCAUDOEUVRES',
                                    '27 - AUCHAN AUBAGNE',
                                    '29 - AUCHAN BORDEAUX MERIADECK',
                                    '45 - AUCHAN VILLARS',
                                    '48 - AUCHAN VELIZY VILLACOUBLAY',
                                    '49 - AUCHAN CERGY',
                                    '54 - AUCHAN BAGNOLET',
                                    '58 - AUCHAN MEAUX',
                                    '65 - AUCHAN MAUREPAS',
                                    '66 - AUCHAN ISSY LES MOULINEAUX',
                                    '67 - AUCHAN MARSEILLE PONT DE VIVAUX',
                                    '69 - AUCHAN ARRAS',
                                    '7 - AUCHAN NOYELLES GODAULT',
                                    '70 - AUCHAN TOULOUSE GRAMONT']
    
    top_30_magasins_auchan = ['10 - AUCHAN LOUVROIL',
                              '104 - AUCHAN MONTIVILLIERS',
                              '107 - AUCHAN TAVERNY',
                              '108 - AUCHAN TOURS NORD',
                              '11 - AUCHAN GRANDE SYNTHE',
                              '137 - AUCHAN NEUILLY SUR MARNE',
                              '14 - AUCHAN LE PONTET',
                              '140 - AUCHAN CHASSENEUIL DU POITOU',
                              '150 - AUCHAN MACON',
                              '152 - AUCHAN DOMERAT',
                              '16 - AUCHAN PLAISIR',
                              '160 - AUCHAN MANOSQUE',
                              '163 - AUCHAN SETE',
                              '19 - AUCHAN OLIVET',
                              '2 - AUCHAN RONCQ',
                              '24 - AUCHAN MARTIGUES',
                              '26 - AUCHAN ESCAUDOEUVRES',
                              '27 - AUCHAN AUBAGNE',
                              '29 - AUCHAN BORDEAUX MERIADECK',
                              '45 - AUCHAN VILLARS',
                              '48 - AUCHAN VELIZY VILLACOUBLAY',
                              '49 - AUCHAN CERGY',
                              '54 - AUCHAN BAGNOLET',
                              '58 - AUCHAN MEAUX',
                              '65 - AUCHAN MAUREPAS',
                              '66 - AUCHAN ISSY LES MOULINEAUX',
                              '67 - AUCHAN MARSEILLE PONT DE VIVAUX',
                              '69 - AUCHAN ARRAS',
                              '7 - AUCHAN NOYELLES GODAULT',
                              '70 - AUCHAN TOULOUSE GRAMONT']
    
    liste_top_30_hors_mag_sous_engages = set(top_30_magasins_auchan) - set(magasins_sous_engages_auchan)
    
    if 'AUCHAN' in data.Enseigne.unique():

        
        data['PreconisationVentesUC'] = np.where(data['NomMagasin'].isin(magasins_sous_engages_auchan), data['PreconisationVentesUC'].apply(f_15), data['PreconisationVentesUC'])
        
        data['PreconisationVentesUC'] = np.where(data['NomMagasin'].isin(liste_top_30_hors_mag_sous_engages), data['PreconisationVentesUC'].apply(f_7_5), data['PreconisationVentesUC'])
        
    else:
        pass
    
    return data
    
    
    