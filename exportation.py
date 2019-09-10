#!/usr/bin/env python
# coding: utf-8

# In[ ]: 


"""
Created on 09/09/2019

@author: Yannick Bottino
"""

from google.cloud import bigquery
import google.datalab.bigquery as bq
from time import time


def BigQuery_exportation(df, bigquery_dataset_name, bigquery_table_name):
    
    print('\nBigQuery exportation started ...')
    start_time = time()

    #Export vers BigQuery
    bigquery_dataset_name = bigquery_dataset_name
    bigquery_table_name = bigquery_table_name

    # Define BigQuery dataset and table
    dataset = bq.Dataset(bigquery_dataset_name)
    table = bq.Table(bigquery_dataset_name + '.' + bigquery_table_name)


    # Create or overwrite the existing table if it exists
    table_schema = bq.Schema.from_data(df)
    table.create(schema = table_schema, overwrite = True)

    # Write the DataFrame to a BigQuery table
    table.insert(df)
    
    print('BigQuery Exportation Finished. \nTotal exportation time = {:0.2f} min'.format((time()-start_time)/60))