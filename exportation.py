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


def BigQuery_exportation(df, dataset_id, table_name):
    
    print('/nBigQuery exportation started ...')
    
    #BigQuery Table information
    dataset_id = dataset_id

    # Export to Big Query

    client = bigquery.Client(location='EU')


    dataset = client.dataset(dataset_id)

    #Create the table and load the data

    table_ref = dataset.table(table_name)
    
    start_time = time()
    
    load_job = client.load_table_from_dataframe(df, table_ref)
    load_job.result()  # Waits for table load to complete.
    
    print('BigQuery Exportation Finished. /nTotal exportation time = {:0.2f} min'.format((time()-start_time)/60))

