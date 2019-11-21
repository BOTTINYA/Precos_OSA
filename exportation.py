#!/usr/bin/env python
# coding: utf-8

# In[ ]: 


"""
Created on 09/09/2019

@author: Yannick Bottino
"""

from google.cloud import bigquery, storage
import google.datalab.bigquery as bq
from time import time
import os



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
    

    
def export_forecast_to_GCS(df, bucket_name, file_destination_name):
    """
    This function converts a pd.DataFrame to csv then exports the csv to the desired GCS bucket, with the desired name.
    It takes for arguments:
    - df : the DataFrame we are willing to export
    - bucket_name : name of the bucket where we are going to export data
    - file_destination_name : the name of the .csv file we are going to export inside the bucket. If the .csv file is inside a file in the bucket. add / to access subfiles
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    
    df.to_csv(file_destination_name)

    blob=bucket.blob('suivi_precos/precos_brutes/'+file_destination_name)
    
    print('\nGCS exportation started ...')
    start_time = time()
    
    blob.upload_from_filename(file_destination_name)
     
    
    print('GCS Exportation Finished. \nTotal exportation time = {:0.2f} min'.format((time()-start_time)/60))

    