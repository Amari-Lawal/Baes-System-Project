#!/usr/bin/env python
import base64
import json
import sys

import numpy as np
import pandas as pd
import pymongo
from IPython.core.display import display


class ImportCSV:
  """
  This initaializes a connection with a database that can be called in other files to use the database, use it's collections and load data into database.
  """
  def __init__(self,database) -> None:
    # Takes in parameters
    #self.filepath = filepath
    #collection_name  =  collection
    password = "<input password here>"
    # Initialises database
    client = pymongo.MongoClient('localhost', 27017,username='admin',password=password,authSource='admin',authMechanism='SCRAM-SHA-256') # Creates pymongo client to connect to database
    self.db = client[database] # Takes initialised database and stores it as variable object  
    
    
    

  
  def load_data(self,collection_name,query=None,init_data=None,filepath = None):
    # Initialises collection 
    db_cm = self.db[collection_name] # Stores collection/table in database
    # Initialises csv data
    def load_n_insert(data):
      data_json = json.loads(data.to_json(orient='records')) # Loads data into database as json
      db_cm.insert_many(data_json) # Inserts data into database

    if query == None and filepath == None: # This basically determines that it is not going to save it as a csv and if raw data is initialized into the class then it will just load it to database
      data = init_data 
      load_n_insert(data)
    
    elif init_data == None: # loads database as csv, I used this in another project.
      data = pd.read_csv(filepath)
      data = pd.DataFrame(data[query],columns=[query])
      
      for column in data.columns: # This is for debugging getting rid of odd columns that would randomly appear
        if "Unnamed" in column or "level" in column:
          data = data.drop([column],axis=1)
          data.to_csv(filepath)
      load_n_insert(data)
     

      


  