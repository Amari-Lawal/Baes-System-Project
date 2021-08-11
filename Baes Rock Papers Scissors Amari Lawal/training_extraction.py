import os
import random

import pandas as pd

from csv_to_db import ImportCSV

# This Program generates at 50000 random rock paper scissors choices of the player at random.

def drop_init(collection):
    # This function is used for debugging for the initialising of the database
    for i in range(1,3):
        collection.delete_many({"Start Columns": i})
# TODO Before using, in the csv_to_db.py file write in your password for the database
importcsv = ImportCSV("Baes_db") # Calls in the database using pymongo. Just input the name of the database
database_training_data_list = list(importcsv.db.training_data.find()) # This initializes the collection table, this will normally return an empty list in order to test if the table exists

if database_training_data_list == []: # Tests if the table exists
    start_df = pd.DataFrame([1,2],columns=["Start Columns"]) # If not it initislises it.
    importcsv.load_data("training_data",init_data=start_df) # and loads data
    database_training_data_list = pd.DataFrame(list(importcsv.db.training_data.find())) # Loads data
elif database_training_data_list != []: # If it exists load it in.
    database_training_data_list = pd.DataFrame(list(importcsv.db.training_data.find())) 

use_database = False
project_directory = r"C:/Users/user1/Desktop/Baes Rock Papers Scissors Amari Lawal/" # Replace with your directory
  
#4.
# Input training data(Person)
print("Training data extraction starting...")
for i in range(0,50000): # Change to 50000 # Creates 50000 data points
    input_actions = ["rock","paper","scissors"] # Sets input data 
    input_action= random.choices(input_actions) # Picks one at random to add to database
    input_df = pd.DataFrame(input_action,columns=["Player"])

    # Output training data(A.I)
    output_actions = ["rock","paper","scissors"] # This was to produce data for neural network but I changed the code
    output_action= random.choices(output_actions) # So it's obselete but I kept it just for my own logical understanfing
    output_df = pd.DataFrame(output_action,columns=["CPU"])

    #next_actions = ["rock","paper","scissors"]
    #next_action = random.choices(next_actions)
    #next_df = pd.DataFrame(next_action,columns=["Next Action"])

    if input_action[0] == output_action[0]:
        result = 0
    elif input_action[0] == "rock" and output_action[0] == "paper":
        result = -1
        next_df = pd.DataFrame(random.choices(output_action))
    elif input_action[0] == "rock" and output_action[0] == "scissors":
        result = 1
    elif input_action[0] == "paper" and output_action[0] == "scissors": # Calculates the rewards for neural network also obslete. However could still potentailly be used.
        result = -1
    elif input_action[0] == "paper" and output_action[0] == "rock":
        result = 1
    elif input_action[0] == "scissors" and output_action[0] == "paper":
        result = 1
    elif input_action[0] == "scissors" and output_action[0] == "rock":
        result = -1

    if use_database == True:
        result_df = pd.DataFrame([result],columns=["Result"]) 
        data = input_df.join(output_df).join(result_df) # creates dataframe of all 50000 points with three columns
        importcsv.load_data("training_data",init_data=data) # Loads it to database
        drop_init(importcsv.db["training_data"]) # drops intialising values from database 
    elif use_database == False: # Loads data to csv file in directory.
        result_df = pd.DataFrame([result],columns=["Result"])
        data = input_df.join(output_df).join(result_df)
        data.to_csv(os.path.join(project_directory,"training_data"),mode='a',header=False) # Adds data to the bottom of the csv after each iteration


print("Training data extraction completed.")

