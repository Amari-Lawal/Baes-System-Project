# Plan
# Increment player and cpu state separately
#0: Make q_table 
#1. List of inputs player action, function deals with one item at a time
#2. Increment player state: 1
#3. Use player action to get random values for cpu_action
#4. create reward for cpu
#5. Incremment cpu state: 2
#6: Update q_table with equation

import os
import random

import numpy as np
import pandas as pd
from IPython.core.display import display
# TODO f using database
#from csv_to_db import ImportCSV
#importcsv = ImportCSV('Baes_db')



class RockPaperScissors:
    """
    Notes
    -----
    Rock Paper Scissors game that implements Q-Learning Reinforcement Learning between two CPU's
    
    Parameters
    ----------
    training_file(str): csv file full of 50000 generated games.
    filepath(str): Input in filepath of training_data
    importcsv(pymongo_obj): Loads in the database 

    ALERT
    ------
    ALERT: There may be a bug when running about no operation on NoneType or float.
    It's a bug with the np.argmax function that returns an increment that is out of range of the 3 possible actions in the q_table so returns None
    Just rerun code.  

            
    
    
    
    """
    def __init__(self,training_file,project_filepath=None,importcsv=None) -> None:
        # TODO Learning rate, discount factor, epsilon are all hyperparameters I can change
        self.project_filepath = project_filepath # Initializes filepath to be used to save data
        if importcsv != None: # Loads data from database
            player_actions = pd.DataFrame(list(importcsv.db.training_data.find()))["Player"] #
        if training_file != None:
            player_df = pd.read_csv(os.path.join(self.project_filepath,str(training_file)))
            player_actions = player_df["Player"]
        self.learning_rate = 0.1 # Sets learning rate, rate in which changes are made in the learning
        self.discount_factor = 0.95 # The extent by percentage the q_table should be changed.
        self.show_every = 1000 # Allows me to print every 1000
        self.player_actions_num = list(player_actions.replace(["rock","paper","scissors"],["0","1","2"])) # Replaces options as encodings to be analysed
        random.shuffle(self.player_actions_num) # shuffles data to avoid same seed.
        self.potential_actions = {0:"rock",1:"paper",2:"scissors"} # dictionary to map encodings to option
        self.game_results = {1:"Won",0:"Draw",-1:"Lose"} # encodings for loses so that I can  print whether the lost, won or draw using reward at the end of each iteration
        self.cpu_wins = []
        self.cpu_loses = []
        self.player_wins = []
        self.player_loses = []
        self.draws = []
    def cpu_reward(self,player_action,cpu_action):
        # Boolean that allocates the rewarding system according to the rules of the game.
        #print(f"Player: {player_action},CPU: {cpu_action}")
        if cpu_action == player_action:
            reward = 0
            return reward
        elif (cpu_action == 0 and player_action == 2) or (cpu_action == 1 and player_action == 0) or (cpu_action == 2 and player_action == 1):
            reward = 1
            return reward
        elif (cpu_action == 1 and player_action == 2) or (cpu_action == 2 and player_action == 0) or (cpu_action == 0 and player_action == 1):
            reward = -1
            return reward



    def run_game(self,verbose_all=0,should_update = False):
        epsilon = 1 # Epsilon determines the riskiness of the agents decisions. It decays over a period of time so it uses the q_table as a "dictionary" to make its next moves.
        start_epsilon_decaying = 0.5 # Where to stop the epsilon decay
        end_epsilon_decaying = len(self.player_actions_num)//10 # The end of the dacay
        epsilon_change = epsilon/(end_epsilon_decaying  - start_epsilon_decaying) # The rate the epsilon will change calculating the gradient.
        for iterations,action in enumerate(self.player_actions_num):   
            # Initializer column with random values
            action = int(action) # sets action as integer
            if iterations == 0:
                q_table = pd.DataFrame([[action,random.randint(0,2)]],columns=["action","cpu"],index=[1]) # Intializes first part of q_table to determine cpu move at random
            
            elif iterations != 0:
                new_cut = pd.DataFrame([[action,cpu_result]],columns=["action","cpu"]) # This uses the previous value computed to then be used for the next so it can recursively learn.
                q_table = q_table.append(new_cut) # Adds it to the q table
            # This determines the amount of risk the agent will take in it's choices.
            if np.random.random() > epsilon and iterations != 0: # If it is large than the epsilon it will follow the q_table more closely making it more predictable.
                #display(q_table)
                cpu_action = np.argmax(q_table["action"])  # This in the players action(state) it gets the indice/action the index values are [0,1,2] where rock= 0, paper= 1, scissors = 2 
                #print(cpu_action)
            else:  # IF epsilon is greater then be risky with the choice
            # Explore - t
                cpu_action = np.random.randint(0,2) # Sets cpu_action at random

            reward_num = self.cpu_reward(action,cpu_action) # Provides the cpu's reward to be used in the equation
            q_table = q_table.append(pd.DataFrame([[action,cpu_action]],columns=["action","cpu"],index=[iterations])) # Adds the believed optimal cpu_action to be used in the equation 
            
            if should_update == True:
                new_q = np.array(q_table["action"]) + self.learning_rate * (reward_num + self.discount_factor * np.array(q_table["cpu"].max()) - np.array(q_table["action"]))
                # Algorithm calculates tha maximum change from the first state. The players move to the second state the cpu's move. Computing the next move.
                new_q = pd.DataFrame(np.round(new_q),columns=["cpu"]) # The values end up as float values but they are very close to the intended value so round the value to indended value.
                player_result = int(q_table.loc[q_table.index == iterations,["action"]].iloc[0,0]) # The Q_table is like a log of all possible games played. So we need to get the value that was done during this specific iteration.
                cpu_result = int(new_q.loc[new_q.index == iterations,["cpu"]].iloc[0,0]) # Same for the new  q value

                play_rew = self.cpu_reward(cpu_result,player_result) # Gets rewards to determine whether we won lost or drew
                cpu_rew = self.cpu_reward(player_result,cpu_result) # Same here
                
                if cpu_rew == 1:
                    self.cpu_wins.append(self.game_results[cpu_rew]) 
                if cpu_rew == -1:
                    self.cpu_loses.append(self.game_results[cpu_rew])
                if play_rew == 1:
                    self.player_wins.append(self.game_results[play_rew]) # This maps the reward with the word, win = 1,draw = 0 ,lose = -1, so I can coint how many overall wins I had.
                if play_rew == -1:
                    self.player_loses.append(self.game_results[play_rew])
                if cpu_rew == 0 or play_rew == 0:
                    self.draws.append(self.game_results[cpu_rew])
                if end_epsilon_decaying >= action >= start_epsilon_decaying: # This enacts the epsilon decay.
                    epsilon = max(0, epsilon - epsilon_change)
                
                if verbose_all == 1:
                    print(f"Round: {iterations}  | Q CPU action:  {self.potential_actions[cpu_result]} | CPU2 action: {self.potential_actions[player_result]} | {self.game_results[cpu_rew]}! ")
                elif verbose_all == 0:
                    if iterations % self.show_every == 0: # Every 1000 games print result.
                        print(f"Round: {iterations}  | Q CPU action:  {self.potential_actions[cpu_result]} | CPU2 action: {self.potential_actions[player_result]} | {self.game_results[cpu_rew]}! ")
                        if iterations% (self.show_every * 10) == 0: # Every 10000 games print result
                            print(f"Q CPU Wins: {len(self.cpu_wins)}")
                            print(f"Q CPU Loses: {len(self.cpu_loses)}")
                            print(f"Draws: {len(self.draws)}")
                            total_games = len(self.draws) + len(self.cpu_wins) + len(self.cpu_loses)
                            print(f"Total games played so far: {total_games}")
                            increment_df = pd.DataFrame([[len(self.cpu_wins),len(self.cpu_loses),len(self.draws),total_games]],columns=["Q CPU Wins","Q CPU Loses","Draws","Total Games Played"])
                            increment_df.to_csv(os.path.join(self.project_filepath,"data/increment_results.csv"),mode='a',header=False) # saves it to csv
         
                    
        # Only saves when we win more games so that I can have the best outcome for future moves also the csv would be large due to storing 50000 each run.                        
        if len(self.cpu_wins) > len(self.cpu_loses):                   
            q_table.to_csv(os.path.join(self.project_filepath,"data\q_table.csv"),mode='a',header=False) # Saves q_table to csv
        # Prints overall results
        print(f"Overall Q CPU Wins: {len(self.cpu_wins)}")
        print(f"Overall CPU Loses: {len(self.cpu_loses)}")
        print(f"Overall Draws: {len(self.draws)}")
        ov_total_games = len(self.draws) + len(self.cpu_wins) + len(self.cpu_loses)
        print(f"Overall Total games played: {ov_total_games}")
        # Saves overall score
        overall_df = pd.DataFrame([[len(self.cpu_wins),len(self.cpu_loses),len(self.draws),ov_total_games]],columns=["Q CPU Wins","Q CPU Loses","Draws","Total Games Played"])
        overall_df.to_csv(os.path.join(self.project_filepath,'data\overall_results.csv'),mode='a',header=False)

if __name__ == "__main__":
    # Make sure the dividers are the right way
    project_filepath = r"C:/Users/user1/Desktop/Baes Rock Papers Scissors Amari Lawal/" # ! Change this to your directory with the slash at the end
    # Runs using data from database
    #rockpaperscissors= RockPaperScissors(importcsv=importcsv)
    #rockpaperscissors.run_game(verbose_all=0,should_update=True)
    # Runs data from csv file.
    rockpaperscissors= RockPaperScissors("training_data.csv",project_filepath)
    rockpaperscissors.run_game(verbose_all=0,should_update=True)
