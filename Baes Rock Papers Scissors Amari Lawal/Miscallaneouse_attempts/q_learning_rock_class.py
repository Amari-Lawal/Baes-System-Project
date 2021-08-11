# Increment player and cpu state separately
#0: Make q_table 
#1. List of inputs player action, function deals with one item at a time
#2. Increment player state: 1
#3. Use player action to get random values for cpu_action
#4. create reward for cpu
#5. Incremment cpu state: 2
#6: Update q_table with equation

import numpy as np
from numpy.core.defchararray import index
from numpy.lib.function_base import disp, extract
import pandas as pd
import random
from IPython.core.display import display

from csv_to_db import ImportCSV

importcsv = ImportCSV('Baes_db')



class RockPaperScissors:
    def __init__(self,filepath=None,training_data = None,importcsv=None) -> None:
        if importcsv != None:
            player_actions = pd.DataFrame(list(importcsv.db.training_data.find()))["Player"]
        if training_data != None:
            player_actions = pd.read_csv(filepath)
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.show_every = 1000
        self.player_actions_num = list(player_actions.replace(["rock","paper","scissors"],["0","1","2"]))
        self.potential_actions = {0:"rock",1:"paper",2:"scissors"}
        self.game_results = {1:"Won",0:"Draw",-1:"Lose"}
        self.cpu_wins = []
        self.cpu_loses = []
        self.player_wins = []
        self.player_loses = []
        self.draws = []
    def cpu_reward(self,player_action,cpu_action):
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
        epsilon = 1
        start_epsilon_decaying = 0.5
        end_epsilon_decaying = len(self.player_actions_num)//10
        epsilon_change = epsilon/(end_epsilon_decaying  - start_epsilon_decaying)
        for iterations,action in enumerate(self.player_actions_num):    
            # Initializer column with random values
            action = int(action)
            if iterations == 0:
                q_table = pd.DataFrame([[action,random.randint(0,2)]],columns=["action","cpu"],index=[1])
                if np.random.random() > epsilon: # If it is large than the epsilon it will follow the q_table more closely making it more predictable.
                # I will decay the epsilon value as I go from 1 - 0.1 so that it goes from extremely unpredictable to reading off of the q_table. The q_table is basically a table that the agent reads off of, learning from the actions already taken. 
                    cpu_action = np.argmax(q_table["action"])  # This in the players action(state) it gets the indice/action the index values are [0,1,2] where rock= 0, paper= 1, scissors = 2 
                else: 
                # Explore - t
                    cpu_action = np.random.randint(0,2)
            # CPU's turn
            elif iterations != 0:
                new_cut = pd.DataFrame([[action,new_q]],columns=["action","cpu"])
                #q_table.append(new_cut)
                # This determines the amount of risk the agent will take in it's choices.
                if np.random.random() > epsilon: # If it is large than the epsilon it will follow the q_table more closely making it more predictable.
                    # I will decay the epsilon value as I go from 1 - 0.1 so that it goes from extremely unpredictable to reading off of the q_table. The q_table is basically a table that the agent reads off of, learning from the actions already taken. 
                    cpu_action = np.argmax(new_cut["cpu"])  # This in the players action(state) it gets the indice/action the index values are [0,1,2] where rock= 0, paper= 1, scissors = 2 
                else: 
                # Explore - t
                    cpu_action = np.random.randint(0,2)
            # CPU's turn
            reward_num = self.cpu_reward(action,cpu_action)
            q_table = q_table.append(pd.DataFrame([[action,cpu_action]],columns=["action","cpu"],index=[iterations]))
            
            if should_update == True:
                #new_q = np.array(q_table["action"]) + self.learning_rate * (reward_num + self.discount_factor * np.array(q_table["cpu"].max()) - np.array(q_table["action"]))
                new_q = reward_num +  self.discount_factor * (np.array(q_table["action"].max()))
                player_result = int(q_table.loc[q_table.index == iterations,["action"]].iloc[0,0])
                play_rew = self.cpu_reward(new_q,player_result)
                cpu_rew = self.cpu_reward(player_result,new_q)
                
                if cpu_rew == 1:
                    self.cpu_wins.append(self.game_results[cpu_rew])
                if cpu_rew == -1:
                    self.cpu_loses.append(self.game_results[cpu_rew])
                if play_rew == 1:
                    self.player_wins.append(self.game_results[play_rew])
                if play_rew == -1:
                    self.player_loses.append(self.game_results[play_rew])
                if cpu_rew == 0 or play_rew == 0:
                    self.draws.append(self.game_results[cpu_rew])
                if end_epsilon_decaying >= action >= start_epsilon_decaying:
                    epsilon = max(0, epsilon - epsilon_change)
                
                if verbose_all == 1:
                    print(f"Round: {iterations} | Player action: {self.potential_actions[player_result]} | CPU action:  {self.potential_actions[new_q]}, {self.game_results[cpu_rew]}! ")
                    #print(f"Round: {iterations} Player action: {self.potential_actions[player_result]}, Player_result: {player_result}, {self.game_results[play_rew]}")
                    #print(f"Round: {iterations} CPU action: {self.potential_actions[cpu_result]}, CPU result: {cpu_result}, {self.game_results[cpu_rew]} ")
                elif verbose_all == 0:
                    if iterations % self.show_every == 0:
                        print(f"Round: {iterations} | Player action: {self.potential_actions[player_result]} | CPU action:  {self.potential_actions[new_q]} | {self.game_results[cpu_rew]}! ")
                        if iterations% (self.show_every * 10) == 0:
                            print(f"CPU Wins: {len(self.cpu_wins)}")
                            print(f"CPU Loses: {len(self.cpu_loses)}")
                            print(f"Draws: {len(self.draws)}")
                            print(f"Total games played so far: {len(self.draws) + len(self.cpu_wins) + len(self.cpu_loses)}")
                        #print(f"Round: {iterations} Player action: {self.potential_actions[player_result]}, Player_result: {player_result}, {self.game_results[play_rew]}")
                        #print(f"Round: {iterations} CPU action: {self.potential_actions[cpu_result]}, CPU result: {cpu_result}, {self.game_results[cpu_rew]} ")
                        #print(f"Round: {iterations} CPU prediction: {potential_actions[new_q_result]}, Prediction: {new_q_result}")

        #q_table.to_csv(r"C:\Users\user1\Desktop\Bae Rock_paper_scissors\q_learning_rock.csv")
        #new_q.to_csv(r"C:\Users\user1\Desktop\Bae Rock_paper_scissors\new_q_learning_rock.csv")

        print(f"Overall CPU Wins: {len(self.cpu_wins)}")
        print(f"Overall CPU Loses: {len(self.cpu_loses)}")
        print(f"Overall Draws: {len(self.draws)}")
        total_games = len(self.draws) + len(self.cpu_wins) + len(self.cpu_loses)
        print(f"Overall Total games played: {total_games}")
        overall_df = pd.DataFrame([len(self.cpu_wins),len(self.cpu_loses),len(self.draws),total_games],columns=["CPU Wins","CPU Loses","Draws","Total Games Played"],index=[f"Episode"])
        overall_df.to_csv(r'C:\Users\user1\Desktop\Bae Rock_paper_scissors\data\overall_results.csv')

if __name__ == "__main__":
    rockpaperscissors= RockPaperScissors(importcsv=importcsv)
    rockpaperscissors.run_game(verbose_all=0,should_update=True)
    #pass
    #should_update = False