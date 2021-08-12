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
player_actions = pd.DataFrame(list(importcsv.db.training_data.find()))["Player"]

learning_rate = 0.1
epsilon = 1
discount_factor = 0.95
episodes = 50000
show_every = 1000
start_epsilon_decaying = 0.5
end_epsilon_decaying = episodes//10
epsilon_change = epsilon/(end_epsilon_decaying  - start_epsilon_decaying)
player_actions_num = list(player_actions.replace(["rock","paper","scissors"],["0","1","2"]))

should_update = False

def cpu_reward(player_action,cpu_action):
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


potential_actions = {0:"rock",1:"paper",2:"scissors"}
game_results = {1:"Won",0:"Draw",-1:"Lose"}
cpu_wins = []
cpu_loses = []
player_wins = []
player_loses = []
draws = []
def run_game(should_update = False):
    for iterations,action in enumerate(player_actions_num[:10]):
        # Initializer column with random values
        action = int(action)
        if iterations == 0:
            q_table = pd.DataFrame([[action,random.randint(0,2)]],columns=["action","cpu"],index=[1])
        # This determines the amount of risk the agent will take in it's choices.
        if np.random.random() > epsilon: # If it is large than the epsilon it will follow the q_table more closely making it more predictable.
            # I will decay the epsilon value as I go from 1 - 0.1 so that it goes from extremely unpredictable to reading off of the q_table. The q_table is basically a table that the agent reads off of, learning from the actions already taken. 
            cpu_action = np.argmax(q_table["action"])  # This in the players action(state) it gets the indice/action the index values are [0,1,2] where rock= 0, paper= 1, scissors = 2 
        else: 
        # Explore - t
            cpu_action = np.random.randint(0,2)
        # CPU's turn
        reward_num = cpu_reward(action,cpu_action)
        q_table = q_table.append(pd.DataFrame([[action,cpu_action]],columns=["action","cpu"],index=[iterations]))
        
        if should_update == True:
            #display(q_table)
            new_q = np.array(q_table["action"]) + learning_rate * (reward_num + discount_factor * np.array(q_table["cpu"].max()) - np.array(q_table["action"]))
            new_q = pd.DataFrame(np.round(new_q),columns=["new_cpu"])
            player_result = int(q_table.loc[q_table.index == iterations,["action"]].iloc[0,0])
            cpu_result = int(q_table.loc[q_table.index == iterations,["cpu"]].iloc[0,0])
            #new_q_result = new_q.loc[new_q.index == iterations,["new_cpu"]].iloc[0,0]
            play_rew = cpu_reward(cpu_result,player_result)
            cpu_rew = cpu_reward(player_result,cpu_result)
            print(f"Round: {iterations} Player action: {potential_actions[player_result]}, Player_result: {player_result}, {game_results[play_rew]}")
            print(f"Round: {iterations} CPU action: {potential_actions[cpu_result]}, CPU result: {cpu_result}, {game_results[cpu_rew]} ")
            #print(f"Round: {iterations} CPU prediction: {potential_actions[new_q_result]}, Prediction: {new_q_result}")
            if cpu_rew == 1:
                cpu_wins.append(game_results[cpu_rew])
            if cpu_rew == -1:
                cpu_loses.append(game_results[cpu_rew])
            if play_rew == 1:
                player_wins.append(game_results[play_rew])
            if play_rew == -1:
                player_loses.append(game_results[play_rew])
            if cpu_rew == 0 or play_rew == 0:
                draws.append(game_results[cpu_rew])
            if end_epsilon_decaying >= action >= start_epsilon_decaying:
                epsilon = max(0, epsilon - epsilon_change)

    q_table.to_csv(r"C:\Users\user1\Desktop\Bae Rock_paper_scissors\q_learning_rock.csv")
    #new_q.to_csv(r"C:\Users\user1\Desktop\Bae Rock_paper_scissors\new_q_learning_rock.csv")

    print(f"CPU Wins: {len(cpu_wins)}")
    print(f"CPU Loses: {len(cpu_loses)}")
    #print(f"Player Wins: {len(player_wins)}")
    #print(f"Player Loses: {len(player_loses)}")
    print(f"Draws: {len(draws)}")
    print(f"Total game played: {len(draws) + len(cpu_wins) + len(cpu_loses)}")
