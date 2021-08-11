# Increment player and cpu state separately
#0: Make q_table 
#1. List of inputs player action, function deals with one item at a time
#2. Increment player state: 1
#3. Use player action to get random values for cpu_action
#4. create reward for cpu
#5. Incremment cpu state: 2
#6: Update q_table with equation

from sys import flags
import numpy as np
from numpy.core.defchararray import index
from numpy.lib.function_base import disp
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
sepsilon_change = epsilon/(end_epsilon_decaying  - start_epsilon_decaying)
player_actions_num = list(player_actions.replace(["rock","paper","scissors"],["1","2","3"]))
should_update = False

def cpu_reward(player_action,cpu_action):
    #print(f"Player: {player_action},CPU: {cpu_action}")
    if cpu_action == player_action:
        reward = 0
        return reward
    elif (cpu_action == 1 and player_action == 3) or (cpu_action == 2 and player_action == 1) or (cpu_action == 3 and player_action == 2):
        reward = 1
        return reward
    elif (cpu_action == 2 and player_action == 3) or (cpu_action == 3 and player_action == 1) or (cpu_action == 1 and player_action == 2):
        reward = -1
        return reward


#q_table = np.random.randint(1,4,size=(1,2))
#action = 1
#cpu_action = 2
#q_cut = np.random.randint(1,4,size=(1,2))
#q_cut[q_cut < 4] = [action,cpu_action]
#print(q_cut)

#q_rand = np.random.randint(1,4,size=(1,2))
should_update = True
potential_actions = {1:"rock",2:"paper",3:"scissors"}
game_results = {1:"Won",0:"Draw",-1:"Lose"}
cpu_wins = []
cpu_loses = []
player_wins = []
player_loses = []
draws = []
for iterations,action in enumerate(player_actions_num[:4]):
    # Initializer column with random values
    action = int(action)
    if iterations == 0:
        q_table = pd.DataFrame([[action,random.randint(1,3)]],columns=["action","cpu"],index=["init"])
        if np.random.random() > epsilon:
            cpu_action = np.max(q_table["action"])
        else:
        # Explore - t
            cpu_action = np.random.randint(1,3)
        # CPU's turn
        reward_num = cpu_reward(action,cpu_action)
        q_table = q_table.append(pd.DataFrame([[action,cpu_action]],columns=["action","cpu"],index=[iterations]))
        if q_table.index[0] == "init":
            q_table = q_table.drop(index="init")
    else:
        q_table = q_table.append(pd.DataFrame([[action]],columns=["action"],index=[iterations]).join(new_q))
        if np.random.random() > epsilon:
            cpu_action = np.argmax(q_table["action"])
        else:
        ## Explore - t
            cpu_action = np.random.randint(1,3)
        ## CPU's turn
        reward_num = cpu_reward(action,cpu_action)
        #display(q_table)
    


        
    
    if should_update == True:
        #display(q_table)
        new_q = np.array(q_table["action"]) + learning_rate * (reward_num + discount_factor * np.array(q_table["cpu"].max()) - np.array(q_table["action"]))
        new_q = new_q.reshape(len(new_q),1)
        new_q = np.round(new_q)
        new_q = pd.DataFrame(new_q,columns=["cpu"])
        new_q.to_csv(r"C:\Users\user1\Desktop\Bae Rock_paper_scissors\q_learning_rock.csv")
        display(new_q)
    #q_table = q_table["action"]
        #display(new_q.reshape((len(round(new_q)),1)))
        #print("---------------")
        #q_table = q_table.append(q_table)
        #q_table = q_table.append(new_q)
        #display(np.array(q_table))
        #player_result = int(q_table.loc[q_table.index == iterations,["action"]].iloc[0,0])
        #cpu_result = int(q_table.loc[q_table.index == iterations,["cpu"]].iloc[0,0])
        #new_q_result = round(np.mean(new_q))
        #new_q_result = round()
        #play_rew = cpu_reward(cpu_result,player_result)
        #cpu_rew = cpu_reward(player_result,cpu_result)
        #print(f"Round: {iterations} Player action: {potential_actions[player_result]}, Player_result: {player_result}, {game_results[play_rew]}")
        #print(f"Round: {iterations} CPU action: {potential_actions[cpu_result]}, CPU result: {cpu_result}, {game_results[cpu_rew]} ")
        #print(f"Round: {iterations} CPU prediction: {potential_actions[new_q_result]}, Prediction: {new_q_result}")
        #if cpu_rew == 1:
        #    cpu_wins.append(game_results[cpu_rew])
        #if cpu_rew == -1:
        #    cpu_loses.append(game_results[cpu_rew])
        #if play_rew == 1:
        #    player_wins.append(game_results[play_rew])
        #if play_rew == -1:
        #    player_loses.append(game_results[play_rew])
        #if cpu_rew == 0 or play_rew == 0:
        #     draws.append(game_results[cpu_rew])
        #if iterations == len(iterations):
        #    new_q.reshape(1,len())
        #    new_q
#print(f"CPU Wins: {len(cpu_wins)}")
#print(f"CPU Loses: {len(cpu_loses)}")
#print(f"Player Wins: {len(player_wins)}")
#print(f"Player Loses: {len(player_loses)}")
#print(f"Draws: {len(draws)}")