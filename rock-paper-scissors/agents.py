'''
This file implements an agent to participate on the "Rock, Paper, Scissors"
competition from Kaggle (https://www.kaggle.com/c/rock-paper-scissors).
'''

import random

# --- Utilities --- #

# Given two actions, returns a number indicating which won.
def get_score(my_action, rival_action):
    diff = my_action - rival_action
    
    if diff == 0:
        return 0
    elif diff == -1 or diff == 2:
        return -1
    else:
        return 1

def winning_action(action):
    return (action + 1)%3

class Observation:
  def __init__(self, step, action):
    self.step = step
    self.lastOpponentAction = action

# --- Stateless agents --- #

# Plays randomly.
def random_agent(observation, configuration):
    return random.randrange(3)

# Returns values 0, 1 and 2 sequentially.
def iterative_agent(observation, configuration):
    return observation.step % 3

# Returns the opponent's last action.
def copy_opponent_agent(observation, configuration):
    if observation.step > 0:
        return observation.lastOpponentAction
    else:
        return 0

# --- Stateful agents --- #

class RememberLastAgent:
    def __init__(self):
        self.rival_penultimate = None
        self.my_penultimate = None
        self.my_last = None
        self.predictions = [None]*9

    def play(self, observation, configuration):
        # Update prediction map.
        if observation.step > 1:
            pred_index = self.my_penultimate*3 + self.rival_penultimate
            self.predictions[pred_index] = observation.lastOpponentAction

        # Update flags and predict next move.
        if observation.step > 0:
            self.rival_penultimate = observation.lastOpponentAction
            self.my_penultimate = self.my_last
    
            pred_index = self.my_last * 3 + observation.lastOpponentAction
            if self.predictions[pred_index] != None:
                self.my_last = winning_action(self.predictions[pred_index])
            else:
                self.my_last = None

        # If can't predict, use the good old random (nothing beats that).
        if self.my_last == None:
            self.my_last = random_agent(observation, configuration)
            
        return self.my_last    

# --- Hybrid agent --- #

# Combines all previous agents, using the more succesful on each case.
class HybridAgent:
    def __init__(self):
        self.agents = [
            random_agent,
            iterative_agent,
            copy_opponent_agent,
            RememberLastAgent()
        ]
        self.scores = [0]*len(self.agents)
        self.last_moves = [0]*len(self.agents)


    def play(self, observation, configuration):
        # Update agents scores.
        if observation.step > 0:
            for i in range(len(self.last_moves)):
                last_result = get_score(self.last_moves[i], observation.lastOpponentAction)
                self.scores[i] += last_result
    
        # Predict next move for all agents.
        for i in range(len(self.agents)):
            agent = self.agents[i]
            self.last_moves[i] = agent(observation, configuration) if callable(agent) else agent.play(observation, configuration)
        
        # Pick an agent.
        if observation.step > 0:
            # Select most successful agent.
            best_agent_index = self.scores.index(max(self.scores))
            next_move = self.last_moves[best_agent_index]
        else:
            # The first time just use the random agent.
            next_move = self.last_moves[0]
        
        # Play.
        return next_move

# --- Final predictor function --- #

my_agent = HybridAgent()
def final_player(observation, configuration):
    return my_agent.play(observation, configuration)

# --- Local testing --- #

'''
print(final_player(Observation(0,0), None))
print(final_player(Observation(1,0), None))
print(final_player(Observation(2,0), None))
print(final_player(Observation(3,0), None))
print(final_player(Observation(4,0), None))
print(final_player(Observation(5,0), None))
print(final_player(Observation(5,0), None))
print(final_player(Observation(5,0), None))
print(final_player(Observation(5,0), None))
print(final_player(Observation(5,0), None))
'''