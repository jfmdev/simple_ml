'''
This file implements an agent to participate on the "Rock, Paper, Scissors"
competition from Kaggle (https://www.kaggle.com/c/rock-paper-scissors).
'''

import random


# --- Utilities --- #

# Given two actions, returns a number indicating which won.
def get_score(action_1, action_2):
    diff = action_1 - action_2
    
    if diff == 0:
        # Tie.
        return 0
    elif diff == -1 or diff == 2:
        # Player 2 won.
        return -1
    else:
        # Player 1 won.
        return 1

# Calculates the move to do to win to a certain action.
def winning_action(action):
    return (action + 1)%3

# Executes an agent (which can be either a function or an object).
def run_agent(agent, observation, configuration):
    return agent(observation, configuration) if callable(agent) else agent.play(observation, configuration)

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
        return random_agent(observation, configuration)

# Beats the opponent's last action.
def reactionary_agent(observation, configuration):
    if observation.step > 0:
        return winning_action(observation.lastOpponentAction)
    else:
        return random_agent(observation, configuration)


# --- Stateful agents --- #

# Implement logic described in https://doi.org/10.1038/srep05830
class BeatHumanAgent:
    def __init__(self):
        self.my_last = None

    def play(self, observation, configuration):
        if observation.step > 0:
            # Check result of last round.
            result = get_score(self.my_last, observation.lastOpponentAction)
            
            if result < 0:
                # If loose, try to beat last opponent action.
                action = winning_action(observation.lastOpponentAction)
            elif result > 0:
                # If won, use last opponent action.
                action = observation.lastOpponentAction
            else:
                # If draw, return random action.
                action = random_agent(observation, configuration)
        else:
            self.my_last = random_agent(observation, configuration)
            
        return self.my_last

# Checks which is the most used action by the rival.
class StatisticalAgent:
    def __init__(self):
        self.action_histogram = {}

    def play(self, observation, configuration):
        if observation.step > 0:
            action = observation.lastOpponentAction
    
            # Update histogram values.
            if action not in self.action_histogram:
                self.action_histogram[action] = 0
                
            self.action_histogram[action] += 1
            
            # Look most used action.
            top_action = None
            top_action_count = None
            for k_action, k_count in self.action_histogram.items():
                if top_action_count is None or k_count > top_action_count:
                    top_action = k_action
                    top_action_count = k_count
                    continue

            # Try to beat most used action.        
            return winning_action(top_action)
        else:
            return random_agent(observation, configuration)

# TODO: Allow to define 'depth'.
class LinearPredictorAgent:
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
            reactionary_agent,
            BeatHumanAgent(),
            StatisticalAgent(),
            LinearPredictorAgent()
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
            self.last_moves[i] = run_agent(agent, observation, configuration)
        
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
# Configuration variables.
rounds = 40
player_1 = reactionary_agent
player_2 = HybridAgent()

# Initialize auxiliary variables.
score = 0
last_action_1 = None
last_action_2 = None

# Run agents.
for step in range(rounds):
    # Play round.
    action_1 = run_agent(player_1, Observation(step, last_action_2), None)
    action_2 = run_agent(player_2, Observation(step, last_action_1), None)
    result = get_score(action_1, action_2)
    
    # Update variables.
    score += result
    last_action_1 = action_1
    last_action_2 = action_2

    # Print result.
    if result == 0:
        print(f'Tie! ({score})')
    elif result > 0:
        print(f'Agent 1 won! ({score})')
    else:
        print(f'Agent 2 won! ({score})')
'''