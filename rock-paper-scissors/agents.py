'''
This file implements an agent to participate on the "Rock, Paper, Scissors"
competition from Kaggle (https://www.kaggle.com/c/rock-paper-scissors).
'''

import random

# Given two actions, returns a number indicating which won.
def score(my_action, opponent_action):
    diff = my_action - opponent_action
    
    if diff == 0:
        return 0
    elif diff == -1 or diff == 2:
        return -1
    else:
        return 1

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

# TODO

# --- Final agent --- #

agents = [
    random_agent,
    iterative_agent,
    copy_opponent_agent
]

scores = [0]*len(agents)

last_moves = [0]*len(agents)


# Combines all previous agents, using the more succesful.
def hybrid_agent(observation, configuration):
    print('----')
    
    # Update agents scores.
    if observation.step > 0:
        for i in range(len(last_moves)):
            last_result = score(last_moves[i], observation.lastOpponentAction)
            scores[i] += last_result
            print(str(i) + ' - ' + str(scores[i]))

    # Predict next move for all agents.
    for i in range(len(agents)):
        agent = agents[i]
        last_moves[i] = agent(observation, configuration)
        print(str(i) + ' = ' + str(last_moves[i]))
    
    # Pick an agent.
    if observation.step > 0:
        # Select the most successful agent.
        best_agent_index = scores.index(min(scores))
        print('best agent:' + str(best_agent_index))
        next_move = last_moves[best_agent_index]
    else:
        # The first time just use the random agent.
        next_move = last_moves[0]
    
    # Play.
    return next_move

# --- Local testing --- #

'''
class Observation:
  def __init__(self, step, action):
    self.step = step
    self.lastOpponentAction = action
 
print(hybrid_agent(Observation(0,0), 2))
print(hybrid_agent(Observation(1,0), 2))
print(hybrid_agent(Observation(2,0), 2))
print(hybrid_agent(Observation(3,0), 2))
print(hybrid_agent(Observation(4,0), 2))
print(hybrid_agent(Observation(5,0), 2))
'''
