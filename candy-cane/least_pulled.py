'''
This file implements an agent to participate on the "Santa 2020 - The Candy Cane Contest"
competition from Kaggle (https://www.kaggle.com/c/santa-2020).
'''

class LeastPulledAgent:
  def __init__(self):
    self.bandit_pulls = None

  def play(self, observation, configuration):
    # Base initialization.
    if self.bandit_pulls is None:
        self.bandit_pulls = [0] * configuration.banditCount
      
    # Update actions counts.
    for action in observation.lastActions:
      self.bandit_pulls[action] += 1

    # Search for less used bandit.
    less_used_index = self.bandit_pulls.index(min(self.bandit_pulls))

    return less_used_index


my_agent = LeastPulledAgent()
def agent_play(observation, configuration):
  return my_agent.play(observation, configuration)
