'''
This file implements an agent to participate on the "Santa 2020 - The Candy Cane Contest"
competition from Kaggle (https://www.kaggle.com/c/santa-2020).
'''

import random

def copycat_agent(observation, configuration):
    if observation.lastActions is not None and len(observation.lastActions) > 0:
        return observation.lastActions[0]

    return random.randrange(configuration.banditCount)
