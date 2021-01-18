'''
This file implements an agent to participate on the "Santa 2020 - The Candy Cane Contest"
competition from Kaggle (https://www.kaggle.com/c/santa-2020).
'''

import random

def random_agent(observation, configuration):
    return random.randrange(configuration.banditCount)