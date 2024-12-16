import torch
import numpy as np
from .base import TemporalGraphDataset
import random


class SimulationDataset(TemporalGraphDataset):

    def __init__(self, G,T_max, *args, **kwargs):

        test = []
        for u,v in G.edges():
            if G.nodes[u]["block"] == G.nodes[v]["block"] :
                nb = random.randint(1,20)
            else :
                nb = random.randint(1,3)
            test.extend([[u,v]]*nb)

        random.shuffle(test)
        test = np.array(test)
        concatenated_array = np.c_[test, np.linspace(0,T_max,len(test))]

   
        super().__init__(concatenated_array,
                         *args,
                         **kwargs)