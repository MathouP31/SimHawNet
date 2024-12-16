import torch
import numpy as np
from .base import TemporalGraphDataset
import random


class SimulationDatasetSimple(TemporalGraphDataset):

    def __init__(self, G,T_max, *args, **kwargs):
        
        edge_list = (np.array(G.edges)).astype(int)
        taille = len(edge_list)
        edge_list = np.concatenate((edge_list, edge_list), axis=0)

        np.random.shuffle(edge_list)

        n = len(edge_list)-1
        min_value = 0
        max_value = T_max 
        gaps = np.random.uniform(low=0.1, high=1.0, size=(n-1,))
        time = np.cumsum(gaps)

        time = time * (max_value - min_value) / time[-1] + min_value

        time = np.concatenate([[min_value], time, [max_value]])
        time = np.expand_dims(time, axis=1)
        concatenated_array = np.concatenate([edge_list,time], axis=1, dtype=object)

        super().__init__(concatenated_array,
                         *args,
                         **kwargs)