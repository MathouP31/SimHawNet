import os
import sys
import urllib
from ipaddress import ip_address

import numpy as np
import pandas as pd
import torch
from genericpath import exists
from torch.utils.data import Dataset, Subset

from datasets.base import TemporalGraphDataset


class Highschool(TemporalGraphDataset):
    def __init__(
        self,
        data_path=None,
        *args,
        **kwargs
    ):
        h = self.load(nb_nodes = 150)
        super().__init__(h, *args, **kwargs) 

    def load(
        self,
        nb_nodes
    ):

        data = pd.read_csv("data_dir\High-School_data_2013.csv", sep=' ', header=None)
        colonne_1 = data.iloc[:, 1]  
        colonne_2 = data.iloc[:, 2]  

        df_filtre = data[(colonne_1 <= 150) & (colonne_2 <= nb_nodes)] # NB de nodes voulu = 150
        data = df_filtre.iloc[:, :3]
        data.columns = ["t", "src", "dst"]
        data = data.reindex(columns=["src", "dst", "t"])

        t_val, t_idx, t_freq = np.unique(data["t"], return_counts=True, return_inverse=True)

        mask = np.zeros(len(data), dtype=bool)
        for i, val in enumerate(t_val):
            if t_freq[i] > 1:
                first_occurrence_idx = np.where(t_idx == i)[0][0]  
                mask[first_occurrence_idx] = True
            else:
                mask[t_idx == i] = True

        data_filtered = data.iloc[mask]
        h = data_filtered.values.astype(float)

        # Normalize times to 10
        h[:, 2] = (h[:, 2] - h[0, 2] + sys.float_info.epsilon)
        h[:, 2] = (h[:, 2] / np.max(h[:, 2]))*10

        h = h[np.argsort(h[:, 2])]

        return [[int(e[0]), int(e[1]), float(e[2])] for e in h]
