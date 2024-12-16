import os
import sys
import urllib
from ipaddress import ip_address

import numpy as np
import pandas as pd
import torch
from datasets.base import TemporalGraphDataset
from genericpath import exists
from torch.utils.data import Dataset, Subset


class SimulDataset(TemporalGraphDataset):

    def __init__(
            self,
            data_path="data_dir",
            user_agent='Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7',
            *args,
            **kwargs):
        file_path = os.path.join(data_path, "simulation.txt")
        df = pd.read_csv(file_path, sep=' ')
        h = [[e[0], e[1], e[2]] for e in df.values]

        super().__init__(h, *args, **kwargs)
