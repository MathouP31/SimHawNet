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


class Enron(TemporalGraphDataset):
    def __init__(
        self,
        url="http://www.cis.jhu.edu/~parky/Enron/execs.email.linesnum",
        data_path="data_dir",
        user_agent='Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7',
        max_event = None,
        T_max = None,
        *args,
        **kwargs
    ):
        h = self.load(url, data_path, user_agent, T_max)
        super().__init__(h,max_events=max_event,*args, **kwargs) #rajouté max_event

    def load(
        self,
        url,
        data_path,
        user_agent,
        T_max
    ):

        os.makedirs(data_path, exist_ok=True)
        file_path = os.path.join(data_path, url.rsplit("/", 1)[1])

        request = urllib.request.Request(
            url,
            None,
            headers={"User-Agent": user_agent},
        )  #The assembled request
        response = urllib.request.urlopen(request)
        if not os.path.exists(file_path):
            print(f"Downloading Enron dataset to{file_path}")
            with open(file_path, "wb") as f:
                f.write(response.read())
            
        else:
            print("Found enron dataset on the disk")
        data = pd.read_csv(file_path, sep=' ', header=None)
        data.columns = ["t", "src", "dst"]
        data = data.reindex(columns=["src", "dst", "t"])

        # Filter broadcasted emails
        t_val, t_idx, t_freq = np.unique(
            data["t"],
            return_counts=True,
            return_inverse=True,
        )
        mask = t_freq[t_idx] == 1
        data = data.iloc[mask]
        t_order = np.argsort(data["t"])
        if T_max == None:
            T_max = 100

        h = data.values.astype(float)
        h = h[t_order]

        # Map nodes to integers
        node_labels, indices = np.unique(h[:, :2], return_inverse=True)
        h[:, :2] = indices.reshape(len(h), 2)

        # Normalize times to 10
        h[:, 2] = (h[:, 2] - h[0, 2] + sys.float_info.epsilon)
        h[:, 2] = (h[:, 2] / np.max(h[:, 2]))*T_max
       
        h = h[np.argsort(h[:, 2])]
        return [[int(e[0]), int(e[1]), float(e[2])] for e in h]
