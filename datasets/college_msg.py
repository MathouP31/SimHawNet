import os
import sys
import urllib

import numpy as np
import pandas as pd

from .base import TemporalGraphDataset


class CollegeMsg(TemporalGraphDataset):

    def __init__(self,
                 url="https://snap.stanford.edu/data/CollegeMsg.txt.gz",
                 data_path="data_dir",
                 max_event = None,
                 *args,
                 **kwargs):
        edge_list = self.load(url, data_path)

        super().__init__(edge_list,max_events=max_event, *args, **kwargs)

    def load(self, url, data_path):
        os.makedirs(data_path, exist_ok=True)
        file_path = os.path.join(data_path, url.rsplit("/", 1)[1])

        if not os.path.exists(file_path):
            print(f"Downloading the CollegeMsg dataset to {file_path}")
            urllib.request.urlretrieve(url, file_path)
            print("Done.")

        data = pd.read_csv(file_path, sep=' ', header=None)
        self.mathilde = data
        data.columns = ["u", "v", "t"]

        # Remove the simulataneous events
        t_val, t_idx, t_freq = np.unique(
            data["t"],
            return_counts=True,
            return_inverse=True,
        )
        mask = t_freq[t_idx] == 1
        data = data.iloc[mask]

        t_order = np.argsort(data["t"])

        # Map nodes to integers
        self.node_labels, indices = np.unique(
            data[["u", "v"]],
            return_inverse=True,
        )
        h = data.values.astype(float)
        h[:, :2] = indices.reshape(len(h), 2)
        # Normalize times to 1
        h[:, 2] = (h[:, 2] - h[0, 2] + sys.float_info.epsilon)
        h[:, 2] = h[:, 2] / np.max(h[:, 2]) *10

        # Remove double events

        h = h[t_order]
        return [[int(e[0]), int(e[1]), float(e[2])] for e in h]
