import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import PaddedFeatureAccessor, PaddedTimeAccessor, dict_to_features

from .history import NetworkHistory

PREPROCESSED_PATH = "tmp"
VERBOSE = True

# random.seed(31)


class TemporalGraphDataset(Dataset):

    def __init__(
        self,
        edge_list,
        features={},
        max_events=None, 
        np_ratio=1,
    ):
        # print("Building the dataset...")
        self.np_ratio = np_ratio
        if (max_events is None):
            max_events = int(len(edge_list))
        self.features = features
        features = dict_to_features(features)
        # Truncate the data
        edge_list = edge_list[:max_events]
        self.edge_list = edge_list

        self.n_features = len(features)
        
        self.src = [e[0] for e in edge_list]
        self.dst = [e[1] for e in edge_list]
        self.t = [e[2] for e in edge_list]

        self.x_pad = PaddedFeatureAccessor(edge_list, features)
        self.history = NetworkHistory(edge_list=edge_list)
        self.t_pad = PaddedTimeAccessor(edge_list)

        self.n_nodes = int(max(self.src + self.dst) + 1)
        self.nodes = torch.arange(self.n_nodes)

        self.n_edges = self.n_nodes * (self.n_nodes - 1) / 2

    def __sample_negatives(self, src, dst):
        # pairs_mask = np.array(self.edge_list)[:,0] == src
        # dst = np.array(self.edge_list)[:,1][pairs_mask]
        # dst_unique = np.unique(dst)
        # mask = (self.nodes != src) & (self.nodes != dst) & (self.nodes !=np.all(dst_unique) ) ==> le mettre en précalculation
        mask = (self.nodes != src) & (self.nodes != dst)
     

       
# pre computation !
        return [
            int(e) for e in random.choices(self.nodes[mask], k=self.np_ratio)
        ]
    # faire negative sampling here
    # 1 for 1 ? or more negative edge ? 

    def __getitem__(self, idx):
        src = self.src[idx]
        dst = self.dst[idx]
        dst_neg = self.__sample_negatives(src, dst)
        batch = {
            "t": self.t[idx],
            "src": src,
            "dst": dst,
            "t_prev": self.t[0] if (idx == 0) else self.t[idx - 1],
            "dst_neg": self.__sample_negatives(src, dst),
            "x_pad_pos": self.x_pad[src, dst],
            "t_pad_pos": self.t_pad[src, dst],
            "x_pad_neg": [self.x_pad[src, d_] for d_ in dst_neg],
            "t_pad_neg": [self.t_pad[src, d_] for d_ in dst_neg],
        }

        return batch

    def __len__(self):
        return len(self.src)


    @staticmethod
    def collate_fn(batch):

        src = []
        dst = []
        t = []
        t_prev = []
        x_pad = []
        t_pad = []

        y = []

        for sample in batch:
            # Positive samples
            src.append(sample["src"])
            dst.append(sample["dst"])
            t.append(sample["t"])
            t_prev.append(sample["t_prev"])
            x_pad.append(sample["x_pad_pos"])
            t_pad.append(sample["t_pad_pos"])
            y.append(1)

            # Negative samples
            for dn_, xn_, tn_ in zip(
                    sample["dst_neg"],
                    sample["x_pad_neg"],
                    sample["t_pad_neg"],
            ):
                src.append(sample["src"])
                dst.append(dn_)
                t.append(sample["t"])
                t_prev.append(sample["t_prev"])
                x_pad.append(xn_)
                t_pad.append(tn_)
                y.append(0)

        y = torch.tensor(y).int()
        src = torch.tensor(src).long()
        dst = torch.tensor(dst).long()
        t = torch.tensor(t).float()
        t_prev = torch.tensor(t_prev).float()
        x_pad = torch.tensor(np.array(x_pad)).float()
        t_pad = torch.tensor(np.array(t_pad)).float()
        return src, dst, t, t_prev, x_pad, t_pad, y


def get_volume_dict(dataset):

    def get_key(k, n):
        return min(k, n), max(k, n)

    volume_dict = {
        get_key(k, n): len(th)
        for k, nh in dataset.x_pad.h.h_dict.items()
        for n, th in nh.nh_dict.items()
    }
    return volume_dict


def get_burstiness_dict(dataset):
    return {k: burstiness(v) for k, v in dataset.t_pad.t_dict.items()}


def plot_burstiness_profiles(dataset, ax, bmin=-1, bmax=1, *args, **kwargs):

    b_dict = get_burstiness_dict(dataset)
    bs = np.array(list(b_dict.values()))
    mask = (bs > bmin) & (bs < bmax)
    _ = ax.hist(bs[mask], *args, **kwargs)


def burstiness(samples):
    samples = np.array(samples)
    samples = samples[np.isfinite(samples)]
    if len(samples) < 2:
        return -2
    dt = np.diff(samples)

    cv = np.std(dt) / np.mean(dt)
    b = (cv - 1) / (cv + 1)

    return b
