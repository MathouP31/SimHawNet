from collections import defaultdict

from copy import deepcopy
from typing import List

import numpy as np
from tqdm import tqdm, trange

from .history import NetworkHistory


def create_feature(name, *args, **kwargs):
    if name == "common_neighbor_d":
        return CommonNeighborsDecayed(*args, **kwargs)
    if name == "volume_d":
        return VolumeDecayed(*args, **kwargs)
    if name == "pref_att_d":
        return PreferentialAttachmentDecay(*args, **kwargs)
    if name == "common_neighbor":
        return CommonNeighbors(*args, **kwargs)
    if name == "volume":
        return Volume(*args, **kwargs)
    if name == "random":
        return RandomFeature(*args, **kwargs)
    if name == "random2":
        return RandomFeature(*args, **kwargs)
    if name == "random3":
        return RandomFeature(*args, **kwargs)
    if name == "preferential_attachment":
        return PreferentialAttachment(*args, **kwargs)
    #### SUITE RAJOUTEE
    if name == "degree_source_decayed":                      
        return DegreeSourceDecayed(*args, **kwargs)

def dict_to_features(f_dict):
    features = []
    for k, v in f_dict.items():
        if v is None:
            features.append(create_feature(k))
        else:
            features.append(create_feature(k, **v))

    return features


class PreferentialAttachmentDecay:
    def __init__(self, decay=0.1):
        self.deg = DegreeDecayed(decay)

    def __call__(self, h: NetworkHistory, u, v, t):

        deg_u = self.deg(h, u,v, t)
        deg_v = self.deg(h, v,u, t)
        return deg_u * deg_v


class VolumeDecayed:
    def __init__(self, decay=0.1):
        self.decay = decay

    def __call__(self, h: NetworkHistory, u, v, t):

        h = h[u, v]
        if (h is None):
            return 0
        h = h(t)
        times = np.array(h)
        weight = 0
        if (len(times) > 0):
            weight += np.sum(np.exp(-self.decay * (t - times)))
       
        return float(weight)
    


class CommonNeighborsDecayed:
    def __init__(self, decay=0.1):
        self.decay = decay
        pass

    def __call__(self, h: NetworkHistory, u, v, t):
   

        hut = h[u](t)
        hvt = h[v](t)

        weight = 0
        for nu, tnu in hut.items():
            for nv, tnv in hvt.items():
                if (nu == nv):
                    
                    last_t = max(max(tnu), max(tnv))
                    weight += np.sum(np.exp(-self.decay * (t - last_t)))
        return float(weight)


class DegreeDecayed:
    def __init__(self, decay=0.10):
        self.decay = decay

    def __call__(self, h: NetworkHistory, u,v, t):
        hut = h[u](t)
        weight = 0
        for nu, tnu in hut.items():

            if (len(tnu) == 0):
                continue
                
            else:
                weight += np.sum(np.exp(-self.decay * (t - max(tnu))))
        
        return float(weight)


class RandomFeature:
    def __init__(self):
        pass

    def __call__(self, h, u, v, t):
        return np.random.rand()


class Degree:
    def __init__(self):
        pass

    def __call__(self, h: NetworkHistory, u,v, t):

        hut = h[u](t)

        return len(hut.keys())


class PreferentialAttachment:
    def __init__(self):
        pass

    def __call__(self, h, u, v, t):

        hu = h[u]
        hv = h[v]
        hut = hu(t)
        hvt = hv(t)

        return len(hut.keys()) * len(hvt.keys())

class CommonNeighbors:
    def __init__(self):
        pass

    def __call__(self, h: NetworkHistory, u, v, t):
     
        hu = h[u]
        hv = h[v]
        hut = hu(t)
        hvt = hv(t)
        u_neighbors = set(hut.keys())
        v_neighbors = set(hvt.keys())
        cn = u_neighbors.intersection(v_neighbors)
        return len(cn)


class Volume:
    def __init__(self):
        pass

    def __call__(self, h: NetworkHistory, u, v, t):
    
        return len(h[u, v](t))


class DegreeSourceDecayed:
    def __init__(self, decay=0.10):
        self.deg = DegreeDecayed(decay)

    def __call__(self, h: NetworkHistory, u, v, t):
   
        return self.deg(h, u, t)


class DegreeDestDecayed:
    def __init__(self, decay=0.10):
        self.deg = DegreeDecayed(decay)

    def __call__(self, h: NetworkHistory, u, v, t):
    
        return self.deg(h, v, t)


class NetworkFeatureValue:
    """ NetworkFeatureValue object.
    if nfv is a NetworkFeatureValue, then nfv[u,v] accesses the feature values
    vector for node u,v
    """
    def __init__(self, feature):
        self.feature = feature

    def __getitem__(self, idx):
        u, v = idx
   
        return self.feature(u, v)


class FeatureAggregator:
    """
    Signatures if fa is a FeatureAggregator:
    -    fa[u,v] is a function taking a NetworkHistory and a float as input,
        and returning a vector of features.
    -    fa(h, t) returns a NetworkFeatureValue object, allowing to access the
    feature values for different edges as
                    nfv[u,v]


    """
    def __init__(self, features):
        self.features = features
        self.feature_names = [f.__class__.__name__ for f in self.features]

    def f_acc_factory(self, h, t):
        def f(u, v):
            return [feature(h, u, v, t) for feature in self.features]

        return NetworkFeatureValue(f)

    def __call__(self, h, t):
        return self.f_acc_factory(h, t)

    def f_factory(self, u, v):
        def f(h, t):
            return [feature(h, u, v, t) for feature in self.features]

        return f

    def __getitem__(self, edge):
        u, v = edge
        return self.f_factory(u, v)

    def __len__(self):
        return len(self.features)

    def plot(self, h, u, v, time_interval, ax):

        for i, fname in enumerate(self.feature_names):
            f = lambda t: self[u, v](h, t)[i]

            t_ = np.linspace(0, 2, 100)

            ax.plot(t_, np.vectorize(f)(t_), label=fname)

        ax.legend()


class PaddedFeatureAccessor:
    """
    pva[u,v] return the padded features(list of list of size T * V
    """
    def __init__(self, edge_list, features):
        self.features = features

        h = NetworkHistory()
        x_dict = defaultdict(list)
        pbar = tqdm(edge_list, desc=" Building PaddedFeature dict", disable=True)
        for u, v, t in pbar:
            key = (min(u, v), max(u, v))
            x_dict[key] += [[f(h, u, v, t) for f in features]]
            h.add_interaction(u, v, t)

        seq_len = max([len(v) for k, v in x_dict.items()])
        feature_dim = len(features)
        for k, v in x_dict.items():
            n = len(v)
            for n in range(n, seq_len):
                x_dict[k].append([0] * feature_dim)

        default = [np.zeros(feature_dim)] * seq_len
        self.x_dict = defaultdict(lambda: default, x_dict)

        self.history = h

    def __getitem__(self, edge):
        return self.x_dict[edge]


class PaddedTimeAccessor:
    """
    pva[u,v] return the padded features(list of list of size T * V
    """
    def __init__(self, edge_list):

        t_dict = defaultdict(list)
        for u, v, t in edge_list:
            # key = (min(u, v), max(u, v))
            # key = (u,v)
            t_dict[(u,v)] += [t]
        
        seq_len = max([len(v) for _, v in t_dict.items()]) 


        for k, v in t_dict.items():
            n = len(v)
            for n in range(n, seq_len):
                t_dict[k] += [float("Inf")]

        self.default = [float("Inf")] * seq_len
        self.t_dict = defaultdict(lambda: self.default, t_dict)

    def __getitem__(self, edge):
        return self.t_dict[edge]


