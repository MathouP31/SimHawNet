from collections import defaultdict
from typing import Dict

import networkx as nx
import numpy as np


def get_undirected_edge_dict(n_nodes):
    all_edges = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    edge_indices = np.arange(len(all_edges)) 

    return dict(zip(all_edges, edge_indices))


def vectorize(f):
    def f_vec(self, t):
        if isinstance(t, np.ndarray):
            output = np.asarray([f(self, t_) for t_ in t])
            return output.T
        else:
            return f(self, t)

    return f_vec


class TimeHistory:
    def __init__(self, times=None, **attr):
        self.times = []
        if times is not None:
            self.times = list(times)

    def add(self, time, **attr):
        self.times.append(time)

    def __getitem__(self, idx):
        if (idx < len(self)):
            return self.times[idx]
        raise IndexError

    def __len__(self):
        return len(self.times)

    def __call__(self, t):
 
        mask = np.array(self.times) < t
        t_arr = np.array(self.times)[mask]
        time_hist = TimeHistory(times=list(t_arr))
        return time_hist

    def __repr__(self):

        return "TimeHistory with times: " + str(self.times)


class NodeHistory:
    """ Dict of time history
    If we are studying the history of node u, then nh[u] returns the TimeHistory
    associated with the edge (u,v)
    """
    def __init__(
        self,
        nh_dict: Dict[int, TimeHistory] = None,
    ):
        self.nh_dict = nh_dict
        if self.nh_dict is None:
            self.nh_dict = {}

    def add(self, v, t):
        if v not in self.nh_dict.keys():
            self.nh_dict[v] = TimeHistory([t])
        else:
            self.nh_dict[v].add(t)

    def __getitem__(self, idx):

        if idx in self.nh_dict.keys():
            return self.nh_dict[idx]
        return TimeHistory()

    def __len__(self):
        return len(self.nh_dict)

    def keys(self):
        return self.nh_dict.keys()

    def items(self):
        return self.nh_dict.items()

    def __call__(self, t):
        nh_dict = {}
        for k, v in self.nh_dict.items():
            vt = v(t)
            if (len(vt) > 0):
                nh_dict[k] = vt

        return NodeHistory(nh_dict)

    def __repr__(self):
        lines = [f"Node history with the following interactions:"]
        h_lines = []
        for k, v in self.nh_dict.items():
            h_lines.append(f"{k}: " + str(v.times))
        if (len(h_lines) > 0):
            return "\n".join(lines + h_lines)
        else:
            return "Empty node history"

    def __repr__(self):

        lines = [f"Node history with the following interactions:"]

        lines += [f"{v}: {t}" for v, t in self.nh_dict.items()]
        return "\n".join(lines)


class NetworkHistory:
    """ Dict of NodeHistory. nh[u] accesses the Node History of node u.
    We consider an undirected graph, thus nh[u,v] should return the same as nh[v,u]
    """
    def __init__(self, edge_list=None, h_dict=None):
        # Init from edge list
        self.h_dict = defaultdict(NodeHistory)
        if edge_list is not None:
            # print("Initializing from edgelist...")

            for u, v, t in edge_list:
                self.add_interaction(u, v, t)

        # init from dictionary
        if h_dict is not None:
            self.h_dict = h_dict

        if edge_list is None:
            edge_list = []

        self.edge_list = edge_list

    def add_interaction(self, u, v, t):
        """ Add (v,t) as temporal neighbor of u
            Add (u,t) as temporal neighbor of v
        """
        self.h_dict[u].add(v, t)
        self.h_dict[v].add(u, t)

    def node_history_factory(self, nh_dict):
        return NodeHistory(nh_dict)
   

    @property
    def nodes(self):
        return list(self.h_dict.keys())

    @property
    def edges(self):
        """ For each temporal edge, return
        """
        edges = []
        for u, nh in self.h_dict.items():
            for v, th in nh.nh_dict.items():
                for t in th:
                    if u < v:
                        edges.append((u, v, t))

        edges = np.array(edges)
        edges = edges[np.argsort(edges[:, 2])]
        return edges

    @property
    def graph(self):
        G = nx.Graph([(e[0], e[1]) for e in self.edges[:, :2]])

        return G

    def __get_key(self, u, v):
        return min((u, v)), max((u, v))

    def __getitem__(self, idx):
        """
        If idx is a tuple : (u,v) access the TimeHistory associated with edge (u,v)
        If idx is a int u, access the NodeHistory associated with node u

        """
        if isinstance(idx, int):
            if idx in self.h_dict.keys():
                return self.h_dict[idx]
            return NodeHistory()
        if isinstance(idx, tuple):
            u, v = idx
            if u in self.h_dict.keys():
                if v in self.h_dict[u].keys():
                    return self.h_dict[u][v]
            return TimeHistory()
        return NodeHistory()

    def __len__(self):
        return len(self.h_dict.keys())

    def __str__(self):
        lines = [f"Network history with the following interactions:"]
        for u, v, t in self.edges:
            lines.append(f"{u},{v},{t}")

        return "\n".join(lines)

    def __call__(self, t):
        return NetworkHistory(
            h_dict={k: v(t)
                    for k, v in self.h_dict.items()},
        )

    def keys(self):
        return list(self.edges()) + list(self.nodes())
