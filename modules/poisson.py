from collections import defaultdict
from email.policy import default

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch_geometric.nn import GCNConv
# from torch_geometric.data import Data

def euclidean(x1, x2):
    return (x1 - x2).norm(dim = 1)

def kronecker(u, v):
    tensor = torch.zeros_like(u)  

    tensor[u != v] = 1
    return tensor


class Poisson(nn.Module):
    def __init__(
        self, n_nodes, n_components=8,partition= None, n_features=None, *args, **kwargs
    ):
        super(Poisson, self).__init__(*args, **kwargs)
        self.partition = partition
        self.offset = nn.Parameter(torch.randn(1, requires_grad=True))

        self.socialities = nn.Parameter(
            torch.randn(n_nodes, requires_grad=True)
        )

        self.embedding = nn.Embedding(n_nodes, n_components, sparse=True)
        # self.embedding = HadamardEmbedder(keyed_vectors=model.wv)
        # self.conv1 = GCNConv(n_features, n_components)
        # self.conv2 = GCNConv(n_components, n_components)


        self.similarity_measure = euclidean
        self.activ1 = F.relu
        self.activation = F.softplus
      

    def forward(self, src, dst, x_pad=None, t_pad=None):
        logit = self.activ1(self.offset)

        logit = logit - self.similarity_measure(
            self.embedding(src), self.embedding(dst)
        ) 
        return torch.exp(logit) 

    def configure_optimizers(self, hparams):
        opt = defaultdict(list)
        opt["sparse_adam"] = [
            optim.SparseAdam(
                self.embedding.parameters(),
                **hparams["sparse_adam"],
            )
        ]
        opt["adam"] = [
            optim.Adam(
                [self.offset, self.socialities],
                **hparams["adam"],
            )
        ]
        opt["adam"] = [
            optim.Adam(
            list(self.parameters()),
            **hparams["adam"],
            )
        ]

        return opt
