from collections import defaultdict
from email.policy import default

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def euclidean(x1, x2):
    return (x1 - x2).norm(dim = 1)

def kronecker(u, v):
    tensor = torch.zeros_like(u)  

    tensor[u != v] = 1
    return tensor


class Poisson_node2vec(nn.Module):
    def __init__(
        self, n_nodes,model = None, n_components=8,partition= None, n_features=None, *args, **kwargs
    ):
        super(Poisson_node2vec, self).__init__(*args, **kwargs)
        self.partition = partition
        self.offset = nn.Parameter(torch.randn(1, requires_grad=True))

        self.socialities = nn.Parameter(
            torch.randn(n_nodes, requires_grad=True)
        )
        self.model = model
        # self.embedding = nn.Embedding(n_nodes, n_components, sparse=True)
        self.similarity_measure = euclidean
        self.activ1 = F.relu
        self.activation = F.softplus
      

    def forward(self, src, dst, x_pad=None, t_pad=None):
        logit = self.activ1(self.offset)

        logit = logit - self.similarity_measure(
            # torch.tensor([self.model.wv.get_vector(str(node.item())) for node in src]), torch.tensor([self.model.wv.get_vector(str(node.item())) for node in dst])
            self.model(src), self.model(dst)
        ) 
        return torch.exp(logit) 

    def configure_optimizers(self, hparams):
        opt = defaultdict(list)
        # opt["sparse_adam"] = [
        #     optim.SparseAdam(
        #         self.embedding.parameters(),
        #         **hparams["sparse_adam"],
        #     )
        # ]
        opt["adam"] = [
            optim.Adam(
                [self.offset, self.socialities],
                **hparams["adam"],
            )
        ]

        return opt
