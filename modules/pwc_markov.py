from collections import defaultdict
from email.policy import default
from ipaddress import ip_address

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .poisson import Poisson
from .poisson_node2vec import Poisson_node2vec

def get_uniform_ticks(n_intervals):
    ticks = torch.arange(n_intervals + 1) / n_intervals
    ticks[-1] = float("Inf")
    return ticks


def get_geom_ticks(n_intervals):
    ticks = (1 + torch.arange(n_intervals + 1)).log()
    ticks /= ticks.max()
    ticks[-1] = float("Inf")
    return ticks


class Markov(nn.Module):

    def __init__(
        self,
        n_nodes,
        model_emb,
        partition = None,
        label = None,
        n_components=20, # au lieu de 8
        kernel=None,
        mean_feat = None,
        var_feat = None,
        penalisation_fairness = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.partition = partition

        self.base_rate = Poisson(n_nodes=n_nodes,
                                 partition = partition,
                                n_components=n_components,
                                *args,
                                **kwargs) 
        
        # self.base_rate = Poisson_node2vec(n_nodes=n_nodes,
        #                                   model = model_emb,
        #                          partition = partition,
        #                         n_components=n_components,
        #                         *args,
        #                         **kwargs)
        
        self.kernel = kernel
        self.label = label
        self.penalisation_fairness = penalisation_fairness
        self.mean_feat = mean_feat
        self.var_feat = var_feat

    def __get_prev_t_mask(self, t, t_pad):
        """ Returns a mask selecting the events previous to t
        """
        mask = t_pad < t[:, None] 
        return mask

    def __get_last_t_mask(self, t, t_pad):
        """ Compute a mask selecting the last event before t, if there is one
        """
        mask = self.__get_prev_t_mask(t, t_pad) 
        mask_acc = mask.cumsum(dim=1) 
        ranks = mask_acc.max(dim=1).values[:, None] - mask_acc  
        mask_last_t = ranks < 1 
        return mask_last_t 
    
    def __get_prev_t_mask_for_simu(self, t, t_pad):
        """ Returns a mask selecting the events previous to t
        """
        mask = t_pad <= t[:, None]
        return mask

    def __get_last_t_mask_for_simu(self, t, t_pad):
        """ Compute a mask selecting the last event before t, if there is one
        """
        mask = self.__get_prev_t_mask_for_simu(t, t_pad) 
        mask_acc = mask.cumsum(dim=1)
        ranks = mask_acc.max(dim=1).values[:, None] - mask_acc  
        mask_last_t = ranks < 1 
        return mask_last_t 
    
    def pred_for_simu(self, src, dst, t, x_pad, t_pad):
            # Poisson Rate
            out = self.base_rate(src, dst)

            mask_prev_t = self.__get_prev_t_mask_for_simu(t, t_pad) 
            mask_last_t = self.__get_last_t_mask_for_simu(t, t_pad)
            mask = mask_last_t & mask_prev_t 
            eps=1e-10
            # Increment using the features at the last event
            x_last = x_pad[mask]
            n = x_last.shape[0]
            if n != 0 :
                x_last = torch.sigmoid((x_last - torch.tensor([self.mean_feat] * n))/torch.tensor([self.var_feat] * n))
      
            t_last = t_pad[mask]
            is_prev_event = mask.any(dim=1)

            t_cur = t[is_prev_event]
            incr = torch.zeros_like(out)

            src_last = src[is_prev_event]
            dst_last = dst[is_prev_event]

            incr[is_prev_event] = self.kernel(
                src_last,
                dst_last,
                t_cur,
                x_last,
                t_last,
            )
            if self.penalisation_fairness == True :
                pena_fair = torch.zeros_like(src)
                for idx,pen in enumerate(pena_fair):
                    if self.label[src[idx]] != self.label[dst[idx]]:
                        pena_fair[idx]=1
                
                return out + incr + pena_fair
    
            else :
               
                return (out+incr).to(torch.float32)

    def forward(self, src, dst, t, x_pad, t_pad):
        # Poisson Rate
        out = self.base_rate(src, dst)

        mask_prev_t = self.__get_prev_t_mask(t, t_pad) 
        mask_last_t = self.__get_last_t_mask(t, t_pad)
        mask = mask_last_t & mask_prev_t 

        # Increment using the features at the last event
        x_last = x_pad[mask] 
        n = x_last.shape[0]
        if n != 0 and self.mean_feat != None and self.var_feat !=None : 
            x_last = torch.sigmoid((x_last - torch.tensor([self.mean_feat] * n))/torch.tensor([self.var_feat] * n))
        eps=1e-10

        t_last = t_pad[mask]
        is_prev_event = mask.any(dim=1)

        t_cur = t[is_prev_event]
        src_last = src[is_prev_event]
        dst_last = dst[is_prev_event]


        incr = torch.zeros_like(out)
  
        incr[is_prev_event] = self.kernel(
            src_last,
            dst_last,
            t_cur,
            x_last,
            t_last,
        )

        return (out+incr).to(torch.float32)

    def configure_optimizers(self, optim_hparams):
        """ Compute a dictionary of optimizers, to separate the ones that are optimized with adam 
        and the ones with sparse Adam
        """
        opt = defaultdict(list)
        for module in [self.base_rate, self.kernel]:
            for k, v in module.configure_optimizers(optim_hparams).items():
                opt[k] += v

        return opt

class PWCKernel(nn.Module):
    """ Piecewise constant transition kernel
    """

    def __init__(self, n_features, n_intervals):
        super().__init__()
        self.x_linear = nn.Linear(n_features, n_intervals)
        self.x_activation = F.softplus
        self.ticks = get_uniform_ticks(n_intervals)


    def forward(self, t_cur, x_last, t_last):
        logits = self.x_linear(x_last)
        pwc_values = self.x_activation(logits)
        pwc = PieceWiseConstant(self.ticks, values=pwc_values)
        return pwc(t_cur - t_last)

    def configure_optimizers(self, optim_hparams):
        opt = defaultdict(list)
        opt["adam"] += [optim.Adam(self.parameters(), **optim_hparams["adam"])]
        return opt

class MarkovPieceWiseConst(Markov):

    def __init__(self,
                 n_features=3,
                 n_intervals=20,
                 n_nodes=20,
                 *args,
                 **kwargs):
        kernel = PWCKernel(
            n_features=n_features,
            n_intervals=n_intervals,
        )
        super().__init__(n_nodes=n_nodes, kernel=kernel, *args, **kwargs)


class PieceWiseConstant:
    """ Class for PieceWise constant function
        Allows to compute values and cumulative valueje pense que c
    """

    def __init__(self, ticks, values=None):
        self.ticks = ticks

        if len(values.shape) == 1:
            self.values = values[None, :]
        else:
            self.values = values

    def __to_one_hot(self, t):
        """ Return a one hot encoding of the times into the bins of the
        Piecewise constant function
        The returned vector has shape (t.shape[0], self.ticks.shape[1])
        """
        t_vert = t[:, None]
        one_hot = (t_vert >= self.ticks[:-1])
        one_hot = one_hot & (t_vert <= self.ticks[1:])
        return one_hot

    def __call__(self, t): 
        """ Return the value of the piecewise Intensity function
        """
        y = self.__to_one_hot(t)
        return (y * self.values).sum(axis=1)


class ExpKernel(nn.Module):

    def __init__(self, n_features, decays, n_nodes):
        super().__init__()
        self.x_linear = nn.Linear(n_features, 1) 
        self.x_activation = F.softplus 
        # self.decays = decays
        # self.decays = nn.Parameter(torch.randn(1, requires_grad=True))
        self.decays = nn.Parameter(
            torch.randn(n_nodes, requires_grad=True)
        )


    def forward(self,src, dst, t_cur, x_last, t_last):

        delta_t = (t_cur - t_last)
        x_last = x_last.to(torch.float32)
        alpha = self.x_activation(self.x_linear(x_last)).view_as(t_cur) 
        decay = (torch.exp(self.decays[src]) + torch.exp(self.decays[dst])) / 2

        out = alpha * torch.exp(-(decay* delta_t))
        # out = decay * torch.exp(-(1/alpha* delta_t))
        return out.view_as(t_cur)

    def configure_optimizers(self, optim_hparams):
        opt = default(list)
        opt["adam"] = [optim.Adam([self.parameters()], **optim_hparams["adam"])]
        return opt


class MarkovExp(Markov):

    def __init__(self, n_feature=3, *args, **kwargs):
        super().__init__(kernel=ExpKernel(n_features=n_feature),
                         *args,
                         **kwargs)
