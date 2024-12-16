from collections import defaultdict
from email.policy import default
from ipaddress import ip_address

from .poisson import Poisson
import torch
import torch.nn as nn


def get_uniform_ticks(n_intervals):
    ticks = torch.arange(n_intervals + 1) / n_intervals
    ticks[-1] = float("Inf")
    return ticks

def get_geom_ticks(n_intervals):
    ticks = (1 + torch.arange(n_intervals + 1)).log()
    ticks /= ticks.max()
    ticks[-1] = float("Inf")
    return ticks


class Markov_fixe(nn.Module):

    def __init__(
        self,
        n_nodes,
        alpha = 1,
        beta = 1,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.beta = beta
        self.alpha = alpha
        self.base_rate = Poisson(n_nodes=n_nodes,
                                 partition = None,
                                n_components=2,
                                *args,
                                **kwargs) 
    
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
    
    def pred_for_simu(self, src, dst, t,x_pad_simu, t_pad):
  
            out = self.base_rate(src, dst)
            mask_prev_t = self.__get_prev_t_mask_for_simu(t, t_pad) 
            mask_last_t = self.__get_last_t_mask_for_simu(t, t_pad)
            mask = mask_last_t & mask_prev_t 

            t_last = t_pad[mask]
            is_prev_event = mask.any(dim=1)

            t_cur = t[is_prev_event]
            incr = torch.zeros_like(out)

            delta_t = (t_cur - t_last)
            incr[is_prev_event] = out = self.alpha * torch.exp(-(self.beta* delta_t)).view_as(t_cur)

            return (incr).to(torch.float32)






