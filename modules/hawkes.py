import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .poisson import Poisson


class Hawkes(nn.Module):
    def __init__(self, n_nodes, order=None, n_components=8, n_features=3):
        super().__init__()
        self.alpha_linear = nn.Linear(n_features, 1)
        self.alpha_activation = F.softplus
        self.beta_linear = nn.Linear(n_features, 1)
        self.beta_activation = F.softplus
        self.kernel_activation = torch.exp
        self.base_rate = Poisson(n_nodes=n_nodes, n_components=n_components)
        self.order = order

    def check_nan(self, *args):
        for arg in args:
            assert not arg.isnan().any()

    def __get_time_mask(self, t, t_pad):
        """ Returns a mask selecting the self.order events previous to t
        """
        mask = t_pad < t[:, None]
        if self.order is not None:
            mask_acc = mask.cumsum(dim=1)
            ranks = mask_acc.max() - mask_acc
            mask_ranks = ranks < self.order
            return mask & mask_ranks

        return mask

    def increment(self, t, x_pad, t_pad):
        """ Compute increment of intensity coming from the network history
        """
        betas = self.beta_activation(
            self.beta_linear(x_pad)
        ).view_as(t_pad).double() 
        alphas = self.alpha_activation(
            self.alpha_linear(x_pad)
        ).view_as(t_pad).double() 
        delta_t = (t[:, None] - t_pad).view_as(t_pad)
        mask = self.__get_time_mask(t, t_pad)
        terms = torch.zeros_like(t_pad).double()
        terms[mask] = alphas[mask]
        kernel_args = betas[mask] * delta_t[mask]
        terms[mask] = terms[mask] * self.kernel_activation(-kernel_args)
        return (mask.float() * terms).sum(dim=1)

    def forward(self, src, dst, t, x_pad, t_pad):
        hr = self.base_rate(src, dst, t, x_pad, t_pad)
        incr = self.increment(t, x_pad, t_pad).view(hr.shape)
        
        hr = hr + incr
        return hr

    def configure_optimizers(self, optim_hparams):
        optimizers = self.base_rate.configure_optimizers(optim_hparams)
        for module in [self.alpha_linear, self.beta_linear]:
            optimizers += [
                optim.Adam(module.parameters(), **optim_hparams["adam"])
            ]

        return optimizers
