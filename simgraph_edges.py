import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import time
from itertools import combinations
from datasets import create_dataset
import random

class Ogata_edge():
    def __init__(self, top=2):
        self.dataset_simu = None
        self.edges = []
        self.top = top
        self.edges_simu = []

    def simu_ogata(self, dataset, t_last, T, N, model, feature, lambda_rate=1):
        temps_debut = time.time()
        N = N *3
        node_list = np.arange(dataset.n_nodes)
        list_edges = list(combinations(node_list, 2)) 
        self.edges = copy.deepcopy(dataset.edge_list) 
        src, dst = torch.tensor(list_edges, dtype=torch.int64).T
        x_pad_init = torch.tensor(np.array([dataset.x_pad[tuple(edge)] for edge in list_edges]), dtype=torch.float32)
        t_pad_init = torch.tensor(np.array([dataset.t_pad[tuple(edge)] for edge in list_edges]), dtype=torch.float32)

        self.dataset_simu = copy.deepcopy(dataset)
        x_pad = copy.deepcopy(x_pad_init)
        t_pad = copy.deepcopy(t_pad_init)
  
        t = t_last
        n = 0
        t_last_tensor = torch.full((len(src),), t)

        with torch.no_grad():
            pred_sup = model.pred_for_simu(src, dst, t_last_tensor, x_pad, t_pad)
            borne_sup2 = pred_sup 

        taille = borne_sup2.shape[0]
        mask = ~torch.isinf(t_pad[:, 0])
        etape = 0
        base_rate = model.base_rate(src, dst, t)
        while t < T and n < N:
            u_multiple = torch.rand(taille)
            delta_t_multiple = -torch.log(u_multiple[mask]) / (lambda_rate*(borne_sup2[mask]))
            t += min(delta_t_multiple).item()
            t_update = torch.full((len(src),), t)

            with torch.no_grad():
                pred_t_update_temp = model.pred_for_simu(src, dst, t_update, x_pad, t_pad)

            D = random.uniform(0, 1)
            if D <= (pred_t_update_temp.sum()- model.base_rate(src, dst, t).sum()) /( borne_sup2.sum()- model.base_rate(src, dst, t).sum()) : 
                borne = torch.zeros_like(pred_sup)
                borne2 = torch.zeros_like(pred_sup)

                with torch.no_grad():
                    borne[mask] = pred_sup[mask] -model.base_rate(src, dst, t)[mask]
                    borne2[mask] = pred_t_update_temp[mask] - model.base_rate(src, dst, t)[mask]
                    borne[~mask] = model.base_rate(src, dst, t)[~mask] + torch.ones_like(pred_sup)[~mask]
                    borne2[~mask] = model.base_rate(src, dst, t) [~mask] + torch.ones_like(pred_sup)[~mask] * torch.exp(-5*(t_update))[~mask]

                U = torch.rand_like(borne) * borne
                proba_edge = torch.zeros_like(pred_sup)
                mask_accepted = U < borne2

                if mask_accepted.any():
                    proba_edge[mask_accepted] = pred_t_update_temp[mask_accepted]
                    proba_edge /= proba_edge.sum()
                    indice = torch.multinomial(proba_edge, num_samples=1)
                    self.edges.append([int(list_edges[indice][0]), int(list_edges[indice][1]), t])
                    self.edges_simu.append([int(list_edges[indice][0]), int(list_edges[indice][1]), t])
                    self.dataset_simu = create_dataset("affichage", list_edges=self.edges, features=feature) 
                    x_pad = torch.stack([torch.tensor(np.array(self.dataset_simu.x_pad[edge])) for edge in list_edges], dim=0).float() 
                    t_pad = torch.stack([torch.tensor(np.array(self.dataset_simu.t_pad[edge])) for edge in list_edges], dim=0).float() 
                    t_last_tensor = torch.full((len(src),), t)
  
                    with torch.no_grad():
                        pred_sup = model.pred_for_simu(src, dst, t_last_tensor, x_pad, t_pad)
                        borne_sup2 = pred_sup 
                etape +=1
            n += 1
        temps_fin = time.time()
        return n, temps_fin - temps_debut
