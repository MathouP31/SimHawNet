from datasets import *
# from simgraph import *
# from cross_validation import *
import netlsd
import networkx as nx
from fastdtw import fastdtw
from joblib import Parallel, delayed 
from collections import Counter
import numpy as np
from itertools import combinations 
import torch

def multi_to_weight(G_multi):
    G_weight = nx.Graph()
    for u,v in G_multi.edges():
        if G_weight.has_edge(u,v):
            G_weight[u][v]['weight'] += 1
        else:
            G_weight.add_edge(u, v, weight=1)
    return G_weight

def compute_dtw(x, y):
    distance, _ = fastdtw(x.cpu().numpy(), y.cpu().numpy())
    return distance

def weighted_degree_centrality(G):
    centrality = {}
    for node in G.nodes():
        centrality[node] = sum(G[node][neighbor]['weight'] for neighbor in G.neighbors(node))
    max_possible_weight = sum(data['weight'] for u, v, data in G.edges(data=True))
    centrality = {node: cent / max_possible_weight for node, cent in centrality.items()}
    
    return centrality

class Metrics:
    def __init__(self, dataset_init):

        self.dataset_init = dataset_init

        edges_temp_init = [[colonne[0], colonne[1]] for colonne in dataset_init.edge_list]
        liste_edges_multigraph_init = [tuple(map(int, ligne)) for ligne in edges_temp_init]
        G_init = nx.Graph()
        G_init.add_edges_from(liste_edges_multigraph_init)  
        self.G_init = G_init

        self.G_init_weight = multi_to_weight(G_init)

        self.t_begin = dataset_init.edge_list[0][-1]
        self.t_end = dataset_init.edge_list[-1][-1]
        self.degree = sum(dict(self.G_init_weight.degree(weight='weight')).values())


    def netlsd_clust(self, dataset_simu):

        edges_temp_simu = [[colonne[0], colonne[1]] for colonne in dataset_simu.edge_list]
        liste_edges_multigraph_simu = [tuple(map(int, ligne)) for ligne in edges_temp_simu] 
        G_simulation = nx.Graph()
        G_simulation.add_edges_from(liste_edges_multigraph_simu)
        G_simu_weight = multi_to_weight(G_simulation)
        clustering_init_avg = nx.average_clustering(self.G_init)
        clustering_simu_avg = nx.average_clustering(G_simulation)
        degree = sum(dict(G_simu_weight.degree(weight='weight')).values())

        return clustering_init_avg,clustering_simu_avg,degree
    
    def accuracy(self,dataset_simu,dataset_pred_new):
        liste1 = [tuple(sorted(ligne[:2])) for ligne in self.dataset_init.edge_list]
        liste2 = [tuple(sorted(ligne[:2])) for ligne in dataset_simu.edge_list]
        liste3 = [tuple(sorted(ligne)) for ligne in dataset_pred_new]

        compteur1 = Counter(tuple(paire) for paire in liste1)  
        compteur2 = Counter(tuple(paire) for paire in liste2)
        compteur3 = Counter(tuple(paire) for paire in liste3)

        evenements_communs = 0
        for evenement in compteur1:
            if evenement in compteur2:
                evenements_communs += min(compteur1[evenement], compteur2[evenement])

        pourcentage_similarite = (evenements_communs /  len(liste2)) * 100

        evenements_communs_new = 0
        for evenement in compteur3:
            if evenement in compteur2:
                evenements_communs_new += min(compteur2[evenement], compteur3[evenement])

        pourcentage_new = (evenements_communs_new /  len(liste3)) * 100

        return int(evenements_communs),pourcentage_similarite, int(evenements_communs_new),pourcentage_new

    def dtw(self, dataset_simu, model):

        node_list = np.arange(self.dataset_init.n_nodes)  
        
        list_edges = np.array(list(combinations(node_list, 2)))

        src, dst = torch.tensor(list_edges, dtype=torch.int64).T

        x_pad_init = torch.tensor(np.array([self.dataset_init.x_pad[tuple(edge)] for edge in list_edges]), dtype=torch.float32)
        t_pad_init = torch.tensor(np.array([self.dataset_init.t_pad[tuple(edge)] for edge in list_edges]), dtype=torch.float32)

        x_pad_simu = torch.tensor(np.array([dataset_simu.x_pad[tuple(edge)] for edge in list_edges]), dtype=torch.float32)
        t_pad_simu = torch.tensor(np.array([dataset_simu.t_pad[tuple(edge)] for edge in list_edges]), dtype=torch.float32)


        temps = torch.linspace(self.t_begin, self.t_end, 100)  
        n_edges = len(src)

        intensity_edges = torch.zeros((n_edges, len(temps)), dtype=torch.float32) 
        intensity_global = torch.zeros(len(temps), dtype=torch.float32) 

        intensity_edges_simu = torch.zeros((n_edges, len(temps)), dtype=torch.float32)  
        intensity_global_simu = torch.zeros(len(temps), dtype=torch.float32) 

        with torch.no_grad():
            for idx, t in enumerate(temps):  
                t_tensor = torch.full((n_edges,), t, dtype=torch.float32)  

                pred_last = model.pred_for_simu(src, dst, t_tensor,  x_pad_init, t_pad_init)
                pred_simu = model.pred_for_simu(src, dst, t_tensor, x_pad_simu, t_pad_simu)
                intensity_global[idx] = pred_last.sum()
                intensity_global_simu[idx] = pred_simu.sum()

        x_global = intensity_global
        y_global = intensity_global_simu
        dtw_global_simu, _ = fastdtw(x_global.cpu().numpy(), y_global.cpu().numpy())

        return dtw_global_simu, intensity_edges_simu, intensity_edges, temps