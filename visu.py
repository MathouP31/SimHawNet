import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
from itertools import combinations
import pandas as pd
from sklearn.manifold import TSNE

def graph_visu(dataset):
    G_init = nx.Graph()
    edge_a_ajouter= [[colonne[0], colonne[1]] for colonne in dataset.edge_list]
    for edge in edge_a_ajouter:
        source, target = edge
        if G_init.has_edge(source, target):
            # L'arête existe déjà, augmentez le nombre d'apparitions
            G_init[source][target]['count'] += 1
        else:
            # L'arête n'existe pas, ajoutez-la avec un attribut 'count' initialisé à 1
            G_init.add_edge(source, target, count=1)
    degrees = dict(G_init.degree())
                
    node_colors = 'pink'
    edge_colors = 'gray'
    node_shape = 'o'

    edge_counts = nx.get_edge_attributes(G_init, 'count')
    edge_widths = [0.15 * edge_counts[edge] for edge in G_init.edges()]
    pos = nx.kamada_kawai_layout(G_init)
    scaling_factor = 1  
    node_sizes = [v * 100 for v in degrees.values()]
    node_sizes_scaled = [scaling_factor * degree for degree in node_sizes]
    vmin = min(node_sizes_scaled)
    vmax = max(node_sizes_scaled)

    custom_colors = ['#FF0000', '#FF3333', '#FF6666', '#FF9999', '#FFCCCC']

    custom_colors.reverse()

    def custom_colormap(value, vmin, vmax, colormap):
        normalized_value = (value - vmin) / (vmax - vmin)
        color_index = int(normalized_value * (len(colormap) - 1))
        return colormap[color_index]

    colors = [custom_colormap(size, vmin, vmax, custom_colors) for size in node_sizes_scaled]

    nx.draw(G_init, pos, with_labels=True, node_size=node_sizes_scaled,width=edge_widths, font_size=10, font_weight='bold',
            node_color=colors, edge_color='gray', alpha=0.8)

    plt.show()


def visu_intensity_global(dataset,dataset_simu, model):
    node_list = np.linspace(0,dataset.n_nodes-1,dataset.n_nodes).astype(int)
    list_edges = list(combinations(node_list,2)) 
        
    node_list = np.linspace(0,dataset.n_nodes-1,dataset.n_nodes).astype(int)
    list_edges = list(combinations(node_list,2)) 
    T = dataset.edge_list[-1][-1]

    src = []
    dst = []

    for edge in list_edges:
        src.append(edge[0])
        dst.append(edge[1])

    src = torch.tensor(src)
    dst = torch.tensor(dst)

    t_last = dataset.edge_list[int(len(dataset.edge_list)/4*3)][-1]

    x_pad_init = []
    t_pad_init = []

    for edge in list_edges:

        x_pad_init.append(torch.tensor(np.array(dataset.x_pad[edge])))
        t_pad_init.append(torch.tensor(np.array(dataset.t_pad[edge])))

    x_pad_init = (torch.stack(x_pad_init,dim=0)).to(torch.float32)
    t_pad_init = (torch.stack(t_pad_init,dim =0)).to(torch.float32) 

    temps = np.linspace(0,T,500)
    indice = np.where(temps >= t_last)[0][0]
    pred_last_dtw = []

    with torch.no_grad():

        lambda_t_last = []
        lambda_t_last_simu = []
        for idx_inutile,t in enumerate(temps):

            t  = torch.full((len(src),),t)

            pred_last = model.pred_for_simu(src,dst,t,x_pad_init,t_pad_init)
            lambda_t_last.append(pred_last.sum())
            pred_last_dtw.append(pred_last) 

    x_pad_simu = []
    t_pad_simu = []

    for edge in list_edges:

        x_pad_simu.append(torch.tensor(np.array(dataset_simu.x_pad[edge])))
        t_pad_simu.append(torch.tensor(np.array(dataset_simu.t_pad[edge])))

    x_pad_simu = (torch.stack(x_pad_simu,dim=0)).to(torch.float32) 
    t_pad_simu = (torch.stack(t_pad_simu,dim =0)).to(torch.float32) 

    temps = np.linspace(0,T,500)

    with torch.no_grad():
        pred_last_dtw_simu= [] 

        lambda_t_last_simu = []

        for idx_inutile,t in enumerate(temps):

            t  = torch.full((len(src),),t)
            pred_last_simu = model.pred_for_simu(src,dst,t,x_pad_simu,t_pad_simu)
            lambda_t_last_simu.append(pred_last_simu.sum())
            pred_last_dtw_simu.append(pred_last_simu) 

        intensity_simu=lambda_t_last_simu

    intensity_simu = np.array(intensity_simu)
    moyenne_colonnes_simu = intensity_simu

    ecart_type_colonnes_simu = np.std(intensity_simu, axis=0)

    plt.plot(temps,moyenne_colonnes_simu, label = "SimGraph", color = 'blue')

    plt.plot(temps,lambda_t_last, label = "Ground truth",color="orange")
    plt.vlines(x=t_last, ymin=np.min(lambda_t_last_simu), ymax=np.max(moyenne_colonnes_simu), linestyles='dashed', colors='red')
    plt.xlabel('t',fontsize=20)
    plt.ylabel(r'$\lambda^{Global}$',fontsize=20)
    plt.legend(fontsize=15, fancybox=True, shadow=True)
    plt.show()

def visu_intensity_global_multiple(dataset,dataset_simu_list, model):
     # Initialisation des noeuds et arêtes
    node_list = np.linspace(0, dataset.n_nodes - 1, dataset.n_nodes).astype(int)
    list_edges = list(combinations(node_list, 2))  
    T = dataset.edge_list[-1][-1]

    # Construction des listes src et dst
    src, dst = zip(*list_edges)
    src = torch.tensor(src)
    dst = torch.tensor(dst)

    # Dernier temps (3/4 du dataset pour la ligne rouge)
    t_last = dataset.edge_list[int(len(dataset.edge_list) / 4 * 3)][-1]

    # Initialisation des pads pour le dataset de vérité terrain
    x_pad_init = torch.stack([torch.tensor(dataset.x_pad[edge]) for edge in list_edges], dim=0).float()
    t_pad_init = torch.stack([torch.tensor(dataset.t_pad[edge]) for edge in list_edges], dim=0).float()

    # Temps pour les prédictions
    temps = np.linspace(0, T, 500)
    indice = np.where(temps >= t_last)[0][0]

    # Calcul des intensités pour la vérité terrain
    lambda_t_last = []
    with torch.no_grad():
        for t in temps:
            t_tensor = torch.full((len(src),), t, dtype=torch.float32)
            pred_last = model.pred_for_simu(src, dst, t_tensor, t_pad_init)
            lambda_t_last.append(pred_last.sum().item())

    # Calcul des intensités pour chaque dataset simulé
    all_simulated_intensities = []
    for dataset_simu in dataset_simu_list:
        x_pad_simu = torch.stack([torch.tensor(dataset_simu.x_pad[edge]) for edge in list_edges], dim=0).float()
        t_pad_simu = torch.stack([torch.tensor(dataset_simu.t_pad[edge]) for edge in list_edges], dim=0).float()
        lambda_t_simu = []

        with torch.no_grad():
            for t in temps:
                t_tensor = torch.full((len(src),), t, dtype=torch.float32)
                pred_last_simu = model.pred_for_simu(src, dst, t_tensor, x_pad_simu, t_pad_simu)
                lambda_t_simu.append(pred_last_simu.sum().item())

        all_simulated_intensities.append(lambda_t_simu)

    # Calcul de la moyenne et de l'écart-type des intensités simulées
    all_simulated_intensities = np.array(all_simulated_intensities)
    moyenne_simu = np.mean(all_simulated_intensities, axis=0)
    ecart_type_simu = np.std(all_simulated_intensities, axis=0)

    # Plot des résultats
    plt.fill_between(temps, moyenne_simu - ecart_type_simu, moyenne_simu + ecart_type_simu, 
                     color='blue', alpha=0.3)
    
   
    plt.plot(temps, moyenne_simu, color='blue')
    plt.plot(temps, lambda_t_last, label='Ground truth', color='orange')
    # plt.axvline(x=t_last, ymin=np.min(lambda_t_last), ymax=np.max(moyenne_simu), 
    #             linestyle='dashed', color='red', label='t_last')
    plt.vlines(x=t_last, ymin=np.min(all_simulated_intensities[0]), ymax=np.max(all_simulated_intensities[0]), linestyles='dashed', colors='red')

    # Configuration de la figure
    plt.xlabel('t', fontsize=20)
    plt.ylabel(r'$\lambda^{Global}$', fontsize=20)
    plt.legend(fontsize=15, fancybox=True, shadow=True)
    plt.show()

def process_edge_datasets(train_data, test_data):
    """
    Process edge datasets to compute various statistics on edges.
    
    Arguments:
    train_data -- numpy array or pandas DataFrame of shape (n, 3) for training edges
    test_data -- numpy array or pandas DataFrame of shape (n, 3) for test edges
    
    Returns:
    edge_stats -- dictionary with the following keys:
        - 'train_edges_count'
        - 'test_edges_count'
        - 'only_train_edges_count'
        - 'only_test_edges_count'
    """
    train_edges = set(tuple(sorted((int(edge[0]), int(edge[1])))) for edge in train_data)
    test_edges = set(tuple(sorted((int(edge[0]), int(edge[1])))) for edge in test_data)

    # Calculate the required statistics
    train_edges_count = len(train_edges)
    test_edges_count = len(test_edges)
    only_train_edges = train_edges - test_edges
    only_test_edges = test_edges - train_edges
    train_not_test_edges_count = len(only_train_edges)
    test_not_train_edges_count = len(only_test_edges)

    edge_stats = {
        'train_edges_count': train_edges_count,
        'test_edges_count': test_edges_count,
        'only_train_edges_count': train_not_test_edges_count,
        'only_test_edges_count': test_not_train_edges_count
    }
    
    return edge_stats

def visu_emb(dataset, model):
    edges_temp_simu = [[colonne[0], colonne[1]] for colonne in dataset.edge_list]
    liste_edges_multigraph_simu = [tuple(map(int, ligne)) for ligne in edges_temp_simu] 
    G = nx.MultiGraph()
    G.add_edges_from(liste_edges_multigraph_simu)
    weights = model.model.base_rate.embedding.weight.detach().numpy()

    tsne = TSNE(n_components=2,perplexity=5)
    embedded_weights = tsne.fit_transform(weights)

    node_degrees = dict(G.degree(weight='weight'))

    node_colors = [node_degrees[node] for node in G.nodes()]

    plt.figure(figsize=(10, 6)) 
    plt.scatter(embedded_weights[:, 0][:50], embedded_weights[:, 1], c=node_colors, cmap='Reds')
    # plt.scatter(embedded_weights[:, 0][50:], embedded_weights[:, 1][50:], c=node_colors[50:], cmap='Reds', label='Cluster 2', s=50)

    plt.show()

def visu_beta(dataset, model):

    node_list = np.arange(dataset.n_nodes) 
    edge_list_neutre = np.array(list(combinations(node_list, 2)))

    decay = model.model.kernel.decays.detach().numpy() 
    beta_by_edge = []
    for edge in edge_list_neutre:
        beta_by_edge.append((np.exp(decay[edge[0]])+np.exp(decay[edge[1]]))/2)
    plt.figure(figsize=(10, 6))
    plt.hist(beta_by_edge, bins=30, color='blue', alpha=0.7, edgecolor='black', density = True)

    plt.title("Beta distribution by Edge", fontsize=16)
    plt.xlabel("Beta", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()