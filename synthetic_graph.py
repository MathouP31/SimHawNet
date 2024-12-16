import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy
import straph as sg

def seq(vect_mu, vect_sigma):
    nb_node = len(vect_mu)
    sequence = np.zeros(nb_node)
    for i,val in enumerate(sequence):
        val =round(np.random.normal(vect_mu[i], vect_sigma[i]))
        val = max(1,val)
        val = min(nb_node-2,val)
        sequence[i] =val
    if np.sum(sequence)%2 != 0:
        sequence[round(random.uniform(0,len(sequence)-1))]+=1
    return sequence.astype(int)

def mu(vect_mu, deg_noeud):
    vect_mu = [a + b for a, b in zip(vect_mu, deg_noeud)]
    return vect_mu

def sigma(vect_sigma,var_noeud):
    vect_sigma = [a + b for a, b in zip(vect_sigma, var_noeud)]
    return vect_sigma

def conf_simu(vect_mu,vect_sigma,deg_noeud, visu = False):
    vect_mu = mu(vect_mu, deg_noeud)
    degree_sequence = seq(vect_mu, vect_sigma)
    G = nx.configuration_model(degree_sequence)
    G.remove_edges_from(nx.selfloop_edges(G))

    list_edges = (np.array(G.edges)[:,:2]).astype(int)
    random.shuffle(list_edges)
    
    if visu == True:
        pos = nx.spring_layout(G)
        nx.draw(G,pos, with_labels=True, font_weight='bold')
        plt.show()

    return list_edges

def create_config_model(nb_node,nb_loop, poisson = False,lambda_rate= 5, visu = False):
    vect_mu = [round(random.uniform(0, 2)) for _ in range(nb_node)]
    vect_sigma = [round(random.uniform(1, 3)) for _ in range(nb_node)]
    deg_noeud = [round(random.uniform(1, 2)) for _ in range(nb_node)]

    list_edges_temp = np.empty((0,2))
    for i in range(nb_loop):
        list_edges = conf_simu(vect_mu,vect_sigma,deg_noeud, visu = visu)
        list_edges_temp = np.concatenate((list_edges_temp,list_edges), axis = 0)

    if poisson == True :
        current_time = 0
        trame_temp = []
        for _ in range(len(list_edges_temp)):
            inter_arrival_time = np.random.exponential(1 / lambda_rate)
            current_time += inter_arrival_time
            trame_temp.append(current_time)
        trame_temp =  np.sort(np.array(trame_temp).reshape(-1, 1))
    else : 
        trame_temp = np.sort(np.random.uniform(0, 10, len(list_edges_temp))).reshape(-1, 1)

    list_edges_temp =np.concatenate((list_edges_temp, trame_temp), axis = 1)
    return list_edges_temp


def create_sbm_model(nb_node_c1, nb_node_c2,nb_loop, poisson = False, lambda_rate = 5, visu = False):
    list_edges_temp = np.empty((0,2))
    p = [[0.35, 0.03], [0.03, 0.25]] 
    blocks = [nb_node_c1, nb_node_c2] 
    for i in range(nb_loop):
        G = nx.stochastic_block_model(blocks, p)
        temp = np.array(copy.copy(G.edges()))
        np.random.shuffle(temp)
        list_edges_temp= np.concatenate((list_edges_temp, temp), axis = 0)
        if visu == True:
            pos = nx.spring_layout(G)
            nx.draw(G,pos, with_labels=True, font_weight='bold')
            plt.show()

        
    if poisson == True :
        current_time = 0
        trame_temp = []
        for _ in range(len(list_edges_temp)):
            inter_arrival_time = np.random.exponential(1 / lambda_rate)
            current_time += inter_arrival_time
            trame_temp.append(current_time)
        trame_temp =  np.sort(np.array(trame_temp)).reshape(-1, 1)
    else : 
        trame_temp = np.array(np.sort(np.random.uniform(0, 10, len(list_edges_temp)))).reshape(-1,1)

    list_edges_temp = np.concatenate((list_edges_temp, trame_temp), axis = 1)
    return list_edges_temp

def create_sbm_temp(nb_node_c1, nb_node_c2,nb_loop, poisson = False, lambda_rate = 5, visu = False):
    list_edges_temp = np.empty((0,2))
    p1 = [[0.35, 0.03], [0.03, 0.02]]
    p2 = [[0.03, 0.03], [0.03, 0.25]] 

    blocks = [nb_node_c1, nb_node_c2] 
    for i in range(int(nb_loop/2)):
        G = nx.stochastic_block_model(blocks, p1)
        temp = np.array(copy.copy(G.edges()))
        np.random.shuffle(temp)
        list_edges_temp= np.concatenate((list_edges_temp, temp), axis = 0)
        if visu == True:
            pos = nx.spring_layout(G)
            nx.draw(G,pos, with_labels=True, font_weight='bold')
            plt.show()

    for i in range(int(nb_loop/2)):
        G = nx.stochastic_block_model(blocks, p2)
        temp = np.array(copy.copy(G.edges()))
        np.random.shuffle(temp)
        list_edges_temp= np.concatenate((list_edges_temp, temp), axis = 0)
        if visu == True:
            pos = nx.spring_layout(G)
            nx.draw(G,pos, with_labels=True, font_weight='bold')
            plt.show()

        
    if poisson == True :
        current_time = 0
        trame_temp = []
        for _ in range(len(list_edges_temp)):
            inter_arrival_time = np.random.exponential(1 / lambda_rate)
            current_time += inter_arrival_time
            trame_temp.append(current_time)
        trame_temp =  np.sort(np.array(trame_temp)).reshape(-1, 1)
    else : 
        trame_temp = np.array(np.sort(np.random.uniform(0, 10, len(list_edges_temp)))).reshape(-1,1)

    list_edges_temp = np.concatenate((list_edges_temp, trame_temp), axis = 1)
    return list_edges_temp




def create_stream_model(nb_node):
    T = [0, 11] 
    occurrence_law_node = 'poisson'
    presence_law_node = 'uniform'

    occurrence_param_node = 1000
    presence_param_node = 0.1

    occurrence_law_link = 'poisson'
    presence_law_link = 'uniform'

    occurrence_param_link = 5
    presence_param_link = 0.1

    p_link = np.sqrt(nb_node)/nb_node

    S = sg.erdos_renyi(T,
                            nb_node,
                            occurrence_law_node,
                            occurrence_param_node,
                            presence_law_node,
                            presence_param_node,
                            occurrence_law_link,
                            occurrence_param_link,
                            presence_law_link,
                            presence_param_link,
                            p_link)
    
    list_edges_temp = np.empty((1, 3))

    for idx,e in enumerate(S.links) :
            for i,val_deb in enumerate(S.link_presence[idx][::2]):
                    t = random.uniform(val_deb,S.link_presence[idx][i+1])
                    list_edges_temp = np.vstack([list_edges_temp, [e[0],e[1],t]])

    indices_tri = np.argsort(list_edges_temp[:, 2])
    list_edges_temp = list_edges_temp[indices_tri]
  
    return list_edges_temp
