import torch

from .base import TemporalGraphDataset


class Affichage(TemporalGraphDataset):

    def __init__(self, liste, *args, **kwargs):
        liste = [[int(e[0]), int(e[1]), float(e[2])] for e in liste]
        super().__init__(edge_list=liste,
                         *args,
                         **kwargs)