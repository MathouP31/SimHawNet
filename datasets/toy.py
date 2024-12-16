import torch

from .base import TemporalGraphDataset


class ToyDataset(TemporalGraphDataset):

    def __init__(self, data_path=None, *args, **kwargs):

        super().__init__(edge_list=[
            [0, 1, 0.1],
            [0, 2, 0.2],
            [1, 2, 0.3],
            [0, 3, 0.4],
            [0, 4, 0.5],
            [3, 5, 0.6],
            [3, 4, 0.8],
            [3, 5, 0.9],
            [3, 5, 1.0],
            [4, 1, 1.1],
            [4, 0, 1.2],
            [0, 2, 1.3],
            [1, 2, 1.4],
            [1, 3, 1.6],

        ],
                         *args,
                         **kwargs)
