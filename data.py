
import torch
from torch.utils.data import DataLoader, Subset


class TemporalGraphDataModule():

    def __init__(
        self,
        dataset,
        tr_test_ratio=0.7,
        train_np_ratio=1,
        val_np_ratio=1,
        num_workers=1,
        batch_size=100,
    ):
        self.dataset = dataset
        self.train_np_ratio = train_np_ratio
        self.val_np_ratio = val_np_ratio
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.train_neg_scale = self.dataset.n_edges / self.dataset.np_ratio
        self.val_neg_scale = self.dataset.n_edges / self.dataset.np_ratio
        
        n_events = len(self.dataset)

        n_train = int(tr_test_ratio * n_events)

        n_val = (n_events - n_train)  // 2

        self.cut_off_times = {
            "train": self.dataset.t[n_train],
            "val": self.dataset.t[n_train + n_val],
        }

        self.train_idx = torch.arange(n_train)
        self.val_idx = torch.arange(n_train, n_train + n_val)
        self.train_small_idx = torch.arange(n_train - n_val, n_train)
        self.test_idx = torch.arange(n_train + n_val, n_events)
        self.n_train = n_train
        self.n_val = n_val

        self.train_dataset = Subset(
            self.dataset,
            indices=self.train_idx,
        )

        self.val_dataset = Subset(
            self.dataset,
            indices=self.val_idx,
        )

        self.train_small_dataset = Subset(
            self.dataset,
            indices=self.train_small_idx,
        )

        self.test_dataset = Subset(
            self.dataset,
            indices=self.test_idx,
        )

        self.train_dataset.np_ratio = train_np_ratio
        self.val_dataset.np_ratio = val_np_ratio
        self.train_small_dataset.np_ratio = val_np_ratio
        self.test_dataset.np_ratio = val_np_ratio

    def train_dataloader(self):
        return DataLoader(self.train_dataset,

                          batch_size=len(self.train_dataset), 
                          collate_fn=self.dataset.collate_fn) 

    def train_small_dataloader(self):
        return DataLoader(self.train_small_dataset,
                          batch_size = len(self.train_small_dataset),
                          collate_fn=self.dataset.collate_fn)

    def val_small_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size = len(self.val_dataset),
                          collate_fn=self.dataset.collate_fn) 

    def val_dataloader(self):
        return [
            self.train_small_dataloader(),
            self.val_small_dataloader(),
        ]

    def test_dataloader(self):

        return DataLoader(self.test_dataset,
                          batch_size=len(self.test_dataset),
                          collate_fn=self.dataset.collate_fn) 
    

