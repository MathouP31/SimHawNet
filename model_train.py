import torch
import torch.optim as optim
from modules.pwc_markov import Markov, ExpKernel
from modules.loss import NLLLoss
from data import TemporalGraphDataModule
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer, roc_auc_score
import numpy as np
import os


class TrainModel():
    def __init__(self, dataset, epoch, nb_feat, model_emb=None, mark=True, decays=1, n_components=128, patience=100):
        self.data_module = TemporalGraphDataModule(dataset)
        self.epoch = epoch
        self.loss_val = []
        self.auc_val = []
        self.dataset = dataset
        self.loss_train = []
        self.auc_train = []
        self.confusion = None

        self.patience = patience  
        self.best_val_loss = np.inf  
        self.epochs_no_improve = 0  

        self.Nloss_train = NLLLoss(neg_scale_factor=self.data_module.train_neg_scale)
        self.Nloss_val = NLLLoss(neg_scale_factor=self.data_module.val_neg_scale)
        # self.Nloss_train = NLLLoss()
        # self.Nloss_val = NLLLoss()

        data_dict = dataset.x_pad.x_dict
        all_non_zero_values_3 = np.concatenate([
            np.array([row[2] for row in value if row[2] != 0]) for value in data_dict.values()
        ])

        all_non_zero_values_2 = np.concatenate([
            np.array([row[1] for row in value if row[1] != 0]) for value in data_dict.values()
        ])

        all_non_zero_values_1 = np.concatenate([
            np.array([row[0] for row in value if row[0] != 0]) for value in data_dict.values()
        ])

        mean_feat_1 = np.mean(all_non_zero_values_1)
        var_feat_1 = np.var(all_non_zero_values_1)

        mean_feat_2 = np.mean(all_non_zero_values_2)
        var_feat_2 = np.var(all_non_zero_values_2)

        mean_feat_3 = np.mean(all_non_zero_values_3)
        var_feat_3 = np.var(all_non_zero_values_3)

        self.mean_feat = [mean_feat_1, mean_feat_2, mean_feat_3]
        self.var_feat = [var_feat_1, var_feat_2, var_feat_3]

        self.model = Markov(n_nodes=dataset.n_nodes, model_emb=model_emb,
                            kernel=ExpKernel(n_features=nb_feat, decays=decays, n_nodes=dataset.n_nodes),
                            mean_feat=self.mean_feat, var_feat=self.var_feat)

    def train_loop_dynamic(self, dataloader):

        loss_train_epoch = 0
        auc_train_epoch = 0

        part_1_parameters = self.model.base_rate.embedding.parameters()
        part_2_parameters = [self.model.base_rate.offset, self.model.base_rate.socialities]
        part_3_parameters = self.model.kernel.parameters()

        opt1 = optim.SparseAdam(part_1_parameters, lr=0.001)
        opt2 = optim.Adam(part_2_parameters, lr=0.001)
        opt3 = optim.Adam(part_3_parameters, lr=0.001)

        opt1.zero_grad()
        opt2.zero_grad()
        opt3.zero_grad()

        batch = dataloader
        src, dst, t, t_prev, x_pad, t_pad, y = batch
        y = y.long()

        pred = self.model(src, dst, t, x_pad, t_pad)
        delta_t = (t - t_prev).ravel()
        loss = self.Nloss_train(pred, delta_t, y)

        loss.backward(retain_graph=True)
        opt1.step()
        opt2.step()
        opt3.step()

        loss_train_epoch += loss.item()
        auc_train_epoch += roc_auc_score(y.detach().numpy(), pred.detach().numpy())

        self.loss_train.append(loss_train_epoch)
        self.auc_train.append(auc_train_epoch)

    def train(self):

        train_dataloader = self.dataset.collate_fn(self.data_module.train_dataset)
        test_dataloader = self.dataset.collate_fn(self.data_module.val_dataset)

        for t in range(self.epoch):
            print(f"Epoch {t + 1}\n-------------------------------")
            self.train_loop_dynamic(train_dataloader)
            val_loss = self.test_loop(test_dataloader)

            # Arrêt précoce
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0  
            else:
                self.epochs_no_improve += 1
                print(f"No improvement for {self.epochs_no_improve} epochs.")
                
            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping at epoch {t + 1}. No improvement in the last {self.patience} epochs.")
                break

        print("Done!")

    def test_loop(self, dataloader):

        test_loss, correct = 0, 0
        auc_val_epoch = 0

        with torch.no_grad():
            batch = dataloader
            src, dst, t, t_prev, x_pad, t_pad, y = batch
            pred = self.model(src, dst, t, x_pad, t_pad)
            delta_t = (t - t_prev).ravel()
            test_loss += self.Nloss_val(pred, delta_t, y).item()
            auc_val_epoch += roc_auc_score(y.detach().numpy(), pred.detach().numpy())

        self.loss_val.append(test_loss)
        self.auc_val.append(auc_val_epoch)

        print(f"Test Error:  Avg loss: {test_loss:>8f} \n")

        return test_loss 
    
    def perf_model(self, path= None):
        dataloader = self.data_module.test_dataloader()
        batch = iter(dataloader)
        with torch.no_grad():
            for src, dst, t ,t_prev, x_pad, t_pad, y in batch:
                pred = self.model(src, dst, t, x_pad, t_pad)
        
            p = pred.detach().numpy()
            Y = y.detach().numpy()
         
            plt.plot(self.loss_train, label ="Train loss")
            plt.plot(self.loss_val, label ="Validation loss")
            plt.legend()
            plt.title("Loss graph_init/Markov")
            plt.show()
            if path !=None:
                loss_plot_path = os.path.join(path, "loss_plot.png")
                plt.savefig(loss_plot_path) 
                plt.close()  
    
        
            # plt.plot(self.auc_train, label ="Train auc")
            # plt.plot(self.auc_val, label ="Validation auc")
            # plt.legend()
            # plt.title("AUC graph_init/Markov")
            # plt.show()
            # if path !=None:
            #     auc_plot_path = os.path.join(path, "auc_plot.png")
            #     plt.savefig(auc_plot_path)  # Save the plot as a PNG file
            #     plt.close()  # Close the figure to free up memory
        

        min_value = pred.min()
        max_value = pred.max()
        normalized_values = (pred - min_value) / (max_value - min_value)
        threshold = torch.mean(normalized_values)
        pred = (normalized_values > threshold)
        self.confusion = confusion_matrix(y, pred)

        print("Confusion matrix : \n",self.confusion) 
