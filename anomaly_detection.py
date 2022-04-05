import glob
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler
from pathlib import Path
import matplotlib.pyplot as plt 
import re
from sklearn.metrics import accuracy_score

from data import UKR_LETTERS_DS, ANOMALY_DS
from models import VAE, Encoder, Decoder


class Trainer:
    def __init__(self, model, data_path, batch_size, learning_rate, epochs, anomaly_class="Ї", out_path="results", train_size=0.8, seed=42):
        self.model = model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.opt = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.training_data, self.valid_data, self.anomaly_data = self.init_data(self.batch_size, data_path, anomaly_class, train_size, seed) 
        self.results_folder = Path(out_path)
        self.results_folder.mkdir(exist_ok = True)
        self.epoch = 0
        self.anomaly_label = self.ds.label_ind_dct[anomaly_class]
        self.threshold = None
        self.train_losses = {}
        self.test_losses = {}

    def init_data(self, batch_size, data_path, anomaly_class, train_size=0.8, seed=42):
        trans = transforms.Compose([
            transforms.Resize((36, 36)),
            transforms.ToTensor()
        ])

        self.ds = UKR_LETTERS_DS(data_path, trans)
        self.anomaly_ds = ANOMALY_DS(data_path, anomaly_class, trans)
        dataset_size = len(self.ds)
        indices = list(range(dataset_size))
        split = int(np.floor(train_size * dataset_size))
        np.random.seed(seed)
        np.random.shuffle(indices)

        train_indices, val_indices = indices[:split], indices[split:]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_data = torch.utils.data.DataLoader(dataset=self.ds, batch_size=batch_size, sampler=train_sampler, num_workers=4, prefetch_factor=10)
        test_data = torch.utils.data.DataLoader(dataset=self.ds, batch_size=batch_size, sampler=valid_sampler, num_workers=4, prefetch_factor=10)
        anomaly_data = torch.utils.data.DataLoader(dataset=self.anomaly_ds, batch_size=batch_size, num_workers=4, prefetch_factor=10)
        return train_data, test_data, anomaly_data

    def save(self):
        data = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'threshold': self.threshold
        }
        torch.save(data, str(self.results_folder / f'model-{self.epoch}.pt'))

    def load(self, epoch):
        data = torch.load(str(self.results_folder / f'model-{epoch}.pt'))
        self.epoch = data['epoch']
        self.train_losses = data['train_losses']
        self.test_losses = data['test_losses']
        self.threshold = data['threshold']
        self.model.load_state_dict(data['model'])

    def train(self):
        epoch = self.epoch + 1
        while epoch <= self.epochs:
            train_loss = 0
            running_loss = 0
            self.model.train()
            for idx, (x, labels) in enumerate(self.training_data):
                non_anomaly_mask = self.anomaly_label != labels # train only on non-anomalies

                x = x[non_anomaly_mask]
                labels = labels[non_anomaly_mask]

                self.opt.zero_grad()
    
                x_hat = self.model(x)
                loss = ((x - x_hat)**2).sum() + self.model.encoder.kl
                loss.backward()
                self.opt.step()

                running_loss += loss.item() * (self.batch_size / non_anomaly_mask.sum())
                train_loss = running_loss / (idx + 1)
      
            self.model.eval()
            with torch.no_grad():
                valid_loss = 0
                running_loss = 0
                for idx, (x, labels) in enumerate(self.valid_data):
                    non_anomaly_mask = self.anomaly_label != labels
                    x = x[non_anomaly_mask]
                    labels = labels[non_anomaly_mask]

                    x_hat = self.model(x)
                    loss = ((x - x_hat)**2).sum() + self.model.encoder.kl
                    running_loss += loss.item() * (self.batch_size / non_anomaly_mask.sum())
                    valid_loss = running_loss / (idx + 1)

            self.test_losses.update({epoch: valid_loss})
            self.train_losses.update({epoch: train_loss})
            print(f"EPOCH {epoch}, Train loss: {train_loss}, Validation loss: {valid_loss}")
            self.epoch = epoch
            if self.epoch % 10 == 0:
                self.set_threshold()
                self.save()
            epoch += 1

    def set_threshold(self):
        normal_losses = []
        self.model.eval()
        with torch.no_grad():
            for idx, (batch_normal, labels) in enumerate(self.valid_data):
                non_anomaly_mask = self.anomaly_label != labels

                batch_normal = batch_normal[non_anomaly_mask]
                reconstructed_normal = self.model(batch_normal)

                loss_normal = ((batch_normal - reconstructed_normal)**2).sum(dim=(1, 2, 3)).mean()
                normal_losses.append(loss_normal.item())

        self.threshold = sum(normal_losses) / len(normal_losses)


    def test(self):
        if not self.threshold:
            self.set_threshold()

        self.model.eval()
        accuracy = []
        with torch.no_grad():
            for ((idx, (batch_normal, labels)), batch_anomaly) in zip(enumerate(self.valid_data), self.anomaly_data):
                non_anomaly_mask = self.anomaly_label != labels

                batch_normal = batch_normal[non_anomaly_mask]
                reconstructed_normal = self.model(batch_normal)
                reconstructed_anomaly = self.model(batch_anomaly)

                loss_normal = ((batch_normal - reconstructed_normal)**2).sum(dim=(1, 2, 3))
                are_anomalies_normal_preds = loss_normal > self.threshold
                are_anomalies_normal_true = torch.ones(len(are_anomalies_normal_preds), dtype=int)
                loss_anomaly = ((batch_anomaly - reconstructed_anomaly)**2).sum(dim=(1, 2, 3))
                are_anomalies_abnormal_preds = loss_anomaly > self.threshold
                are_anomalies_abnormal_true = torch.zeros(len(are_anomalies_abnormal_preds), dtype=int)

                preds = torch.cat([are_anomalies_normal_preds, are_anomalies_abnormal_preds])
                true = torch.cat([are_anomalies_normal_true, are_anomalies_abnormal_true])

                accuracy.append(accuracy_score(true, preds))

        return sum(accuracy) / len(accuracy)
                

    def plot_losses(self):
        plt.plot(self.train_losses.keys(), self.train_losses.values(), color ='tab:blue') 
        plt.plot(self.test_losses.keys(), self.test_losses.values(), color ='tab:red') 
        plt.title("Train and test losses")
        plt.savefig("losses_anomaly.png")


if __name__ == '__main__':
    latent_dim = 128
    data_path = "letters"
    out_path = "checkpoints_anomalies"
    batch_size = 64
    epochs = 100
    learning_rate = 0.001

    model = VAE(Encoder(latent_dim), Decoder(latent_dim))
    trainer = Trainer(model, data_path, batch_size, learning_rate, epochs, out_path=out_path)

    epoch = 0
    for model_path in os.listdir(out_path): # start with partially trained model if possible
        epoch = max(int(re.findall(r'\d+', model_path)[0]), epoch)
    
    if epoch:
        trainer.load(epoch)

    trainer.train()
    trainer.plot_losses()
    print(trainer.test())
