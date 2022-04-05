import glob
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from pathlib import Path
import matplotlib.pyplot as plt 
import re
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

from data import UKR_LETTERS_DS
from models import UKR_LETTERS_NET


class Trainer:
    def __init__(self, model, data_path, batch_size, learning_rate, epochs, out_path="results", train_size=0.8, seed=42):
        self.model = model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.opt = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.training_data, self.valid_data = self.init_data(self.batch_size, data_path, train_size, seed) 
        self.results_folder = Path(out_path)
        self.results_folder.mkdir(exist_ok = True)
        self.epoch = 0
        self.train_losses = {}
        self.test_losses = {}
        self.test_accuracy = {}
        self.metric_funcs = {"accuracy": accuracy_score, "f1": f1_score, 
        "recall": recall_score, "precision": precision_score}

    def init_data(self, batch_size, data_path, train_size=0.8, seed=42):
        trans = transforms.Compose([
            transforms.Resize((36, 36)),
            transforms.ToTensor()
        ])

        self.ds = UKR_LETTERS_DS(data_path, trans)
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
        return train_data, test_data

    def save(self):
        data = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'test_accuracy': self.test_accuracy
        }
        torch.save(data, str(self.results_folder / f'model-{self.epoch}.pt'))

    def load(self, epoch):
        data = torch.load(str(self.results_folder / f'model-{epoch}.pt'))
        self.epoch = data['epoch']
        self.train_losses = data['train_losses']
        self.test_losses = data['test_losses']
        self.test_accuracy = data['test_accuracy']
        self.model.load_state_dict(data['model'])

    def train(self):
        epoch = self.epoch + 1
        while epoch <= self.epochs:
            train_loss = 0
            running_loss = 0
            self.model.train()
            for idx, (batch, labels) in enumerate(self.training_data):
                self.opt.zero_grad()
                preds = self.model(batch)
                loss = self.criterion(preds, labels)
                loss.backward()
                self.opt.step()

                running_loss += loss.item()
                train_loss = running_loss / (idx + 1)
      
            self.model.eval()
            with torch.no_grad():
                valid_loss = 0
                running_loss = 0
                accuracy = 0
                for idx, (batch, labels) in enumerate(self.valid_data):
                    preds = self.model(batch)
                    loss = self.criterion(preds, labels)
                    running_loss += loss.item()
                    valid_loss = running_loss / (idx + 1)

                    _, pred_labels = torch.max(preds, dim=1)
                    accuracy += accuracy_score(labels.numpy(), pred_labels.numpy())

            accuracy = accuracy / len(self.valid_data)
            self.test_losses.update({epoch: valid_loss})
            self.train_losses.update({epoch: train_loss})
            self.test_accuracy.update({epoch: accuracy})
            print(f"EPOCH {epoch}, Train loss: {train_loss}, Validation loss: {valid_loss}, Validation accuracy: {accuracy}")
            self.epoch = epoch
            if self.epoch % 10 == 0:
                self.save()
            epoch += 1

    def test(self):
        metrics = {}
        self.model.eval()
        with torch.no_grad():
            for idx, (batch, labels) in enumerate(self.valid_data):
                preds = self.model(batch)
                _, pred_labels = torch.max(preds, dim=1)
                for metric, func in self.metric_funcs.items():
                    if metric not in metrics:
                        metrics[metric] = 0
                    if metric != "accuracy":
                        metrics[metric] += func(labels.numpy(), pred_labels.numpy(), average="weighted")
                    else:
                        metrics[metric] += func(labels.numpy(), pred_labels.numpy())
            for key in metrics:
                metrics[key] = metrics[key] / len(self.valid_data)
        return metrics
                

    def plot_losses(self):
        plt.plot(self.train_losses.keys(), self.train_losses.values(), color ='tab:blue') 
        plt.plot(self.test_losses.keys(), self.test_losses.values(), color ='tab:red') 
        plt.plot(self.test_accuracy.keys(), self.test_accuracy.values(), color='tab:green')
        plt.title("Train and test losses")
        plt.savefig("losses.png")


if __name__ == '__main__':
    model = UKR_LETTERS_NET()
    data_path = "letters"
    out_path = "checkpoints"
    batch_size = 64
    epochs = 100
    learning_rate = 0.0001

    trainer = Trainer(model, data_path, batch_size, learning_rate, epochs, out_path)

    epoch = 0
    for model_path in os.listdir(out_path): # start with partially trained model if possible
        epoch = max(int(re.findall(r'\d+', model_path)[0]), epoch)
    
    if epoch:
        trainer.load(epoch)

    trainer.train()
    trainer.plot_losses()
    print(trainer.test())
