
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt

from irisDataset import IrisDataset
from model import Autoencoder


class Run:
    def __init__(self, dataset_paths: str="", model_path: str="", epochs: int=50, lr: float=0.001, batch_size: int=4, continue_: bool=False):
        self.dataset_paths = dataset_paths
        self.model_path = model_path

        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        self.train_set, self.test_set = self._create_dataloader()
        self.continue_ = continue_
    
    def _create_dataloader(self):
        train_path, test_path = self.dataset_paths[0], self.dataset_paths[1]

        train_set = IrisDataset(path=train_path)
        test_set = IrisDataset(path=test_path)

        train_dataloader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.batch_size,
            num_workers=1,
            shuffle=True,
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_set,
            batch_size=self.batch_size,
            num_workers=1,
            shuffle=True,
        )

        return train_dataloader, test_dataloader

    def train(self):
        autoencoder = Autoencoder().cuda()
        if self.continue_:
            autoencoder.load_state_dict(torch.load(self.model_path))

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=self.lr)

        total_loss = []
        for epoch in range(1, self.epochs + 1):

            epoch_loss = []
            for samples, _ in self.train_set:
                optimizer.zero_grad()

                samples = samples.float().cuda()

                predictions = autoencoder.train()(samples)

                loss = criterion(predictions, samples)
                loss.backward()
                optimizer.step()

                epoch_loss.append(loss.item())

            current_loss = np.mean(epoch_loss)
            total_loss.append(current_loss)

            print("epoch: [", epoch, "/", self.epochs, "] - loss:", round(current_loss, 4))
        
        torch.save(autoencoder.state_dict(), self.model_path)
        print("finished training")

    def save_bottlenecks(self, bottleneck_path: str="", dataset_type: str="train"):
        if dataset_type == "train":
            dataset = self.train_set
        elif dataset_type == "test":
            dataset = self.test_set
        else:
            raise ValueError("'dataset_type' must be 'train' or 'test' !")

        autoencoder = Autoencoder().cuda()
        autoencoder.load_state_dict(torch.load(self.model_path))
        
        total_bottlenecks = []
        for samples, targets in dataset:
            samples = samples.float().cuda()

            bottlenecks = autoencoder.eval()(samples, return_bottlenecks=True)
            bottlenecks = bottlenecks.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()

            for i in range(len(bottlenecks)):
                total_bottlenecks.append([bottlenecks[i].tolist(), targets[i].tolist()])

        with open(bottleneck_path, "w+") as f:
            json.dump(total_bottlenecks, f, indent=4)


run = Run(dataset_paths=["../datasets/iris-dataset/iris_train.json", "../datasets/iris-dataset/iris_test.json"],
          model_path="model.pt",
          epochs=100,
          lr=0.01,
          batch_size=4,
          continue_=False)

# run.train()
# run.save_bottlenecks(bottleneck_path="iris_train_reduced.json", dataset_type="train")
# run.save_bottlenecks(bottleneck_path="iris_test_reduced.json", dataset_type="test")

