import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD
import unittest
import torchtrainer
from torchtrainer import SupervisedTrainer

def sign(x):
    return 1 if x > 0 else -1

class TrainerTests(unittest.TestCase):
    def test_01(self):
        w = 4
        model = nn.Linear(1, 1, bias=False)
        model.weight.data = torch.Tensor([[w]])
        criterion = nn.L1Loss()
        optimizer = SGD(model.parameters(), lr=1, momentum=0, weight_decay=0)
        dataset = TensorDataset(torch.arange(1, 5).view(4, 1), torch.zeros(4, 1))
        dataloader = DataLoader(dataset, batch_size=2)
        trainer = SupervisedTrainer(model=model, optimizer=optimizer, criterion=criterion)
        trainer.train(dataloader, epochs=2)

        for i in range(2):
            w -= (sign(w*1-0)*1 + sign(w*2-0)*2)/2.0
            w -= (sign(w*3-0)*3 + sign(w*4-0)*4)/2.0
        self.assertAlmostEqual(model.weight.data[0][0], w)

    def test_02(self):
        w = 4
        model = nn.Linear(1, 1, bias=False)
        model.weight.data = torch.Tensor([[w]])
        criterion = nn.L1Loss()
        optimizer = SGD(model.parameters(), lr=1, momentum=0, weight_decay=0)
        train_dataset = TensorDataset(torch.arange(1, 3).view(2, 1), torch.zeros(2, 1))
        valid_dataset = TensorDataset(torch.arange(3, 5).view(2, 1), torch.zeros(2, 1))
        train_dataloader = DataLoader(train_dataset, batch_size=1)
        valid_dataloader = DataLoader(valid_dataset, batch_size=1)
        trainer = SupervisedTrainer(model=model, optimizer=optimizer, criterion=criterion)
        trainer.train(train_dataloader, valid_dataloader=valid_dataloader, epochs=1)

        # for i in range(2):
        #     w -= (sign(w*1-0)*1 + sign(w*2-0)*2)/2.0
        #     w -= (sign(w*3-0)*3 + sign(w*4-0)*4)/2.0
        # self.assertAlmostEqual(model.weight.data[0][0], w)
