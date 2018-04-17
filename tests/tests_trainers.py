import math
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD
import unittest
import torchtrainer
from torchtrainer import SupervisedTrainer
from torchtrainer.hooks import History
from torchtrainer.meters import MSE

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

    def test_train_val_loss_are_calculated_after_every_log_event(self):
        w = 4
        history = History()
        model = nn.Linear(1, 1, bias=False)
        model.weight.data = torch.Tensor([[w]])
        criterion = nn.L1Loss()
        optimizer = SGD(model.parameters(), lr=1, momentum=0, weight_decay=0)
        train_dataset = TensorDataset(torch.arange(1, 5).view(4, 1), torch.zeros(4, 1))
        valid_dataset = TensorDataset(torch.arange(0, 10).view(10, 1), torch.zeros(10, 1))
        train_dataloader = DataLoader(train_dataset, batch_size=2)
        valid_dataloader = DataLoader(valid_dataset, batch_size=2)
        trainer = SupervisedTrainer(model=model, optimizer=optimizer, criterion=criterion, hooks=[history], logging_frecuency=1)
        trainer.train(train_dataloader, valid_dataloader=valid_dataloader, epochs=1)

        self.assertEqual(len(history.registry), 2)

        for i in range(2):
            keys = ('epoch', 'step', 'train_loss', 'val_loss')
            self.assertEqual(len(history.registry[i]), len(keys))
            for name in keys:
                self.assertIn(name, history.registry[i].keys())

        self.assertEqual(history.registry[0]['epoch'], 0)
        self.assertEqual(history.registry[0]['step'], 0)
        self.assertEqual(history.registry[1]['epoch'], 0)
        self.assertEqual(history.registry[1]['step'], 1)

        # first train batch loss is calculated before weights update
        self.assertAlmostEqual(history.registry[0]['train_loss'], sum(abs(w*i) for i in range(1, 3))/2)
        # Validation loss is calculated over the entire valid dataset after weights update
        w -= (sign(w*1-0)*1 + sign(w*2-0)*2)/2.0
        self.assertAlmostEqual(history.registry[0]['val_loss'], sum(abs(w*i) for i in range(0,10))/10)

        self.assertAlmostEqual(history.registry[1]['train_loss'], sum(abs(w*i) for i in range(3, 5))/2)
        w -= (sign(w*3-0)*1 + sign(w*4-0)*2)/2.0
        self.assertAlmostEqual(history.registry[1]['val_loss'], sum(abs(w*i) for i in range(0,10))/10)

    def test_trainer_with_acc_meter_argument_measure_train_and_valid_accuracy_with_same_metric(self):
        w = 4
        history = History()
        acc_meter = MSE()
        model = nn.Linear(1, 1, bias=False)
        model.weight.data = torch.Tensor([[w]])
        criterion = nn.L1Loss()
        optimizer = SGD(model.parameters(), lr=1, momentum=0, weight_decay=0)
        train_dataset = TensorDataset(torch.arange(1, 5).view(4, 1), torch.zeros(4, 1))
        valid_dataset = TensorDataset(torch.arange(0, 10).view(10, 1), torch.zeros(10, 1))
        train_dataloader = DataLoader(train_dataset, batch_size=2)
        valid_dataloader = DataLoader(valid_dataset, batch_size=2)
        trainer = SupervisedTrainer(model=model, optimizer=optimizer, acc_meter=acc_meter, criterion=criterion, hooks=[history], logging_frecuency=1)
        trainer.train(train_dataloader, valid_dataloader=valid_dataloader, epochs=1)
        self.assertEqual(len(history.registry), 2)

        for i in range(2):
            keys = ('epoch', 'step', 'train_loss', 'val_loss', 'train_acc', 'val_acc')
            self.assertEqual(len(history.registry[i]), len(keys))
            for name in keys:
                self.assertIn(name, history.registry[i].keys())

        self.assertAlmostEqual(history.registry[0]['train_acc'], sum((w*i)**2 for i in range(1, 3))/2)
        w -= (sign(w*1-0)*1 + sign(w*2-0)*2)/2.0
        self.assertAlmostEqual(history.registry[0]['val_acc'], sum((w*i)**2 for i in range(0,10))/10)
        self.assertAlmostEqual(history.registry[1]['train_acc'], sum((w*i)**2 for i in range(3, 5))/2)
        w -= (sign(w*3-0)*1 + sign(w*4-0)*2)/2.0
        self.assertAlmostEqual(history.registry[1]['val_acc'], sum((w*i)**2 for i in range(0,10))/10)

    def test_trainer_with_val_acc_meter_argument_measure_valid_accuracy_with_diferent_metric(self):
        w = 4
        history = History()
        acc_meter = MSE()
        val_acc_meter = MSE(take_sqrt=True)
        model = nn.Linear(1, 1, bias=False)
        model.weight.data = torch.Tensor([[w]])
        criterion = nn.L1Loss()
        optimizer = SGD(model.parameters(), lr=1, momentum=0, weight_decay=0)
        train_dataset = TensorDataset(torch.arange(1, 5).view(4, 1), torch.zeros(4, 1))
        valid_dataset = TensorDataset(torch.arange(0, 10).view(10, 1), torch.zeros(10, 1))
        train_dataloader = DataLoader(train_dataset, batch_size=2)
        valid_dataloader = DataLoader(valid_dataset, batch_size=2)
        trainer = SupervisedTrainer(model=model, optimizer=optimizer, acc_meter=acc_meter, val_acc_meter=val_acc_meter, criterion=criterion, hooks=[history], logging_frecuency=1)
        trainer.train(train_dataloader, valid_dataloader=valid_dataloader, epochs=1)
        self.assertEqual(len(history.registry), 2)

        for i in range(2):
            keys = ('epoch', 'step', 'train_loss', 'val_loss', 'train_acc', 'val_acc')
            self.assertEqual(len(history.registry[i]), len(keys))
            for name in keys:
                self.assertIn(name, history.registry[i].keys())

        self.assertAlmostEqual(history.registry[0]['train_acc'], sum((w*i)**2 for i in range(1, 3))/2)
        w -= (sign(w*1-0)*1 + sign(w*2-0)*2)/2.0
        self.assertAlmostEqual(history.registry[0]['val_acc'], math.sqrt(sum((w*i)**2 for i in range(0,10))/10))
        self.assertAlmostEqual(history.registry[1]['train_acc'], sum((w*i)**2 for i in range(3, 5))/2)
        w -= (sign(w*3-0)*1 + sign(w*4-0)*2)/2.0
        self.assertAlmostEqual(history.registry[1]['val_acc'], math.sqrt(sum((w*i)**2 for i in range(0,10))/10))
