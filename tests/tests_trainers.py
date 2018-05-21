import math
from torch.utils.data import TensorDataset, DataLoader
import torchtrainer
from torchtrainer import SupervisedTrainer, AutoencoderTrainer
from torchtrainer.base import ValidationGranularity
from .common import *

def sign(x):
    return 1 if x > 0 else -1

class TrainerTests(unittest.TestCase):
    def setUp(self):
        self.w = 4
        self.model = nn.Linear(1, 1, bias=False)
        self.model.weight.data = torch.Tensor([[self.w]])
        self.criterion = nn.L1Loss()
        self.optimizer = SGD(self.model.parameters(), lr=1, momentum=0, weight_decay=0)
        self.load_training_dataset()
        self.load_validation_dataset()

    def load_training_dataset(self, start=1, end=5, batch_size=2):
        self.training_dataset = TensorDataset(torch.arange(start, end).view(-1, 1), torch.zeros(end-start, 1))
        self.training_dataloader = DataLoader(self.training_dataset, batch_size=batch_size)

    def load_validation_dataset(self, start=0, end=10, batch_size=2):
        self.validation_dataset = TensorDataset(torch.arange(start, end).view(-1, 1), torch.zeros(end-start, 1))
        self.validation_dataloader = DataLoader(self.validation_dataset, batch_size=batch_size)

    def test_train_val_loss_are_calculated_after_every_log_event(self):
        trainer = SupervisedTrainer(model=self.model, optimizer=self.optimizer, criterion=self.criterion, logging_frecuency=1, validation_granularity=ValidationGranularity.AT_LOG)
        trainer.train(self.training_dataloader, valid_dataloader=self.validation_dataloader, epochs=1)

        self.assertEqual(len(trainer.history), 2)

        for i in range(2):
            keys = ('epoch', 'step', 'train_loss', 'val_loss')
            self.assertEqual(len(trainer.history[i]), len(keys))
            for name in keys:
                self.assertIn(name, trainer.history[i].keys())

        self.assertEqual(trainer.history[0]['epoch'], 0)
        self.assertEqual(trainer.history[0]['step'], 1)
        # first train batch loss is calculated before weights update
        self.assertAlmostEqual(trainer.history[0]['train_loss'], sum(abs(self.w*i) for i in range(1, 3))/2)
        # Validation loss is calculated over the entire valid dataset after weights update
        self.w -= (sign(self.w*1-0)*1 + sign(self.w*2-0)*2)/2.0
        self.assertAlmostEqual(trainer.history[0]['val_loss'], sum(abs(self.w*i) for i in range(0,10))/10)

        self.assertEqual(trainer.history[1]['epoch'], 0)
        self.assertEqual(trainer.history[1]['step'], 2)
        self.assertAlmostEqual(trainer.history[1]['train_loss'], sum(abs(self.w*i) for i in range(3, 5))/2)
        self.w -= (sign(self.w*3-0)*3 + sign(self.w*4-0)*4)/2.0
        self.assertAlmostEqual(trainer.history[1]['val_loss'], sum(abs(self.w*i) for i in range(0,10))/10)
        self.assertAlmostEqual(self.model.weight.data[0][0], self.w)

    def test_trainer_with_acc_meter_argument_measure_train_and_valid_accuracy_with_same_metric(self):
        acc_meter = MSE()
        trainer = SupervisedTrainer(model=self.model, optimizer=self.optimizer, acc_meters={'acc': acc_meter}, criterion=self.criterion, logging_frecuency=1, validation_granularity=ValidationGranularity.AT_LOG)
        trainer.train(self.training_dataloader, valid_dataloader=self.validation_dataloader, epochs=1)

        for i in range(2):
            keys = ('epoch', 'step', 'train_loss', 'val_loss', 'train_acc', 'val_acc')
            for name in keys:
                self.assertIn(name, trainer.history[i].keys())

        self.assertAlmostEqual(trainer.history[0]['train_acc'], sum((self.w*i)**2 for i in range(1, 3))/2)
        self.w -= (sign(self.w*1-0)*1 + sign(self.w*2-0)*2)/2.0
        self.assertAlmostEqual(trainer.history[0]['val_acc'], sum((self.w*i)**2 for i in range(0,10))/10)
        self.assertAlmostEqual(trainer.history[1]['train_acc'], sum((self.w*i)**2 for i in range(3, 5))/2)
        self.w -= (sign(self.w*3-0)*3 + sign(self.w*4-0)*4)/2.0
        self.assertAlmostEqual(trainer.history[1]['val_acc'], sum((self.w*i)**2 for i in range(0,10))/10)

    def test_trainer_with_val_acc_meter_argument_cant_differ_from_train_acc_meter(self):
        acc_meter = MSE()
        val_acc_meter = RMSE()
        trainer = SupervisedTrainer(model=self.model, optimizer=self.optimizer, acc_meters={'acc': acc_meter}, val_acc_meters={'acc': val_acc_meter}, criterion=self.criterion, logging_frecuency=1, validation_granularity=ValidationGranularity.AT_LOG)
        trainer.train(self.training_dataloader, valid_dataloader=self.validation_dataloader, epochs=1)

        self.w -= (sign(self.w*1-0)*1 + sign(self.w*2-0)*2)/2.0
        self.assertAlmostEqual(trainer.history[0]['val_acc'], math.sqrt(sum((self.w*i)**2 for i in range(0,10))/10))
        self.w -= (sign(self.w*3-0)*3 + sign(self.w*4-0)*4)/2.0
        self.assertAlmostEqual(trainer.history[1]['val_acc'], math.sqrt(sum((self.w*i)**2 for i in range(0,10))/10))

class UnsupervisedTrainerTests(unittest.TestCase):
    def setUp(self):
        self.w = 4
        self.model = nn.Linear(1, 1, bias=False)
        self.model.weight.data = torch.Tensor([[self.w]])
        self.criterion = nn.L1Loss()
        self.optimizer = SGD(self.model.parameters(), lr=1, momentum=0, weight_decay=0)
        self.load_training_dataset()
        self.load_validation_dataset()

    def load_training_dataset(self, start=1, end=5, batch_size=2):
        self.training_dataset = torch.arange(start, end).view(-1, 1)
        self.training_dataloader = DataLoader(self.training_dataset, batch_size=batch_size)

    def load_validation_dataset(self, start=0, end=10, batch_size=2):
        self.validation_dataset = torch.arange(start, end).view(-1, 1)
        self.validation_dataloader = DataLoader(self.validation_dataset, batch_size=batch_size)

    def test_autoencoder_trainer_use_supervised_trainer(self):
        trainer = AutoencoderTrainer(model=self.model, optimizer=self.optimizer, criterion=self.criterion, logging_frecuency=1, validation_granularity=ValidationGranularity.AT_LOG)
        trainer.train(self.training_dataloader, valid_dataloader=self.validation_dataloader, epochs=1)

        self.assertEqual(len(trainer.history), 2)

        for i in range(2):
            keys = ('epoch', 'step', 'train_loss', 'val_loss')
            self.assertEqual(len(trainer.history[i]), len(keys))
            for name in keys:
                self.assertIn(name, trainer.history[i].keys())

        self.assertEqual(trainer.history[0]['epoch'], 0)
        self.assertEqual(trainer.history[0]['step'], 1)
        # first train batch loss is calculated before weights update
        self.assertAlmostEqual(trainer.history[0]['train_loss'], sum(abs(self.w*i -  i) for i in range(1, 3))/2)
        # Validation loss is calculated over the entire valid dataset after weights update
        self.w -= (sign(self.w*1-1)*1 + sign(self.w*2-2)*2)/2.0
        self.assertAlmostEqual(trainer.history[0]['val_loss'], sum(abs(self.w*i - i) for i in range(0,10))/10)

        self.assertEqual(trainer.history[1]['epoch'], 0)
        self.assertEqual(trainer.history[1]['step'], 2)
        self.assertAlmostEqual(trainer.history[1]['train_loss'], sum(abs(self.w*i - i) for i in range(3, 5))/2)
        self.w -= (sign(self.w*3-3)*3 + sign(self.w*4-4)*4)/2.0
        self.assertAlmostEqual(trainer.history[1]['val_loss'], sum(abs(self.w*i - i) for i in range(0,10))/10)
        self.assertAlmostEqual(self.model.weight.data[0][0], self.w)
