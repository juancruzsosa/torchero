import math
from torch.utils.data import TensorDataset, DataLoader
import torchero
from torchero import SupervisedTrainer, AutoencoderTrainer
from torchero.base import ValidationGranularity
from torchero.meters import LossMeter
from .common import *

def sign(x):
    return 1 if x > 0 else -1

class TrainerTests(unittest.TestCase):
    def setUp(self):
        self.w = 4
        self.model = nn.Linear(1, 1, bias=False)
        self.model.weight.data = torch.FloatTensor([[self.w]])
        self.criterion = nn.L1Loss()
        self.optimizer = SGD(self.model.parameters(), lr=1.0, momentum=0.0, weight_decay=0.0)
        self.load_training_dataset()
        self.load_validation_dataset()

    def load_training_dataset(self, start=1, end=5, batch_size=2):
        self.training_dataset = TensorDataset(torch.arange(start, end).view(-1, 1).float(),
                                              torch.zeros(end-start, 1).float())
        self.training_dataloader = DataLoader(self.training_dataset, batch_size=batch_size)

    def load_validation_dataset(self, start=0, end=10, batch_size=2):
        self.validation_dataset = TensorDataset(torch.arange(start, end).view(-1, 1).float(),
                                                torch.zeros(end-start, 1).float())
        self.validation_dataloader = DataLoader(self.validation_dataset, batch_size=batch_size)


    def test_trainer_with_acc_meter_argument_measure_train_and_valid_accuracy_with_same_metric(self):
        acc_meter = MSE()
        trainer = SupervisedTrainer(model=self.model, optimizer=self.optimizer, acc_meters={'acc': acc_meter}, criterion=self.criterion, logging_frecuency=1, validation_granularity=ValidationGranularity.AT_LOG)

        train_meters = trainer.train_meters
        val_meters = trainer.val_meters
        meters = trainer.meters

        self.assertListEqual(sorted(meters.keys()), ['train_acc', 'train_loss', 'val_acc', 'val_loss'])
        self.assertListEqual(sorted(train_meters.keys()), ['acc', 'loss'])
        self.assertListEqual(sorted(val_meters.keys()), ['acc', 'loss'])
        self.assertIsInstance(meters['train_acc'], MSE)
        self.assertIsInstance(meters['val_acc'], MSE)
        self.assertIs(meters['train_acc'], train_meters['acc'])
        self.assertIs(meters['val_acc'], val_meters['acc'])

        trainer.train(self.training_dataloader, valid_dataloader=self.validation_dataloader, epochs=1)

        for i in range(2):
            keys = ('epoch', 'step', 'train_loss', 'val_loss', 'train_acc', 'val_acc')
            for name in keys:
                self.assertIn(name, trainer.history[i].keys())

        first_loss = sum((self.w*i)**2 for i in range(1, 3))
        self.assertAlmostEqual(trainer.history[0]['train_acc'], first_loss/2)
        self.w -= (sign(self.w*1-0)*1 + sign(self.w*2-0)*2)/2.0
        self.assertAlmostEqual(trainer.history[0]['val_acc'], sum((self.w*i)**2 for i in range(0,10))/10)
        second_loss = sum((self.w*i)**2 for i in range(3, 5))
        self.assertAlmostEqual(trainer.history[1]['train_acc'], (first_loss + second_loss)/4)
        self.w -= (sign(self.w*3-0)*3 + sign(self.w*4-0)*4)/2.0
        self.assertAlmostEqual(trainer.history[1]['val_acc'], sum((self.w*i)**2 for i in range(0,10))/10)

    def test_column_layout(self):
        acc_meter = MSE()
        trainer = SupervisedTrainer(model=self.model, optimizer=self.optimizer, acc_meters={'acc': acc_meter}, criterion=self.criterion, logging_frecuency=1, validation_granularity=ValidationGranularity.AT_LOG)
        trainer.train(self.training_dataloader, valid_dataloader=self.validation_dataloader, epochs=2)
        col_layout = trainer.history.get_column_layout()
        expected_layout = { (0,0): { "metrics": "train_acc",
                                     "title": "train_acc",
                                     "ylabel": "acc" },
                            (0,1): { "metrics": "val_acc",
                                     "title": "val_acc",
                                     "ylabel": "acc" },
                            (1,0): { "metrics": "train_loss",
                                     "title": "train_loss",
                                     "ylabel": "loss" },
                            (1,1): { "metrics": "val_loss",
                                     "title": "val_loss",
                                     "ylabel": "loss" }}
        self.assertEqual(col_layout, expected_layout)

    def test_trainer_with_val_acc_meter_argument_cant_differ_from_train_acc_meter(self):
        acc_meter = MSE()
        val_acc_meter = RMSE()
        trainer = SupervisedTrainer(model=self.model, optimizer=self.optimizer, acc_meters={'acc': acc_meter}, val_acc_meters={'acc': val_acc_meter}, criterion=self.criterion, logging_frecuency=1, validation_granularity=ValidationGranularity.AT_LOG)
        trainer.train(self.training_dataloader, valid_dataloader=self.validation_dataloader, epochs=1)

        self.w -= (sign(self.w*1-0)*1 + sign(self.w*2-0)*2)/2.0
        self.assertAlmostEqual(trainer.history[0]['val_acc'], math.sqrt(sum((self.w*i)**2 for i in range(0,10))/10))
        self.w -= (sign(self.w*3-0)*3 + sign(self.w*4-0)*4)/2.0
        self.assertAlmostEqual(trainer.history[1]['val_acc'], math.sqrt(sum((self.w*i)**2 for i in range(0,10))/10))

class BinaryClassificationTrainerTest(unittest.TestCase):
    def setUp(self):
        self.model = BinaryNetwork()
        self.train_ds = TensorDataset(torch.Tensor([[0, 0], [0, 1], [1, 1]]).float(), torch.LongTensor([0, 1, 1]))
        self.val_ds = TensorDataset(torch.Tensor([[1, 0]]).float(), torch.LongTensor([1]))
        self.train_dl = DataLoader(self.train_ds, batch_size=1)
        self.val_dl = DataLoader(self.val_ds, batch_size=1)

    def test_train_with_binary_cross_entropy(self):
        trainer = SupervisedTrainer(model=self.model,
                                    criterion='binary_cross_entropy_wl',
                                    optimizer='adam',
                                    acc_meters=['binary_accuracy_wl', 'precision_wl', 'recall_wl', 'f1_wl'])
        trainer.train(self.train_dl, valid_dataloader=self.val_dl, epochs=1)
        self.assertEqual(trainer.epochs_trained, 1)
        self.assertIn('train_acc', trainer.metrics.keys())
        self.assertIn('val_acc', trainer.metrics.keys())
        self.assertIn('train_precision', trainer.metrics.keys())
        self.assertIn('val_precision', trainer.metrics.keys())
        self.assertIn('train_recall', trainer.metrics.keys())
        self.assertIn('val_recall', trainer.metrics.keys())
        self.assertIn('train_f1', trainer.metrics.keys())
        self.assertIn('val_f1', trainer.metrics.keys())
        self.assertNotIn('train_precision_wl', trainer.metrics.keys())
        self.assertNotIn('train_recall_wl', trainer.metrics.keys())
        self.assertNotIn('train_f1_wl', trainer.metrics.keys())
        default_layout = trainer.history.get_default_layout()
        expected_layout = { (0,0): { "metrics": ["train_acc", "val_acc"],
                                     "title": "train_acc/val_acc",
                                     "ylabel": "acc"},
                            (0,1): { "metrics": ["train_f1", "val_f1"],
                                     "title": "train_f1/val_f1",
                                     "ylabel": "f1" },
                            (1,0): { "metrics": ["train_loss","val_loss"],
                                     "title": "train_loss/val_loss",
                                     "ylabel": "loss" },
                            (1,1): { "metrics": ["train_precision","val_precision"],
                                     "title": "train_precision/val_precision",
                                     "ylabel": "precision"},
                            (2,0): { "metrics": ["train_recall","val_recall"],
                                     "title": "train_recall/val_recall",
                                     "ylabel": "recall" } }
        self.assertEqual(default_layout, expected_layout)
        metrics = trainer.evaluate(self.val_dl)
        self.assertIn('loss', metrics)
        self.assertIn('acc', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)

class UnsupervisedTrainerTests(unittest.TestCase):
    def setUp(self):
        self.w = 4
        self.model = nn.Linear(1, 1, bias=False)
        self.model.weight.data = torch.FloatTensor([[self.w]])
        self.criterion = nn.L1Loss()
        self.optimizer = SGD(self.model.parameters(), lr=1, momentum=0, weight_decay=0)
        self.load_training_dataset()
        self.load_validation_dataset()

    def load_training_dataset(self, start=1, end=5, batch_size=2):
        self.training_dataset = torch.arange(start, end).view(-1, 1).float()
        self.training_dataloader = DataLoader(self.training_dataset, batch_size=batch_size)

    def load_validation_dataset(self, start=0, end=10, batch_size=2):
        self.validation_dataset = torch.arange(start, end).view(-1, 1).float()
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
        first_loss = sum(abs(self.w*i -  i) for i in range(1, 3))
        self.assertAlmostEqual(trainer.history[0]['train_loss'], first_loss/2)
        # Validation loss is calculated over the entire valid dataset after weights update
        self.w -= (sign(self.w*1-1)*1 + sign(self.w*2-2)*2)/2.0
        self.assertAlmostEqual(trainer.history[0]['val_loss'], sum(abs(self.w*i - i) for i in range(0,10))/10)

        second_loss = sum(abs(self.w*i -  i) for i in range(3, 5))
        self.assertEqual(trainer.history[1]['epoch'], 0)
        self.assertEqual(trainer.history[1]['step'], 2)
        self.assertAlmostEqual(trainer.history[1]['train_loss'], (first_loss + second_loss)/4)
        self.w -= (sign(self.w*3-3)*3 + sign(self.w*4-4)*4)/2.0
        self.assertAlmostEqual(trainer.history[1]['val_loss'], sum(abs(self.w*i - i) for i in range(0,10))/10)
        self.assertAlmostEqual(self.model.weight.data[0][0], self.w)
