import unittest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import torchtrainer
from torchtrainer.base import BaseTrainer

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.train(mode=True)

    def forward(self, x):
        return x

class TorchBasetrainerTest(unittest.TestCase):
    def assertTensorsEqual(self, a, b):
        return self.assertTrue(torch.eq(a,b).all())

    class TestTrainer(BaseTrainer):
        def __init__(self, update_batch_fn, model):
            super(TorchBasetrainerTest.TestTrainer, self).__init__(model)
            self.update_batch_fn = update_batch_fn

        def update_batch(self, *args, **kwargs):
            self.update_batch_fn(self, *args, **kwargs)

    def load_one_vector_dataset(self):
        self.dataset = TensorDataset(torch.Tensor([[1.0]]), torch.Tensor([[1.0]]))
        self.dataloader = DataLoader(self.dataset, shuffle=False)

    def load_multiple_vector_dataset(self):
        self.dataset = TensorDataset(torch.Tensor([[0.0], [1.0], [-1.0], [-2.0]]),
                                     torch.Tensor([[1.0], [0.0], [1.0], [0.0]]))
        self.dataloader = DataLoader(self.dataset, batch_size=2, shuffle=False)

    def setUp(self):
        self.model = DummyModel()

    def test_cant_train_negative_epochs(self):
        def update_batch_fn(trainer, x, y): pass
        trainer = self.__class__.TestTrainer(update_batch_fn=update_batch_fn, model=self.model)
        self.load_one_vector_dataset()
        try:
            trainer.train(dataloader=self.dataloader, epochs=-1)
            self.fail()
        except Exception as e:
            self.assertEqual(str(e), BaseTrainer.INVALID_EPOCH_MESSAGE.format(epochs=-1))

    def test_not_appling_train_does_not_change_weights(self):
        call = False
        def update_batch_fn(trainer, x, y):
            call = True
        self.model.eval()

        trainer = self.__class__.TestTrainer(update_batch_fn=update_batch_fn, model=self.model)
        self.assertFalse(call)
        self.assertIs(trainer.model, self.model)
        self.assertEqual(trainer.epochs_trained, 0)
        self.assertEqual(self.model.training, False)

    def test_train_turn_batch_into_variables(self):
        batchs = []

        def update_batch_fn(trainer, x, y):
            self.assertTrue(trainer.model.training, True)
            self.assertIsInstance(x, Variable)
            self.assertTensorsEqual(x.data, torch.Tensor([[1]]))
            self.assertIsInstance(y, Variable)
            self.assertTensorsEqual(y.data, torch.Tensor([[1]]))
            batchs.append((x, y))

        trainer = self.__class__.TestTrainer(update_batch_fn=update_batch_fn, model=self.model)
        self.load_one_vector_dataset()
        trainer.train(self.dataloader, epochs=1)

        self.assertEqual(len(batchs), 1)
        self.assertEqual(trainer.epochs_trained, 1)
        self.assertEqual(self.model.training, False)

    def test_train_2_epochs_update_model_two_times(self):
        batchs = []

        def update_batch_fn(trainer, x, y):
            self.assertTrue(trainer.model.training, True)
            self.assertEqual(trainer.total_epochs, 1)
            self.assertEqual(trainer.total_steps, 1)
            self.assertEqual(trainer.step, 0)
            batchs.append((x, y))

        trainer = self.__class__.TestTrainer(update_batch_fn=update_batch_fn, model=self.model)
        self.load_one_vector_dataset()
        trainer.train(self.dataloader, epochs=1)
        trainer.train(self.dataloader, epochs=1)

        self.assertEqual(len(batchs), 2)
        self.assertEqual(self.model.training, False)
        self.assertEqual(trainer.epochs_trained, 2)

    def test_train_multiple_epochs_change_epoch_state_every_time(self):
        epochs = []
        batchs = []

        def update_batch_fn(trainer, x, y):
            self.assertTrue(trainer.model.training, True)
            self.assertIsInstance(x, Variable)
            self.assertTensorsEqual(x.data, torch.Tensor([[1]]))
            self.assertIsInstance(y, Variable)
            self.assertTensorsEqual(y.data, torch.Tensor([[1]]))
            self.assertEqual(trainer.total_epochs, 2)
            self.assertEqual(trainer.total_steps, 1)
            self.assertEqual(trainer.step, 0)
            epochs.append(trainer.epoch)
            batchs.append((x, y))

        trainer = self.__class__.TestTrainer(update_batch_fn=update_batch_fn, model=self.model)
        self.load_one_vector_dataset()
        trainer.train(self.dataloader, epochs=2)

        self.assertEqual(len(batchs), 2)
        for x, y in batchs:
            self.assertIsInstance(y, Variable)
            self.assertTensorsEqual(y.data, torch.Tensor([[1]]))
        self.assertEqual(self.model.training, False)
        self.assertEqual(trainer.epochs_trained, 2)

    def test_train_on_multiply_batch_change_step_state_every_batch(self):
        steps = []
        batchs = []

        def update_batch_fn(trainer, x, y):
            self.assertTrue(trainer.model.training, True)
            self.assertIsInstance(x, Variable)
            self.assertIsInstance(y, Variable)
            self.assertEqual(trainer.total_epochs, 1)
            self.assertEqual(trainer.total_steps, 2)
            steps.append(trainer.step)
            batchs.append((x, y))

        trainer = self.__class__.TestTrainer(update_batch_fn=update_batch_fn, model=self.model)
        self.load_multiple_vector_dataset()
        trainer.train(self.dataloader, epochs=1)

        self.assertEqual(len(batchs), 2)

        for (x_batch, y_batch), (x_true, y_true) in zip(batchs, self.dataloader):
            self.assertTensorsEqual(x_batch.data, x_true)
            self.assertTensorsEqual(y_batch.data, y_true)

        self.assertEqual(steps, [0, 1])

        self.assertEqual(self.model.training, False)
        self.assertEqual(trainer.epochs_trained, 1)

    def test_epochs_trained_is_not_writeable(self):
        def update_batch_fn(trainer, x, y): pass
        trainer = self.__class__.TestTrainer(update_batch_fn=update_batch_fn, model=self.model)
        try:
            trainer.epochs_trained = 0
            self.fail()
        except AttributeError as e:
            self.assertEqual(trainer.epochs_trained, 0)

    def test_can_train_with_no_targets_too(self):
        def update_batch_fn(trainer, x):
            self.assertIsInstance(x, Variable)
        trainer = self.__class__.TestTrainer(update_batch_fn=update_batch_fn, model=self.model)
        tensors = [torch.Tensor([1]), torch.Tensor([2])]
        dataloader = DataLoader(tensors, shuffle=False)
        trainer.train(dataloader, epochs=1)
