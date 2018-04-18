import sys
import unittest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import torchtrainer
from torchtrainer.base import BatchTrainer
from torchtrainer.hooks import Hook, History
from torchtrainer.meters import Averager

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.train(mode=True)
        self.is_cuda = False

    def cuda(self):
        self.is_cuda = True

    def cpu(self):
        self.is_cuda = False

    def forward(self, x):
        return x

def requires_cuda(f):
    def closure(*args, **kwargs):
        if not torch.cuda.is_available():
            print("Skipping `{}´ test cause use CUDA but CUDA isn't available !!".format(f.__name__), file=sys.stderr)
            return
        return f(*args, **kwargs)
    return closure

class TestTrainer(BatchTrainer):
    def __init__(self, model, update_batch_fn=None, valid_batch_fn=None, logging_frecuency=1, hooks=[]):
        super(TestTrainer, self).__init__(model, logging_frecuency=logging_frecuency, hooks=hooks)
        self.update_batch_fn = update_batch_fn
        self.valid_batch_fn = valid_batch_fn

    def update_batch(self, *args, **kwargs):
        if self.update_batch_fn:
            self.update_batch_fn(self, *args, **kwargs)

    def validate_batch(self, *args, **kwargs):
        if self.valid_batch_fn:
            self.valid_batch_fn(self, *args, **kwargs)

class TorchBasetrainerTest(unittest.TestCase):
    def assertTensorsEqual(self, a, b):
        return self.assertTrue(torch.eq(a,b).all())

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
        trainer = TestTrainer(update_batch_fn=update_batch_fn, model=self.model)
        self.load_one_vector_dataset()
        try:
            trainer.train(dataloader=self.dataloader, epochs=-1)
            self.fail()
        except Exception as e:
            self.assertEqual(str(e), BatchTrainer.INVALID_EPOCH_MESSAGE.format(epochs=-1))

    def test_not_appling_train_does_not_change_weights(self):
        call = False
        def update_batch_fn(trainer, x, y):
            call = True
        self.model.eval()

        trainer = TestTrainer(update_batch_fn=update_batch_fn, model=self.model)
        self.assertFalse(call)
        self.assertIs(trainer.model, self.model)
        self.assertFalse(self.model.is_cuda)
        self.assertEqual(trainer.epochs_trained, 0)
        self.assertEqual(self.model.training, False)

    def test_train_turn_batch_into_variables(self):
        batchs = []

        def update_batch_fn(trainer, x, y):
            self.assertTrue(trainer.model.training, True)
            self.assertIsInstance(x, Variable)
            self.assertTensorsEqual(x.data, torch.Tensor([[1]]))
            self.assertFalse(x.is_cuda)
            self.assertIsInstance(y, Variable)
            self.assertTensorsEqual(y.data, torch.Tensor([[1]]))
            self.assertFalse(y.is_cuda)
            batchs.append((x, y))

        trainer = TestTrainer(update_batch_fn=update_batch_fn, model=self.model)
        self.load_one_vector_dataset()
        trainer.train(self.dataloader, epochs=1)

        self.assertEqual(len(batchs), 1)
        self.assertFalse(self.model.is_cuda)
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

        trainer = TestTrainer(update_batch_fn=update_batch_fn, model=self.model)
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
            self.assertFalse(x.is_cuda)
            self.assertIsInstance(y, Variable)
            self.assertTensorsEqual(y.data, torch.Tensor([[1]]))
            self.assertFalse(y.is_cuda)
            self.assertEqual(trainer.total_epochs, 2)
            self.assertEqual(trainer.total_steps, 1)
            self.assertEqual(trainer.step, 0)
            epochs.append(trainer.epoch)
            batchs.append((x, y))

        trainer = TestTrainer(update_batch_fn=update_batch_fn, model=self.model)
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
            self.assertFalse(x.is_cuda)
            self.assertIsInstance(y, Variable)
            self.assertFalse(y.is_cuda)
            self.assertEqual(trainer.total_epochs, 1)
            self.assertEqual(trainer.total_steps, 2)
            steps.append(trainer.step)
            batchs.append((x, y))

        trainer = TestTrainer(update_batch_fn=update_batch_fn, model=self.model)
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
        trainer = TestTrainer(update_batch_fn=update_batch_fn, model=self.model)
        try:
            trainer.epochs_trained = 0
            self.fail()
        except AttributeError as e:
            self.assertEqual(trainer.epochs_trained, 0)

    def test_can_train_with_no_targets_too(self):
        def update_batch_fn(trainer, x):
            self.assertIsInstance(x, Variable)
            self.assertFalse(x.is_cuda)

        trainer = TestTrainer(update_batch_fn=update_batch_fn, model=self.model)
        tensors = [torch.Tensor([1]), torch.Tensor([2])]
        dataloader = DataLoader(tensors, shuffle=False)
        trainer.train(dataloader, epochs=1)

    @requires_cuda
    def test_turn_trainer_to_cuda_turns_models_and_batchs_to_cuda(self):
        def update_batch_fn(trainer, x, y):
            self.assertTrue(x.is_cuda)
            self.assertTrue(y.is_cuda)
        trainer = TestTrainer(update_batch_fn=update_batch_fn, model=self.model)
        trainer.cuda()
        self.assertTrue(self.model.is_cuda)
        self.load_one_vector_dataset()
        trainer.train(self.dataloader, epochs=1)

    @requires_cuda
    def test_turn_trainer_to_cpu_turns_models_and_batchs_to_cpu(self):
        def update_batch_fn(trainer, x, y):
            self.assertFalse(x.is_cuda)
            self.assertFalse(y.is_cuda)
        self.model.cuda()
        trainer = TestTrainer(update_batch_fn=update_batch_fn, model=self.model)
        trainer.cpu()
        self.assertFalse(self.model.is_cuda)
        self.load_one_vector_dataset()
        trainer.train(self.dataloader, epochs=1)

    def test_validate_batch_iterate_over_every_valid_batch_after_each_train_batch(self):
        batchs = []

        train_dataset = [torch.Tensor([i]) for i in range(4)]
        valid_dataset = [torch.Tensor([-i]) for i in range(5)]

        def update_batch_fn(trainer, x):
            self.assertTrue(self.model.training)
            batchs.append(x)

        def validate_batch_fn(trainer, x):
            self.assertFalse(self.model.training)
            batchs.append(x)

        train_dl = DataLoader(train_dataset, shuffle=False, batch_size=1)
        valid_dl = DataLoader(valid_dataset, shuffle=False, batch_size=1)

        trainer = TestTrainer(update_batch_fn=update_batch_fn, model=self.model, valid_batch_fn=validate_batch_fn)
        trainer.train(train_dl, valid_dataloader=valid_dl, epochs=1)

        it = iter(batchs)
        for true_train_batch in train_dl:
            train_batch = next(it)
            self.assertIsInstance(train_batch, Variable)
            self.assertTensorsEqual(train_batch.data, true_train_batch)
            for true_valid_batch in valid_dl:
                valid_batch = next(it)
                self.assertIsInstance(valid_batch, Variable)
                self.assertTensorsEqual(valid_batch.data, true_valid_batch)

    def test_cannot_construct_with_negative_logging_frecuency(self):
        try:
            trainer = TestTrainer(model=self.model, logging_frecuency=-1)
            self.fail()
        except Exception as e:
            self.assertEqual(str(e), BatchTrainer.INVALID_LOGGING_FRECUENCY_MESSAGE.format(logging_frecuency=-1))

    def test_zero_logging_frecuency_not_do_validations(self):
        valid_batchs = []
        train_batchs = []

        train_dataset = [torch.Tensor([i]) for i in range(4)]
        valid_dataset = [torch.Tensor([-i]) for i in range(5)]

        def update_batch_fn(trainer, x):
            train_batchs.append(x)

        def validate_batch_fn(trainer, x):
            valid_batchs.append(x)

        train_dl = DataLoader(train_dataset, shuffle=False, batch_size=2)
        valid_dl = DataLoader(valid_dataset, shuffle=False, batch_size=3)

        trainer = TestTrainer(update_batch_fn=update_batch_fn, model=self.model, valid_batch_fn=validate_batch_fn, logging_frecuency=0)
        trainer.train(dataloader=train_dl, valid_dataloader=valid_dl, epochs=1)
        self.assertEqual(len(train_batchs), len(train_dl))
        self.assertEqual(len(valid_batchs), 0)

        for true_batch, batch in zip(train_dl, train_batchs):
            self.assertIsInstance(batch, Variable)
            self.assertTensorsEqual(batch.data, true_batch)

    def test_validate_batch_iterate_over_every_valid_batch_after_each_logging_frecuency_batchs(self):
        batchs = []

        train_dataset = [torch.Tensor([i]) for i in range(10)]
        valid_dataset = [torch.Tensor([-i]) for i in range(4)]

        def update_batch_fn(trainer, x):
            self.assertTrue(self.model.training)
            batchs.append(x)

        def validate_batch_fn(trainer, x):
            self.assertFalse(self.model.training)
            batchs.append(x)

        train_dl = DataLoader(train_dataset, shuffle=False, batch_size=2)
        valid_dl = DataLoader(valid_dataset, shuffle=False, batch_size=2)

        trainer = TestTrainer(update_batch_fn=update_batch_fn, model=self.model, valid_batch_fn=validate_batch_fn, logging_frecuency=2)
        trainer.train(train_dl, valid_dataloader=valid_dl, epochs=1)

        it = iter(batchs)

        i = 0
        for true_train_batch in train_dl:
            train_batch = next(it)
            self.assertIsInstance(train_batch, Variable)
            self.assertTensorsEqual(train_batch.data, true_train_batch)
            if i % 2 == 1:
                for true_valid_batch in valid_dl:
                    valid_batch = next(it)
                    self.assertIsInstance(valid_batch, Variable)
                    self.assertTensorsEqual(valid_batch.data, true_valid_batch)
            i += 1

    def test_pre_epoch_hooks_are_triggered_before_every_epoch(self):
        dataset = torch.arange(4).view(-1,1)

        dl = DataLoader(dataset, shuffle=False, batch_size=2)

        class CustomHook(Hook):
            def __init__(hook):
                self.epoch = 0

            def pre_epoch(hook):
                self.assertEqual(hook.trainer.epoch, self.epoch)
                self.epoch += 1

        hook = CustomHook()
        trainer = TestTrainer(model=self.model, hooks=[hook])
        self.assertIs(hook.trainer, trainer)
        trainer.train(dl, epochs=2)
        self.assertEqual(self.epoch, 2)

    def test_post_epoch_hooks_are_triggered_before_every_epoch(self):
        batchs = []
        dataset = torch.zeros(2).view(-1,1)

        dl = DataLoader(dataset, shuffle=False, batch_size=1)

        def update_batch_fn(trainer, x):
            batchs.append(x)

        class CustomHook(Hook):
            def __init__(hook):
                self.epoch = 0

            def post_epoch(hook):
                self.assertEqual(hook.trainer.epoch, self.epoch)
                x = batchs[len(dataset) * self.epoch:len(dataset) * (self.epoch + 1)]
                self.assertTensorsEqual(torch.stack(x), Variable(dataset.unsqueeze(0)))
                self.epoch += 1

        trainer = TestTrainer(model=self.model, update_batch_fn=update_batch_fn, hooks=[CustomHook()])
        trainer.train(dl, epochs=2)
        self.assertEqual(self.epoch, 2)

    def test_attaching_multiple_hooks_triggers_sequentially_every_hook(self):
        self.epochs = []

        dataset = torch.zeros(2).view(-1,1)
        dl = DataLoader(dataset, shuffle=False, batch_size=1)

        class CustomHook(Hook):
            def __init__(hook, idx):
                hook.idx = idx
                hook.epoch = 0

            def pre_epoch(hook):
                self.epochs.append((hook.idx, hook.epoch))
                hook.epoch += 1

        hook_0 = CustomHook(0)
        hook_1 = CustomHook(1)

        trainer = TestTrainer(model=self.model, hooks=[hook_0, hook_1])
        trainer.train(dl, epochs=2)
        self.assertEqual(self.epochs, [(0, 0), (1, 0), (0, 1), (1, 1)])

class HooksTests(unittest.TestCase):
    def test_history_hook_register_every_training_stat(self):
        self.model = DummyModel()
        train_dataset = torch.arange(10).view(-1, 1)
        valid_dataset = torch.arange(10).view(-1, 1)

        train_dl = DataLoader(train_dataset, shuffle=False, batch_size=1)
        valid_dl = DataLoader(valid_dataset, shuffle=False, batch_size=1)
        hook = History()

        def update_batch(trainer, x):
            trainer.stats_meters['t_c'].measure(x.data[0][0])

        def validate_batch(trainer, x):
            trainer.stats_meters['v_c'].measure(x.data[0][0])

        trainer = TestTrainer(model=self.model,
                              hooks=[hook],
                              logging_frecuency=5,
                              update_batch_fn=update_batch,
                              valid_batch_fn=validate_batch)
        trainer.stats_meters['t_c'] = Averager()
        trainer.stats_meters['v_c'] = Averager()
        self.assertEqual(trainer.last_stats, {})

        trainer.train(train_dl, valid_dataloader=valid_dl, epochs=1)

        expected_registry = [{'epoch': 0, 'step': 4, 't_c': 2.0, 'v_c': 4.5},
                             {'epoch': 0, 'step': 9, 't_c': 7.0, 'v_c': 4.5}]

        self.assertEqual(hook.registry, expected_registry)
        self.assertEqual(trainer.last_stats, {'t_c': 7.0, 'v_c': 4.5})
