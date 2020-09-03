from .common import *
from torch.autograd import Variable

class TorchBasetrainerTest(unittest.TestCase):
    def assertTensorsEqual(self, a, b):
        if isinstance(a, Variable):
            a = a.data

        if isinstance(b, Variable):
            b = b.data

        return self.assertEqual(a.tolist(), b.tolist())

    def load_one_vector_dataset(self):
        self.dataset = TensorDataset(torch.Tensor([[1.0]]), torch.Tensor([[1.0]]))
        self.dataloader = DataLoader(self.dataset, shuffle=False)

    def load_multiple_vector_dataset(self):
        self.dataset = TensorDataset(torch.Tensor([[0.0], [1.0], [-1.0], [-2.0]]),
                                     torch.Tensor([[1.0], [0.0], [1.0], [0.0]]))
        self.dataloader = DataLoader(self.dataset, batch_size=2, shuffle=False)

    def load_arange_training_dataset(self, limit, batch_size):
        self.training_dataset = torch.arange(limit).view(limit, 1)
        self.training_dataloader = DataLoader(self.training_dataset, batch_size=batch_size, shuffle=False)

    def load_arange_validation_dataset(self, limit, batch_size):
        self.validation_dataset = torch.arange(limit).view(limit, 1)
        self.validation_dataloader = DataLoader(self.validation_dataset, batch_size=batch_size, shuffle=False)

    def load_list_dataset(self):
        self.training_dataset = [1, 2, 3]
        self.training_dataloader = [[1], [2], [3]]

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
            trainer.epochs_trained = -1
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

        self.load_arange_training_dataset(4, 1)
        self.load_arange_validation_dataset(5, 1)

        def update_batch_fn(trainer, x):
            self.assertTrue(self.model.training)
            batchs.append(x)

        def validate_batch_fn(validator, x):
            self.assertFalse(self.model.training)
            batchs.append(x)

        trainer = TestTrainer(update_batch_fn=update_batch_fn, model=self.model, valid_batch_fn=validate_batch_fn)
        trainer.train(self.training_dataloader, valid_dataloader=self.validation_dataloader, epochs=1)

        it = iter(batchs)
        for true_train_batch in self.training_dataloader:
            train_batch = next(it)
            self.assertIsInstance(train_batch, Variable)
            self.assertTensorsEqual(train_batch.data, true_train_batch)
            for true_valid_batch in self.validation_dataloader:
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

        self.load_arange_training_dataset(4, 2)
        self.load_arange_validation_dataset(5, 3)

        def update_batch_fn(trainer, x):
            train_batchs.append(x)

        def validate_batch_fn(validator, x):
            valid_batchs.append(x)

        trainer = TestTrainer(update_batch_fn=update_batch_fn, model=self.model, valid_batch_fn=validate_batch_fn, logging_frecuency=0)
        trainer.train(dataloader=self.training_dataloader, valid_dataloader=self.validation_dataloader, epochs=1)
        self.assertEqual(len(train_batchs), len(self.training_dataloader))
        self.assertEqual(len(valid_batchs), 0)

        for true_batch, batch in zip(self.training_dataloader, train_batchs):
            self.assertIsInstance(batch, Variable)
            self.assertTensorsEqual(batch.data, true_batch)

    def test_validate_batch_iterate_over_every_valid_batch_after_each_logging_frecuency_batchs(self):
        batchs = []

        self.load_arange_training_dataset(10, 2)
        self.load_arange_validation_dataset(4, 2)

        def update_batch_fn(trainer, x):
            self.assertTrue(self.model.training)
            batchs.append(x)

        def validate_batch_fn(validator, x):
            self.assertFalse(self.model.training)
            batchs.append(x)

        trainer = TestTrainer(update_batch_fn=update_batch_fn, model=self.model, valid_batch_fn=validate_batch_fn, logging_frecuency=2)
        trainer.train(self.training_dataloader, valid_dataloader=self.validation_dataloader, epochs=1)

        it = iter(batchs)

        i = 0
        for true_train_batch in self.training_dataloader:
            train_batch = next(it)
            self.assertIsInstance(train_batch, Variable)
            self.assertTensorsEqual(train_batch.data, true_train_batch)
            if i % 2 == 1:
                for true_valid_batch in self.validation_dataloader:
                    valid_batch = next(it)
                    self.assertIsInstance(valid_batch, Variable)
                    self.assertTensorsEqual(valid_batch.data, true_valid_batch)
            i += 1

    def test_on_epoch_begin_callbacks_are_triggered_before_every_epoch(self):
        self.load_arange_training_dataset(4, 2)

        class CustomCallback(Callback):
            def __init__(callback):
                self.epoch = 0

            def on_epoch_begin(callback):
                self.assertEqual(callback.trainer.epoch, self.epoch)
                self.epoch += 1

        callback = CustomCallback()
        trainer = TestTrainer(model=self.model, callbacks=[callback])
        self.assertIs(callback.trainer, trainer)
        trainer.train(self.training_dataloader, epochs=2)
        self.assertEqual(self.epoch, 2)

    def test_on_epoch_end_callbacks_are_triggered_before_every_epoch(self):
        batchs = []
        dataset = torch.zeros(2).view(-1,1)

        dl = DataLoader(dataset, shuffle=False, batch_size=1)

        def update_batch_fn(trainer, x):
            batchs.append(x)

        class CustomCallback(Callback):
            def __init__(callback):
                self.epoch = 0

            def on_epoch_end(callback):
                self.assertEqual(callback.trainer.epoch, self.epoch)
                x = batchs[len(dataset) * self.epoch:len(dataset) * (self.epoch + 1)]
                self.assertTensorsEqual(torch.stack(x), Variable(dataset.unsqueeze(1)))
                self.epoch += 1

        trainer = TestTrainer(model=self.model, update_batch_fn=update_batch_fn, callbacks=[CustomCallback()])
        trainer.train(dl, epochs=2)
        self.assertEqual(self.epoch, 2)

    def test_attaching_multiple_callbacks_triggers_sequentially_every_callback(self):
        self.epochs = []

        dataset = torch.zeros(2).view(-1,1)
        dl = DataLoader(dataset, shuffle=False, batch_size=1)

        class CustomCallback(Callback):
            def __init__(callback, idx):
                callback.idx = idx
                callback.epoch = 0

            def on_epoch_begin(callback):
                self.epochs.append((callback.idx, callback.epoch))
                callback.epoch += 1

        callback_0 = CustomCallback(0)
        callback_1 = CustomCallback(1)

        trainer = TestTrainer(model=self.model, callbacks=[callback_0, callback_1])
        trainer.train(dl, epochs=2)
        self.assertEqual(self.epochs, [(0, 0), (1, 0), (0, 1), (1, 1)])

    def test_log_is_performed_at_end_if_logging_frecuency_not_divides_nr_of_batchs(self):
        self.validations = 0
        dataset = torch.zeros(7, 1)
        dl = DataLoader(dataset, shuffle=False, batch_size=1)

        class CustomCallback(Callback):
            def on_log(callback):
                self.validations += 1

        trainer = TestTrainer(model=self.model, logging_frecuency=5, callbacks=[CustomCallback()])
        trainer.train(dl, epochs=1)

        self.assertEqual(self.validations, 2)

    def test_meters_should_be_reseted_before_training(self):
        dataset = torch.zeros(1, 1)
        dl = DataLoader(dataset, shuffle=False, batch_size=1)
        meter = meters.Averager()
        meter.measure(0)

        def update_batch_fn(trainer, x):
            trainer.train_meters['x'].measure(1)

        trainer = TestTrainer(model=self.model, update_batch_fn=update_batch_fn, logging_frecuency=1, train_meters={'x' : meter})
        trainer.train(dl, epochs=1)

        self.assertEqual(trainer.metrics, {'x': 1.0})

    def test_validation_granularity_at_epoch_validate_only_after_epoch(self):
        train_batchs = []
        self.validations = 0

        self.load_arange_training_dataset(10, 1)
        self.load_arange_validation_dataset(1, 1)

        def update_batch_fn(trainer, x):
            train_batchs.append(x)

        def validate_batch_fn(validator, x):
            if validator.trainer.epochs_trained == 0:
                self.assertEqual(len(train_batchs), 10)
            elif validator.trainer.epochs_trained == 1:
                self.assertEqual(len(train_batchs), 20)
            self.validations += 1

        trainer = TestTrainer(model=self.model,
                              logging_frecuency=1,
                              update_batch_fn=update_batch_fn,
                              valid_batch_fn=validate_batch_fn,
                              validation_granularity=ValidationGranularity.AT_EPOCH)

        trainer.train(self.training_dataloader, valid_dataloader=self.validation_dataloader, epochs=2)
        self.assertEqual(self.validations, 2)

    def test_validation_granularity_at_epoch_does_not_reset_previous_metrics(self):
        logs = []

        self.load_arange_training_dataset(2, 1)
        self.load_arange_validation_dataset(1, 1)

        def update_batch_fn(trainer, x):
            trainer.train_meters['t'].measure(x.data[0][0])

        def validate_batch_fn(validator, x):
            validator.meters['v'].measure(x.data[0][0])

        class CustomCallback(Callback):
            def on_log(callback):
                logs.append(callback.trainer.metrics)

        trainer = TestTrainer(model=self.model,
                              logging_frecuency=1,
                              update_batch_fn=update_batch_fn,
                              valid_batch_fn=validate_batch_fn,
                              callbacks=[CustomCallback()],
                              train_meters={'t': Averager()},
                              val_meters={'v': Averager()},
                              validation_granularity=ValidationGranularity.AT_EPOCH)

        trainer.train(self.training_dataloader, valid_dataloader=self.validation_dataloader, epochs=2)
        self.assertEqual(logs, [{'t': 0.0}, {'t': 0.5, 'v': 0.0}, {'t': 0.0}, {'t': 0.5, 'v': 0.0}])

    def test_invalid_validation_granularity_argument_raises_exception(self):
        try:
            TestTrainer(model=self.model,
                        logging_frecuency=1,
                        validation_granularity='xyz')
            self.fail()
        except Exception as e:
            self.assertEqual(str(e), BatchTrainer.INVALID_VALIDATION_GRANULARITY_MESSAGE.format(mode='xyz'))

    def test_batch_input_should_not_converted_into_variable_if_is_not_a_tensor(self):
        self.trained = False
        self.load_list_dataset()

        def update_batch_fn(trainer, x):
            self.trained = True
            self.assertIsInstance(x, int)

        t = TestTrainer(model=self.model,
                        logging_frecuency=1,
                        update_batch_fn=update_batch_fn)

        t.train(self.training_dataloader)

        self.assertEqual(self.trained, True)

    def test_add_existent_meter_raises_exception(self):
        t = TestTrainer(model=self.model,
                        logging_frecuency=1,
                        train_meters={'t': Averager()},
                        update_batch_fn=lambda x: None)
        try:
            t.add_named_train_meter('t', Averager())
            self.fail()
        except Exception as e:
            self.assertEqual(str(e), TestTrainer.METER_ALREADY_EXISTS_MESSAGE.format(name='t'))

    def test_add_meter_reads_meters_after_every_log(self):
        logs = []

        self.load_arange_training_dataset(2, 1)
        self.load_arange_validation_dataset(1, 1)

        def update_batch_fn(trainer, x):
            trainer.train_meters['t'].measure(x.data[0][0])

        class CustomCallback(Callback):
            def on_log(callback):
                logs.append(callback.trainer.metrics)

        trainer = TestTrainer(model=self.model,
                              logging_frecuency=1,
                              update_batch_fn=update_batch_fn,
                              callbacks=[CustomCallback()],
                              validation_granularity=ValidationGranularity.AT_EPOCH)
        avg = Averager()
        trainer.add_named_train_meter('t', avg)
        self.assertTrue('t' in trainer.meters_names())
        self.assertTrue(avg in trainer.meters.values())
        self.assertFalse('t' in trainer.metrics)
        trainer.train(self.training_dataloader, epochs=2)
        self.assertEqual(logs, [{'t': 0.0}, {'t': 0.5}, {'t': 0.0}, {'t': 0.5}])
