import tempfile

import torchero
from torchero.models.text.nn import (LinearModel,
                                     LSTMForTextClassification,
                                     TransformerForTextClassification)
from torchero import SupervisedTrainer
from torchero.utils.text.transforms import basic_text_transform
from torchero.utils.text.datasets import TextClassificationDataset
from torchero.models import ModelNotCompiled, load_model_from_file
from torchero.models.text import BinaryTextClassificationModel
from torchero import meters

from .common import *

class NLPNNBaseTests(object):
    def test_output_has_correct_shape(self):
        with torch.no_grad():
            for l in range(1, 7):
                with self.subTest(len=l):
                    x = torch.randint(1,50, (3,l))
                    l = torch.tensor([l] * 3)
                    self.assertEqual(self.model(x, l).shape,
                                     torch.Size([3, 4]))
                    
    def assertTensorsEqual(self, a, b):
        return self.assertEqual(a.tolist(), b.tolist())
    
    def test_config(self):
        self.assertParams(self.model)
    
    def test_pad_does_not_affect_output(self):
        with torch.no_grad():
            x1 = torch.tensor([[1, 2, 0], [3, 0, 0]])
            x2 = torch.tensor([[1, 2], [3, 0]])
            l = torch.tensor([2, 1])
            o1 = self.model(x1, l)
            o2 = self.model(x2, l)
            self.assertTensorsEqual(o1, o2)
            
    def test_can_create_from_config(self):
        model2 = self.model
        x = torch.tensor([[1, 2]])
        l = torch.tensor([2])
        self.assertEqual(model2(x, l).shape,
                         torch.Size([1, 4]))
        self.assertParams(model2)
            
class LinearModelTest(unittest.TestCase, NLPNNBaseTests):
    def setUp(self):
        self.model = LinearModel(vocab_size=50,
                                 embedding_dim=9,
                                 output_size=4)
        self.model.eval()
    
    def assertParams(self, model):
        config = model.config
        self.assertEqual(config, {'vocab_size': 50,
                                  'embedding_dim': 9,
                                  'output_size': 4})
        
class LSTMForTextClassificationTest(unittest.TestCase, NLPNNBaseTests):
    def setUp(self):
        self.model = LSTMForTextClassification(vocab_size=50,
                                               output_size=4,
                                               embedding_dim=9,
                                               hidden_size=12,
                                               num_layers=3,
                                               bidirectional=True,
                                               mode='max',
                                               dropout_clf=0.2)
        self.model.eval()
        
    def assertParams(self, model):
        config = model.config
        self.assertEqual(config, {'vocab_size': 50,
                                  'embedding_dim': 9,
                                  'output_size': 4,
                                  'hidden_size': 12,
                                  'num_layers': 3,
                                  'bidirectional': True,
                                  'mode': 'max',
                                  'dropout_clf': 0.2})
        
class TransformForTextClassificationTest(unittest.TestCase, NLPNNBaseTests):
    def setUp(self):
        self.model = TransformerForTextClassification(vocab_size=50,
                                                      num_outputs=4,
                                                      dim_model=10,
                                                      num_layers=3,
                                                      num_heads=2,
                                                      dim_feedforward=11,
                                                      emb_dropout=0.4,
                                                      tfm_dropout=0.1,
                                                      max_position_len=100)
        self.model.eval()
        
    def assertParams(self, model):
        config = model.config
        self.assertEqual(config, {'vocab_size': 50,
                                  'num_outputs': 4,
                                  'dim_model': 10,
                                  'num_layers': 3,
                                  'num_heads': 2,
                                  'dim_feedforward': 11,
                                  'emb_dropout': 0.4,
                                  'tfm_dropout': 0.1,
                                  'max_position_len': 100})


class BinaryTextClassificationModelTests(unittest.TestCase):
    def setUp(self):
        self.transform = basic_text_transform()
        self.dataset = TextClassificationDataset([
            'This movie is terrible but it has some good effects.',
            'Adrian Pasdar is excellent is this film. He makes a fascinating woman.',
            "I wouldn't rent this one even on dollar rental night.",
            "I don't know why I like this movie so well, but I never get tired of watching it.",
            'Ming The Merciless does a little Bardwork and a movie most foul!',
            'This is the definitive movie version of Hamlet. Branagh cuts nothing, but there are no wasted moments.'
        ], [0, 1, 0, 1, 0, 1], transform=self.transform)
        self.transform.fit(self.dataset.texts())
        self.network = LinearModel(vocab_size=100,
                                   embedding_dim=9,
                                   output_size=1)
        self.model = BinaryTextClassificationModel(self.network, transform=self.transform)
        self.expected_config = {
            'compiled': False,
            'torchero_version': torchero.__version__,
            'torchero_model_type': {'module': 'torchero.models.text.model',
                                    'type': 'BinaryTextClassificationModel'},
            'net': {
                'type': {
                    'module': 'torchero.models.text.nn.linear',
                    'type': 'LinearModel'
                },
                'config': {
                    'vocab_size': 100,
                    'embedding_dim': 9,
                    'output_size': 1
                }
            },
            'labels': None
        }
        self.expected_train_metrics = {'epoch',
                                       'step',
                                       'train_acc',
                                       'train_f1',
                                       'train_loss',
                                       'train_precision',
                                       'train_recall'}

    def test_not_compiled_model_raises_errors(self):
        try:
            optimizer = self.model.optimizer
            self.fail()
        except Exception as e:
            self.assertIsInstance(e, ModelNotCompiled)
        try:
            loss = self.model.loss
            self.fail()
        except Exception as e:
            self.assertIsInstance(e, ModelNotCompiled)
        try:
            trainer = self.model.trainer
            self.fail()
        except Exception as e:
            self.assertIsInstance(e, ModelNotCompiled)
        try:
            history = self.model.history
            self.fail()
        except Exception as e:
            self.assertIsInstance(e, ModelNotCompiled)
        try:
            self.model.fit(self.dataset)
        except Exception as e:
            self.assertIsInstance(e, ModelNotCompiled)

    def test_model_default_compile_params(self):
        self.model.compile('adam')
        self.model.to('cpu')
        self.assertIsInstance(self.model.optimizer, torch.optim.Adam)
        self.assertIsInstance(self.model.loss, torch.nn.BCEWithLogitsLoss)
        self.assertIsInstance(self.model.trainer, SupervisedTrainer)
        self.assertEqual(len(self.model.history), 0)
        self.assertEqual(self.model.device, torch.device('cpu'))

    def test_model_config(self):
        self.assertEqual(self.model.config, self.expected_config)
        self.model.compile('adam')
        self.expected_config['compiled'] = True
        self.assertEqual(self.model.config, self.expected_config)

    def test_fit_on_dataset(self):
        self.model.compile('adam')
        history = self.model.fit(self.dataset, batch_size=3, num_workers=2, pin_memory=True)
        self.assertEqual(len(history), 2)
        self.assertEqual(set(history[0].keys()), self.expected_train_metrics)

    def test_eval_on_dataset(self):
        self.model.compile('adam')
        metrics = self.model.evaluate(self.dataset)
        self.assertEqual(set(metrics.keys()),
                         {'acc',
                          'f1',
                          'loss',
                          'precision',
                          'recall'})

    def test_predict_on_dataset(self):
        predictions = self.model.predict(self.dataset, has_targets=True, to_tensor=False)
        self.assertEqual(len(predictions), len(self.dataset))
        pred = predictions[0].as_dict()
        self.assertTrue(pred[0], 0 < pred[0] < 1)
        tensor = predictions.tensor
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, torch.Size([len(self.dataset), 1]))

    def test_save(self):
        self.model.compile('adam')
        history = self.model.fit(self.dataset, batch_size=3, num_workers=2, pin_memory=True)
        with tempfile.NamedTemporaryFile() as file:
            self.model.save(file.name)
            model = load_model_from_file(file.name)
        self.expected_config['compiled'] = True
        self.assertEqual(model.config, self.expected_config)
        metrics = self.model.evaluate(self.dataset)
        self.assertEqual(set(metrics.keys()),
                         {'acc',
                          'f1',
                          'loss',
                          'precision',
                          'recall'})

    def test_predict_with_named_dataset(self):
        self.model = BinaryTextClassificationModel(self.network, transform=self.transform, labels=['sentiment'])
        text = 'This movie is terrible but it has some good effects.'
        prediction = self.model.predict(text)
        self.assertEqual(list(prediction.as_dict().keys()), ['sentiment'])
