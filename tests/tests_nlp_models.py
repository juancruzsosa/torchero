from .common import *
from torchero.models.text.nn import (LinearModel,
                                     LSTMForTextClassification,
                                     TransformerForTextClassification)

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

