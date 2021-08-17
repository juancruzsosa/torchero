import math
import tempfile
import gzip
import pickle

from torch import nn

from torchero.utils.text.vectors import KeyedVectors
from torchero.utils.text.transforms import Vocab

from .common import *

class VectorsUtilsTests(unittest.TestCase):
    def assertTensorsEqual(self, a, b):
        return self.assertEqual(a.tolist(), b.tolist())
    
    def test_cannot_create_empty_vectors(self):
        try:
            vectors = KeyedVectors({}, [])
            self.fail()
        except Exception as e:
            self.assertIsInstance(e, ValueError)
            self.assertEqual(str(e), "Can not create a KeyedVectors instance with 0 vectors")
    
    def test_cannot_create_vectors_with_mismatch_word_vectors(self):
        try:
            vectors = KeyedVectors(['a'], [(1, 2, 3), (2,1,1)])
            self.fail()
        except Exception as e:
            self.assertIsInstance(e, ValueError)
            self.assertEqual(str(e), "The same number of vectors and words was expected")
    
    def test_single_vector(self):
        vectors = KeyedVectors(['a'], [(1, 0, 0)])
        self.assertEqual(len(vectors), 1)
        self.assertEqual(vectors.vector_size, 3)
        self.assertTensorsEqual(vectors['a'], torch.Tensor([1,0,0]))
        self.assertTensorsEqual(vectors[['a', 'a']],
                                torch.Tensor([[1,0,0],
                                              [1,0,0]]))
        self.assertTensorsEqual(vectors.similarity('a', 'a'),
                                torch.tensor(1))
    
    def test_two_orthogonal_vectors(self):
        vectors = KeyedVectors(['a', 'b'], [(1, 1),
                                            (-1, 1)])
        self.assertEqual(len(vectors), 2)
        self.assertEqual(vectors.vector_size, 2)
        self.assertTensorsEqual(vectors['a'], torch.Tensor([1,1]))
        self.assertTensorsEqual(vectors['b'], torch.Tensor([-1,1]))
        self.assertTensorsEqual(vectors[['a', 'b']],
                                torch.Tensor([[1,1],
                                              [-1,1]]))
        self.assertTensorsEqual(vectors.similarity('a', 'b'),
                                torch.tensor(0))
        results_similar = vectors.most_similar('a', topn=10)
        self.assertEqual(len(results_similar), 2)
        self.assertEqual(results_similar[0], ('a', 1.0))
        self.assertEqual(results_similar[1], ('b', 0.0))
        
    def test_two_close_vectors(self):
        vectors = KeyedVectors.from_dict({'a': [1, 1],
                                          'b': [-1,-1],
                                          'c': [0, 1],})
        sim = 1/math.sqrt(2)
        self.assertTensorsEqual(vectors.similarity('a', 'c'),
                                torch.tensor(sim))
        results_similar = vectors.most_similar('a', topn=2)
        self.assertEqual(len(results_similar), 2)
        self.assertEqual(results_similar[0][1], 1.0)
        self.assertAlmostEqual(results_similar[1][1], sim)
        
    def test_word2vec_plain_dump(self):
        vectors = KeyedVectors.from_dict({'a': [1, 0.5],
                                          'b': [0.5, 1]})
        with tempfile.NamedTemporaryFile() as file:
            expected_content = 'a 1.0 0.5\nb 0.5 1.0\n'
            vectors.save(file.name, format='plain', compressed=False)
            with open(file.name, 'r') as fp:
                self.assertEqual(fp.read(), expected_content)
                fp.seek(0)
                vectors2 = KeyedVectors.from_w2v_plain_format(fp)
                self.assertEqual(len(vectors), 2)
                self.assertTensorsEqual(vectors[['a', 'b']],
                                        torch.Tensor([[1,0.5],
                                                      [0.5,1]]))
            vectors.save(file.name, format='plain', compressed=True)
            with gzip.open(file.name, 'rt') as fp:
                self.assertEqual(fp.read(), expected_content)
                fp.seek(0)
                vectors2 = KeyedVectors.from_w2v_plain_format(fp)
                self.assertEqual(len(vectors), 2)
                self.assertTensorsEqual(vectors[['a', 'b']],
                                        torch.Tensor([[1,0.5],
                                                      [0.5,1]]))
    
    def test_word2vec_binary_dump(self):
        vectors = KeyedVectors.from_dict({'a': [1, 0.5],
                                          'b': [0.5, 1]})
        with tempfile.NamedTemporaryFile() as file:
            expected_content = b'2 2\na \x00\x00\x80?\x00\x00\x00?b \x00\x00\x00?\x00\x00\x80?'
            vectors.save(file.name, format='binary', compressed=False)
            with open(file.name, 'rb') as fp:
                self.assertEqual(fp.read(), expected_content)
                fp.seek(0)
                vectors2 = KeyedVectors.from_w2v_binary_format(fp)
                self.assertEqual(len(vectors), 2)
                self.assertTensorsEqual(vectors[['a', 'b']],
                                        torch.Tensor([[1,0.5],
                                                      [0.5,1]]))
            
            vectors.save(file.name, format='binary', compressed=True)
            with gzip.open(file.name, 'r') as fp:
                self.assertEqual(fp.read(), expected_content)
                fp.seek(0)
                vectors2 = KeyedVectors.from_w2v_binary_format(fp)
                self.assertEqual(len(vectors), 2)
                self.assertTensorsEqual(vectors[['a', 'b']],
                                        torch.Tensor([[1,0.5],
                                                      [0.5,1]]))
        
    def test_vectors_can_be_pickled(self):
        vectors = KeyedVectors.from_dict({'a': [1, 0.5],
                                          'b': [0.5, 1]})
        with tempfile.NamedTemporaryFile() as file:
            with open(file.name, 'wb') as fp:
                pickle.dump(vectors, fp)
            with open(file.name, 'rb') as fp:
                vectors = pickle.load(fp)
                self.assertEqual(len(vectors), 2)
                self.assertTensorsEqual(vectors[['a', 'b']],
                                        torch.Tensor([[1,0.5],
                                                      [0.5,1]]))
    
    def test_pickle_dump(self):
        vectors = KeyedVectors.from_dict({'a': [1, 0.5],
                                          'b': [0.5, 1]})
        with tempfile.NamedTemporaryFile() as file:
            vectors.save(file.name, format='pickle', compressed=False)
            with open(file.name, 'rb') as fp:
                vectors = pickle.load(fp)
                self.assertEqual(len(vectors), 2)
                self.assertTensorsEqual(vectors[['a', 'b']],
                                        torch.Tensor([[1,0.5],
                                                      [0.5,1]]))
                
    def test_patch_embeddings(self):
        model = nn.Embedding(5, 2, padding_idx=0)
        embedding_data = torch.tensor(model.weight.data)
        vocab = Vocab(['a', 'b', 'c', 'd'], pad='<pad>')
        vectors = KeyedVectors(['a', 'c'], [(0.5, 0.25), (0.125, 1.0)])
        vectors.replace_embeddings(vocab, model)
        self.assertTensorsEqual(model(torch.tensor([0,1,2,3,4])),
                                torch.stack([torch.zeros(2),
                                             vectors['a'],
                                             embedding_data[2],
                                             vectors['c'],
                                             embedding_data[4]]))
    
    def test_patch_embeddings_can_freeze_embeddings(self):
        model = nn.Embedding(5, 2, padding_idx=0)
        embedding_data = torch.tensor(model.weight.data)
        vocab = Vocab(['a', 'b', 'c', 'd'], pad='<pad>')
        vectors = KeyedVectors(['a', 'c'], [(0.5, 0.25), (0.125, 1.0)])
        vectors.replace_embeddings(vocab, model, freeze=True)
        self.assertFalse(model.weight.data.requires_grad)