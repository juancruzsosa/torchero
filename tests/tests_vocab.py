from collections import OrderedDict
from torchero.utils.text.transforms import Vocab, OutOfVocabularyError

from .common import *

class VocabTests(unittest.TestCase):
    def test_empty_vocab(self):
        v = Vocab(pad=None)
        self.assertEqual(len(v), 0)
        self.assertEqual(v.max_size, None)
        self.assertListEqual(list(v), [])
        self.assertListEqual(list(v.items()), [])
        self.assertListEqual(list(v.indexes()), [])

    def test_adding_word_to_a_empty_vocab(self):
        v = Vocab(['word'], pad=None)
        self.assertEqual(len(v), 1)
        self.assertEqual(v['word'], 0)
        self.assertEqual(v.freq['word'], 1)
        self.assertListEqual(list(v), ['word'])
        self.assertListEqual(list(v.indexes()), [0])
        self.assertListEqual(list(v.items()), [('word', 0)])

    def test_words_are_added_by_frequency_by_default(self):
        v = Vocab(['a', 'b', 'b', 'b', 'c', 'c'], pad=None)
        self.assertEqual(len(v), 3)
        self.assertEqual(list(v), ['b', 'c', 'a'])
        self.assertListEqual(list(v.indexes()), [0, 1, 2])
        self.assertEqual(v['a'], 2)
        self.assertEqual(v['b'], 0)
        self.assertEqual(v['c'], 1)
        self.assertEqual(v.freq['a'], 1)
        self.assertEqual(v.freq['b'], 3)
        self.assertEqual(v.freq['c'], 2)

    def test_max_size_limits_prevents_from_keep_adding_element(self):
        v = Vocab(['a', 'b', 'b', 'b', 'c', 'c', 'b'], max_size=2, pad=None)
        self.assertEqual(list(v), ['b', 'c'])
        self.assertTrue('a' not in v)
        self.assertEqual(v.freq['b'], 4)
        self.assertEqual(v.freq['c'], 2)
        v.add(['a'])
        self.assertEqual(list(v), ['b', 'c'])
        v.max_size += 2
        v.add(['a', 'd', 'e'])
        self.assertTrue('a' in v)
        self.assertListEqual(list(v), ['b', 'c', 'a', 'd'])
        self.assertEqual(v.freq['a'], 1)
        self.assertEqual(v.freq['b'], 4)
        self.assertEqual(v.freq['c'], 2)
        self.assertEqual(v.freq['d'], 1)

    def test_old_elements_still_counting_after_reach_max_vocabulary_size(self):
        v = Vocab(['a', 'b'], max_size=2, pad=None)
        self.assertTrue('a' in v)
        self.assertTrue('b' in v)
        v.add(['c', 'c', 'b'])
        self.assertFalse('c' in v)
        self.assertEqual(v.freq['b'], 2)

    def test_frequencies_are_added_for_old_words_and_order_is_preserved(self):
        v = Vocab(['a', 'a', 'b'], pad='<pad>')
        self.assertListEqual(list(v), ['<pad>', 'a', 'b'])
        self.assertEqual(v.freq['a'], 2)
        self.assertEqual(v.freq['b'], 1)
        v.add(['b'] * 5 + ['c'] * 3 + ['d'] * 4)
        self.assertListEqual(list(v), ['<pad>', 'a', 'b', 'd', 'c'])
        self.assertEqual(v.freq['a'], 2)
        self.assertEqual(v.freq['b'], 6)
        self.assertEqual(v.freq['c'], 3)
        self.assertEqual(v.freq['d'], 4)

    def test_raises_if_word_is_not_in_vocabulary_when_no_unk(self):
        v = Vocab(['a'], pad=None)
        self.assertEqual(v['a'], 0)
        try:
            v['b']
            self.fail()
        except Exception as e:
            self.assertIsInstance(e, OutOfVocabularyError)

    def test_vocab_can_be_created_from_dictionary_of_frequencies(self):
        v = Vocab({'b': 2, 'a': 5}, pad='<pad>')

        self.assertEqual(v['a'], 1)
        self.assertEqual(v['b'], 2)
        self.assertEqual(v.freq['a'], 5)
        self.assertEqual(v.freq['b'], 2)

    def test_vocab_can_be_built_from_list_of_sequences(self):
        v = Vocab.build_from_texts([['a', 'b', 'a', 'b', 'b'],
                                    ['c', 'a', 'b', 'b', 'b']], pad=None)
        self.assertEqual(list(v), ['b', 'a', 'c'])
        self.assertEqual(v.freq['a'], 3)
        self.assertEqual(v.freq['b'], 6)
        self.assertEqual(v.freq['c'], 1)

    def test_vocab_call_ignores_unknown_words(self):
        v = Vocab({'a': 2, 'b': 1}, pad=None)
        self.assertListEqual(v(['a', 'b', 'c', 'b', 'a']), [0, 1, 1, 0])

    def test_vocab_can_be_created_with_insertion_order(self):
        v = Vocab(['a', 'b', 'b'], order_by='insertion')
        self.assertEqual(v['a'], 1)
        self.assertEqual(v.freq['a'], 1)
        self.assertEqual(v['b'], 2)
        self.assertEqual(v.freq['b'], 2)
        v.add(OrderedDict([('c', 1), ('d', 1)]))
        self.assertEqual(v['c'], 3)
        self.assertEqual(v['d'], 4)

    def test_vocab_can_be_created_with_alphabetical_order(self):
        v = Vocab(['d', 'd', 'c'], order_by='alpha', pad=None)
        self.assertDictEqual(v.freq, {'c': 1, 'd': 2})
        self.assertListEqual(list(v), ['c', 'd'])
        v.add(['b', 'b', 'a'])
        self.assertListEqual(list(v), ['c', 'd', 'a', 'b'])

    def test_invalid_order_by_option(self):
        try:
            v = Vocab(['a'], order_by=None)
            self.fail()
        except Exception as e:
            self.assertIsInstance(e, ValueError)
        try:
            v = Vocab(['a'], order_by='xyz')
            self.fail()
        except Exception as e:
            self.assertIsInstance(e, ValueError)

    def test_padding_is_stored_at_index_0(self):
        v = Vocab(['a'], pad='p')
        self.assertEqual(v.pad, 'p')
        self.assertEqual(v['p'], 0)
        self.assertEqual(v['a'], 1)

    def test_unk_replace_missing_values(self):
        v = Vocab(['a'], unk='u', pad='p')
        self.assertEqual(v.unk, 'u')
        self.assertEqual(v['u'], 1)
        self.assertEqual(v['a'], 2)
        self.assertEqual(v['x'], 1)
        self.assertListEqual(v(['a', 'x', 'x', 'a']), [2, 1, 1, 2])

    def test_eos_appears_at_end_of_list(self):
        v = Vocab(['a'], eos='!', pad=None)
        self.assertEqual(v.eos, '!')
        self.assertListEqual(v(['a']), [1, 0])


if __name__ == '__main__':
    unittest.main()
