import tempfile
import pickle
import unittest

import spacy

from collections import OrderedDict
from torchero.utils.text.transforms import Vocab, OutOfVocabularyError
from torchero.utils.text.transforms import tokenizers
from torchero.utils.text.transforms import Compose
from torchero.utils.text.transforms import LeftTruncator, RightTruncator, CenterTruncator
from torchero.utils.text.transforms import basic_text_transform
from torchero.utils.text.transforms import (convert_to_unicode,
                                            strip_accents,
                                            strip_tags,
                                            strip_numeric)

from .common import *

class BasicComposeTransformsTests(unittest.TestCase):
    def assertTensorsEqual(self, a, b):
        return self.assertEqual(a.tolist(), b.tolist())

    def setUp(self):
        self.vocab = Vocab()
        self.transform = Compose(
            pre=str.lower,
            tok=str.split,
            vocab=self.vocab,
        )
        self.transform.fit(['a b c'])

    def test_invalid_assert_pipeline_invalid_stepname_raises_error(self):
        try:
            compose = Compose(fit=str.lower)
            self.fail()
        except Exception as e:
            self.assertIsInstance(e, ValueError)
            self.assertEqual(str(e), "Invalid step_name 'fit'. Choose another name")

    def test_basic_text_transform(self):
        transform = basic_text_transform(pre_process=str.lower,
                                         tokenizer='spacy',
                                         vocab_max_size=10,
                                         vocab_min_count=1,
                                         max_len=5,
                                         truncate_mode='center')
        transform.fit(["This is the first text", "I arrive Second"])
        self.assertTensorsEqual(transform("This is the second text"), torch.tensor([1,2,3,8,5]))

    def test_compose_has_all_the_members(self):
        self.assertEqual(len(self.transform), 3)
        self.assertTrue(hasattr(self.transform, 'pre'))
        self.assertIs(self.transform.pre, str.lower)
        self.assertIs(self.transform[0], str.lower)
        self.assertTrue(hasattr(self.transform, 'tok'))
        self.assertIs(self.transform.tok, str.split)
        self.assertIs(self.transform[1], str.split)
        self.assertTrue(hasattr(self.transform, 'vocab'))
        self.assertIs(self.transform.vocab, self.vocab)
        self.assertIs(self.transform[2], self.vocab)

    def test_compose_call_process(self):
        self.assertEqual(self.transform('c b a'), [3, 2, 1])

    def test_creation_from_dict(self):
        transform = Compose.from_dict({'pre': str.lower,
                                       'tok': str.split,
                                       'vocab': self.vocab})
        self.assertIs(self.transform.pre, str.lower)
        self.assertIs(self.transform.tok, str.split)
        self.assertIs(self.transform.vocab, self.vocab)

    def test_compose_can_be_pickled(self):
        with tempfile.NamedTemporaryFile() as tmp_file:
            with open(tmp_file.name, 'w+b') as fp:
                pickle.dump(self.transform, fp)
            with open(tmp_file.name, 'rb') as fp:
                transform_2 = pickle.load(fp)
                self.assertIs(transform_2.pre, str.lower)
                self.assertIs(transform_2.tok, str.split)
                self.assertIsInstance(transform_2.vocab, Vocab)
                self.assertEqual(self.transform.vocab['a'], 1)
                self.assertEqual(self.transform.vocab['b'], 2)
                self.assertEqual(self.transform.vocab['c'], 3)
                self.assertEqual(self.transform('b c a'), [2, 3, 1])

class TokenizerTests(unittest.TestCase):
    def test_english_tokenizer(self):
        tokenizer = tokenizers.EnglishSpacyTokenizer()
        self.assertEqual(tokenizer(''), [])
        self.assertEqual(tokenizer('This is a simple text to tokenize'), ['This', 'is', 'a', 'simple', 'text', 'to', 'tokenize'])
        self.assertEqual(tokenizer('She was upset when it didn\'t boil!, #water'), ['She', 'was', 'upset', 'when', 'it', 'did', 'n\'t', 'boil', '!', ',', '#', 'water'])

    def test_spanish_tokenizer(self):
        tokenizer = tokenizers.SpanishSpacyTokenizer()
        self.assertEqual(tokenizer(''), [])
        self.assertEqual(tokenizer('Ella es un buena doctora.He\'s not'), ['Ella', 'es', 'un', 'buena', 'doctora', '.', 'He\'s', 'not'])

    def test_german_tokenizer(self):
        tokenizer = tokenizers.GermanSpacyTokenizer()
        self.assertEqual(tokenizer(''), [])
        self.assertEqual(tokenizer('Drück mir die Daumen'), ['Drück', 'mir', 'die', 'Daumen'])

    def test_nltk_word_tokenizer(self):
        tokenizer = tokenizers.NLTKWordTokenizer()
        self.assertEqual(tokenizer(''), [])
        self.assertEqual(tokenizer('This is a simple text to tokenize'), ['This', 'is', 'a', 'simple', 'text', 'to', 'tokenize'])
        self.assertEqual(tokenizer('She was upset when it didn\'t boil!, #water'), ['She', 'was', 'upset', 'when', 'it', 'did', 'n\'t', 'boil', '!', ',', '#', 'water'])

    def test_nltk_tweet_tokenizer(self):
        tokenizer = tokenizers.NLTKTweetTokenizer()
        self.assertEqual(tokenizer(''), [])
        self.assertEqual(tokenizer('This is a simple text to tokenize'), ['This', 'is', 'a', 'simple', 'text', 'to', 'tokenize'])
        self.assertEqual(tokenizer('She was upset when it didn\'t boil!, #water'), ['She', 'was', 'upset', 'when', 'it', 'didn\'t', 'boil', '!', ',', '#water'])

    def test_tokenizers_can_be_pickled(self):
        for tokenizer_class in (tokenizers.EnglishSpacyTokenizer,
                                tokenizers.SpanishSpacyTokenizer,
                                tokenizers.GermanSpacyTokenizer,
                                tokenizers.NLTKWordTokenizer,
                                tokenizers.NLTKTweetTokenizer):
            tokenizer = tokenizer_class()
            with self.subTest(tokenizer=tokenizer.__class__.__name__):
                with tempfile.NamedTemporaryFile() as tmp_file:
                    with open(tmp_file.name, 'w+b') as fp:
                        pickle.dump(tokenizer, fp)
                    with open(tmp_file.name, 'rb') as fp:
                        tokenizer = pickle.load(fp)
                    self.assertEqual(tokenizer('a b'), ['a', 'b'])

class TruncatorTests(unittest.TestCase):
    def setUp(self):
        self.short_text = ['Very', 'short', 'text']
        self.long_text = ['This', 'is', 'a', 'very', 'long', 'text', 'to', 'test']

    def tests_left_truncate(self):
        truncator = LeftTruncator(4)
        self.assertEqual(truncator.max_len, 4)
        self.assertEqual(truncator(self.short_text), ['Very', 'short', 'text'])
        self.assertEqual(truncator(self.long_text), ['This', 'is', 'a', 'very'])

    def tests_right_truncate(self):
        truncator = RightTruncator(4)
        self.assertEqual(truncator.max_len, 4)
        self.assertEqual(truncator(self.short_text), ['Very', 'short', 'text'])
        self.assertEqual(truncator(self.long_text), ['long', 'text', 'to', 'test'])

    def tests_center_truncate(self):
        truncator = CenterTruncator(4)
        self.assertEqual(truncator.max_len, 4)
        self.assertEqual(truncator(self.short_text), ['Very', 'short', 'text'])
        self.assertEqual(truncator(self.long_text), ['a', 'very', 'long', 'text'])

    def test_truncators_can_be_pickled(self):
        for truncator_class  in (LeftTruncator, RightTruncator, CenterTruncator):
            truncator = truncator_class(4)
            with self.subTest(truncator=truncator_class.__name__):
                with tempfile.NamedTemporaryFile() as tmp_file:
                    with open(tmp_file.name, 'w+b') as fp:
                        pickle.dump(truncator, fp)
                    with open(tmp_file.name, 'rb') as fp:
                        truncator = pickle.load(fp)
                self.assertEqual(truncator.max_len, 4)
                self.assertEqual(truncator(self.short_text), ['Very', 'short', 'text'])

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

class PreprocessingTests(unittest.TestCase):
    def test_convert_to_unicode(self):
        self.assertEqual(convert_to_unicode('áÉ'.encode('utf-8')), 'áÉ')

    def test_strip_accents(self):
        self.assertEqual(strip_accents('abcdefghijklmnñopqrstuvwxyz '
                                       'ÀÁÂÃÄÅÇÈÉÊËÌÍÎÏÒÓÔÕÖÙÚÛÜ '
                                       'àáâãäåèéêëìíîïðòñóôõöùúûü'),
                        'abcdefghijklmnnopqrstuvwxyz '
                        'AAAAAACEEEEIIIIOOOOOUUUU '
                        'aaaaaaeeeeiiiiðonoooouuuu')

    def test_strip_tags(self):
        self.assertEqual(strip_tags('text <b>bold text</b>'), 'text bold text')

    def test_strip_numeric(self):
        self.assertEqual(strip_numeric('abc 123 def -45 ghi 67.8 9a'), 'abc  def - ghi  a')
