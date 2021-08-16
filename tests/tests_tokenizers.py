import spacy
import unittest
from torchero.utils.text.transforms import tokenizers

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
