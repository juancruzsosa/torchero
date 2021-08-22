import unicodedata
from abc import abstractmethod, ABCMeta

from torchero.utils.text.transforms.preprocessing import convert_to_unicode

__all__ = ['WhitespaceTokenizer',
           'SpacyTokenizer',
           'EnglishSpacyTokenizer',
           'SpanishSpacyTokenizer',
           'FrenchSpacyTokenizer',
           'GermanSpacyTokenizer',
           'BasicTokenizer',
           'NLTKWordTokenizer',
           'NLTKTweetTokenizer']

class WhitespaceTokenizer(object):
    """ Breaks text into words on spaces
    """

    def __call__(self, text):
        text = text.strip()
        if not text:
            return []
        tokens = text.split()
        return tokens

class SpacyTokenizer(object):
    """ Spacy tokenizer
    """
    def __init__(self, lang='en'):
        try:
            import spacy
        except ImportError:
            raise ImportError("spacy not found. Install it using pip install spacy")
        self.lang = lang
        self.model = spacy.blank(lang)

    def __call__(self, text):
        return [token.text for token in self.model(text)]

    def __getstate__(self):
        return {'lang': self.lang}

    def __setstate__(self, state):
        self.lang = state['lang']
        try:
            import spacy
        except ImportError:
            raise ImportError("spacy not found. Install it using pip install spacy")
        self.model = spacy.blank(self.lang)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.lang)


class EnglishSpacyTokenizer(SpacyTokenizer):
    pass

class SpanishSpacyTokenizer(SpacyTokenizer):
    def __init__(self):
        return super(SpanishSpacyTokenizer, self).__init__('es')

class FrenchSpacyTokenizer(SpacyTokenizer):
    def __init__(self):
        return super(FrenchSpacyTokenizer, self).__init__('fr')

class GermanSpacyTokenizer(SpacyTokenizer):
    def __init__(self):
        return super(GermanSpacyTokenizer, self).__init__('de')

class NLTKTokenizer(object, metaclass=ABCMeta):
    """ NLTK Tokenizer
    """
    def __init__(self):
        try:
            import nltk
        except ImportError:
            raise ImportError("nltk not found. Install it using pip install nltk")
        else:
            self.nltk = nltk

    @abstractmethod
    def __call__(self, text):
        pass

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        try:
            import nltk
        except ImportError:
            raise ImportError("nltk not found. Install it using pip install nltk")
        else:
            self.nltk = nltk

class NLTKWordTokenizer(NLTKTokenizer):
    def __call__(self, text):
        return self.nltk.tokenize.word_tokenize(text)

class NLTKTweetTokenizer(NLTKTokenizer):
    def __init__(self):
        super(NLTKTweetTokenizer, self).__init__()
        self.tokenizer = self.nltk.tokenize.TweetTokenizer()

    def __call__(self, text):
        return self.tokenizer.tokenize(text)

    def __getstate__(self):
        return {'tokenizer': self.tokenizer}

    def __setstate__(self, d):
        super(NLTKTweetTokenizer, self).__setstate__(d)
        self.tokenizer = d['tokenizer']

## adapted from https://github.com/google-research/bert/blob/master/tokenization.py

def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self):
        self.whitespace_tokenize = WhitespaceTokenizer()

    def __call__(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)

        orig_tokens = self.whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = self.whitespace_tokenize(" ".join(split_tokens))
        return output_tokens


    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)
