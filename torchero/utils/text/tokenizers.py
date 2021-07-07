from abc import abstractmethod, ABCMeta

tokenizers = {
    'split': lambda: str.split
}

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

tokenizers.update({
    'spacy': SpacyTokenizer,
    'spacy:en': EnglishSpacyTokenizer,
    'spacy:es': SpanishSpacyTokenizer,
    'spacy:fr': FrenchSpacyTokenizer,
    'spacy:de': GermanSpacyTokenizer,
})

class NLTKTokenizer(object, metaclass=ABCMeta):
    """ NLTK Tokenizer
    """
    def __init__(self):
        try:
            import nltk
        except ImportError:
            raise ImportError("spacy not found. Install it using pip install spacy")
        else:
            self.nltk = nltk

    @abstractmethod
    def __call__(self, text):
        pass

class NLTKWordTokenizer(NLTKTokenizer):
    def __call__(self, text):
        return self.nltk.tokenize.word_tokenize(text)

class NLTKTweetTokenizer(NLTKTokenizer):
    def __init__(self):
        super(NLTKTweetTokenizer, self).__init__()
        self.tokenizer = self.nltk.tokenize.TweetTokenizer()

    def __call__(self, text):
        return self.tokenizer.tokenize(text)

tokenizers.update({'nltk': NLTKWordTokenizer,
                   'nltk:word': NLTKWordTokenizer,
                   'nltk:tweet': NLTKTweetTokenizer})
