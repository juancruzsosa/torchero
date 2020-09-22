tokenizers = {
    'split': str.split
}

try:
    import spacy
except ImportError:
    spacy = None
else:
    def create_spacy_tokenizer(lang):
        blank_model = spacy.blank(lang)
        def tokenizer(text):
            return [token.text for token in blank_model(text)]
        return tokenizer

    tokenizers.update({
        'spacy': create_spacy_tokenizer('en'),
        'spacy:en': create_spacy_tokenizer('en'),
        'spacy:es': create_spacy_tokenizer('es'),
        'spacy:fr': create_spacy_tokenizer('fr'),
        'spacy:de': create_spacy_tokenizer('de'),
    })

try:
    import nltk
except ImportError:
    nltk = None
else:
    from nltk.tokenize import word_tokenize, TweetTokenizer
    tokenizers.update({'nltk': word_tokenize,
                       'nltk:word': word_tokenize,
                       'nltk:tweet': TweetTokenizer().tokenize})
