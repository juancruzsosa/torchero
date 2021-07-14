import pickle

import torch
from torch.utils.data import DataLoader

from torchero.utils.collate import PadSequenceCollate
from torchero.utils.text import TextTransform
from torchero.models import (Model,
                             BinaryClassificationModel,
                             ClassificationModel,
                             RegressionModel)



class TextModel(Model):
    """ Model class that wrap nn.Module models to add
    training, prediction, saving & loading capabilities
    for Natural Language Processing (NLP) tasks
    """
    def __init__(self, model, transform=None):
        super(TextModel, self).__init__(model)
        self.transform = transform

    def _create_dataloader(self, *args, **kwargs):
        if self.transform is not None:
            collate_fn = self.collate_fn = PadSequenceCollate(pad_value=self.transform.vocab[self.transform.vocab.pad])
            kwargs['collate_fn'] = kwargs.get("collate_fn") or collate_fn
        return DataLoader(*args, **kwargs)

    def input_to_tensor(self, text):
        if self.transform is not None:
            return self.transform(text)
        else:
            return text

    def _save_to_zip(self, zip_fp):
        super(TextModel, self)._save_to_zip(zip_fp)
        with zip_fp.open('transform.pkl', 'w') as fp:
            pickle.dump(self.transform, fp)
        
    def _load_from_zip(self, zip_fp):
        super(TextModel, self)._load_from_zip(zip_fp)
        with zip_fp.open('transform.pkl', 'r') as fp:
            self.transform = pickle.load(fp)

class BinaryTextClassificationModel(TextModel, BinaryClassificationModel):
    """ Model class for NLP Binary Classification (single or multilabel) tasks.
    E.g: sentiment analysis (without neutral class), Toxicity category of user comments
    """
    def __init__(self, model, transform=None, use_logits=True, threshold=0.5):
        super(BinaryTextClassificationModel, self).__init__(model=model,
                                                            transform=transform)
        super(TextModel, self).__init__(model=model,
                                        use_logits=use_logits,
                                        threshold=threshold)

class TextClassificationModel(TextModel, ClassificationModel):
    """ Model class for NLP Binary Classification (single or multilabel) tasks.
    E.g: Detect topic of an user comment
    """
    def __init__(self, model, transform=None, use_softmax=True, threshold=0.5):
        super(TextClassificationModel, self).__init__(model=model,
                                                      transform=transform)
        super(TextModel, self).__init__(model=model,
                                        use_softmax=use_softmax,
                                        threshold=threshold)

class TextRegressionModel(TextModel, RegressionModel):
    """ Model class for NLP Binary Classification (single or multilabel) tasks.
    """
    def __init__(self, model, transform=None):
        super(TextRegressionModel, self).__init__(model=model,
                                                  transform=transform)
