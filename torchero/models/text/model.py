import pickle

import torch
from torch.utils.data import DataLoader

from torchero.utils.collate import PadSequenceCollate
from torchero.models import (Model,
                             BinaryClassificationModel,
                             ClassificationModel,
                             RegressionModel)



class TextModel(Model):
    """ Model class that wraps nn.Module models to add
    training, prediction, saving & loading capabilities
    for Natural Language Processing (NLP) tasks
    """
    def __init__(self, model, transform=None):
        super(TextModel, self).__init__(model)
        self.transform = transform

    def _create_dataloader(self, *args, **kwargs):
        if self.transform is not None:
            collate_fn = PadSequenceCollate(pad_value=self.transform.vocab[self.transform.vocab.pad])
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

    def predict(self, ds, batch_size=None, to_tensor=True, has_targets=False, num_workers=0, prefetch_factor=2):
        if isinstance(ds, str):
            return self.predict([ds])[0]
        else:
            return super(TextModel, self).predict(ds,
                                                  batch_size=batch_size,
                                                  to_tensor=to_tensor,
                                                  has_targets=has_targets,
                                                  num_workers=num_workers,
                                                  prefetch_factor=prefetch_factor)


class BinaryTextClassificationModel(TextModel, BinaryClassificationModel):
    """ Model class for NLP Binary Classification (single or multilabel) tasks.
    E.g: sentiment analysis (without neutral class), Toxicity category of user comments
    """
    def __init__(self, model, transform=None, use_logits=True, threshold=0.5, labels=None):
        super(BinaryTextClassificationModel, self).__init__(model=model,
                                                            transform=transform)
        super(TextModel, self).__init__(model=model,
                                        use_logits=use_logits,
                                        threshold=threshold,
                                        labels=labels)

class TextClassificationModel(TextModel, ClassificationModel):
    """ Model class for NLP Binary Classification (single or multilabel) tasks.
    E.g: Detect topic of an user comment
    """
    def __init__(self, model, transform=None, use_softmax=True, classes=None):
        super(TextClassificationModel, self).__init__(model=model,
                                                      transform=transform)
        super(TextModel, self).__init__(model=model,
                                        use_softmax=use_softmax,
                                        classes=classes)

class TextRegressionModel(TextModel, RegressionModel):
    """ Model class for NLP Binary Classification (single or multilabel) tasks.
    """
    def __init__(self, model, transform=None):
        super(TextRegressionModel, self).__init__(model=model,
                                                  transform=transform)
