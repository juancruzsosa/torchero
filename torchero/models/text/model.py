import torch
from torch.utils.data import DataLoader

from torchero.utils.collate import PadSequenceCollate
from torchero.models import (Model,
                             BinaryClassificationModel,
                             ClassificationModel,
                             RegressionModel)



class TextModel(Model):
    """ Model class that wrap nn.Module models to add
    training, prediction, saving & loading capabilities
    for Natural Language Processing (NLP) tasks
    """
    def __init__(self, model, transform):
        super(TextModel, self).__init__(model)
        self.transform = transform
        self.collate_fn = PadSequenceCollate(pad_value=self.transform.vocab[self.transform.vocab.pad])

    def _create_dataloader(self, *args, **kwargs):
        kwargs['collate_fn'] = kwargs.get("collate_fn") or self.collate_fn
        return DataLoader(*args, **kwargs)

    def input_to_tensor(self, text):
        return self.transform(text)

class BinaryTextClassificationModel(TextModel, BinaryClassificationModel):
    """ Model class for NLP Binary Classification (single or multilabel) tasks.
    E.g: sentiment analysis (without neutral class), Toxicity category of user comments
    """
    def __init__(self, model, transform, use_logits=True, threshold=0.5):
        super(BinaryTextClassificationModel, self).__init__(model=model,
                                                            transform=transform)
        super(TextModel, self).__init__(model=model,
                                        use_logits=use_logits,
                                        threshold=threshold)

class TextClassificationModel(TextModel, ClassificationModel):
    """ Model class for NLP Binary Classification (single or multilabel) tasks.
    E.g: Detect topic of an user comment
    """
    def __init__(self, model, transform, use_softmax=True, threshold=0.5):
        super(TextClassificationModel, self).__init__(model=model,
                                                      transform=transform)
        super(TextModel, self).__init__(model=model,
                                        use_softmax=use_softmax,
                                        threshold=threshold)

class TextRegressionModel(TextModel, RegressionModel):
    """ Model class for NLP Binary Classification (single or multilabel) tasks.
    """
    def __init__(self, model, transform):
        super(TextRegressionModel, self).__init__(model=model,
                                                  transform=transform)
