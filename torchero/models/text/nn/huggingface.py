import torch
from torch import nn
from torchero.utils.text import transforms

try:
    import transformers
except ImportError:
    transformers = True
    IMPORT_ERROR_MESSAGE = "transformers library not installed. Install it using pip install transformers"

__all__ = ['HFAutoModelForTextClassification']

class HFTokenizer(object):
    def __init__(self, tokenizer, truncate=True, max_length=True):
        self.tokenizer = tokenizer
        self.truncate = truncate
        self.max_length = max_length

    def __call__(self, text):
        return self.tokenizer.encode(truncate=self.truncate, max_length=self.max_length)

class HFAutoModelForTextClassification(nn.Module):
    arch_names = ['bert-base-uncased',
                  'bert-base-cased',
                  'roberta-large',
                  'distilbert-base-cased',
                  'distilbert-base-uncased']

    body_name = {
        'bert-base-uncased': 'bert',
        'bert-base-cased': 'bert',
        'roberta-base': 'roberta',
        'roberta-large': 'roberta',
        'distilbert-base-uncased': 'distilbert',
        'distilbert-base-cased': 'distilbert',
    }

    @classmethod
    def from_config(cls, config):
        return cls(arch=config['arch'],
                   num_outputs=config['num_outputs'])

    def __init__(self, num_outputs, arch='bert-base-uncased'):
        super(HFAutoModelForTextClassification, self).__init__()
        self.arch = arch
        self.num_outputs = num_outputs
        if transformers is None:
            raise ImportError(IMPORT_ERROR_MESSAGE)
        self.transformer = transformers.AutoModelForSequenceClassification.from_pretrained(arch, num_labels=num_outputs)

    @property
    def config(self):
        return {'arch': self.arch,
                'num_outputs': self.num_outputs}

    def body_parameters(self):
        return getattr(self.transformer, self.body_name[self.arch]).parameters()

    def freeze_body(self):
        for param in self.body_parameters():
            param.requires_grad = False

    def unfreeze_body(self):
        for param in self.body_parameters():
            param.requires_grad = True

    def transform(self, max_len=128):
        if transformers is None:
            raise ImportError(IMPORT_ERROR_MESSAGE)
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.arch, model_max_length=max_len)
        return transforms.Compose(encode=HFTokenizer(tokenizer),
                                  to_tensor=transforms.ToTensor)

    def forward(self, texts, lens):
        max_len = texts.shape[1]
        src_mask = (torch.arange(max_len)[None, :].to(texts.device) < lens[:, None]).float()
        return self.transformer(input_ids=texts,
                                attention_mask=src_mask).logits
