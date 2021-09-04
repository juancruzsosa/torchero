from abc import ABCMeta, abstractmethod

from torch.utils.data import Dataset

class TorchtextClassificationDataset(Dataset, metaclass=ABCMeta):
    """ Base class for torchero dataset classes
    """
    @abstractmethod
    def _create(*args, **kwargs):
        pass

    default_target_transform = None

    def __init__(self,
                 train=True,
                 transform=None,
                 target_transform=None,
                 root='.data'):
        """ Constructor

        Arguments:
            train (bool): If ``True``, creates a dataset from the train split, otherwise from the test split.
            transform (callable, optional): A function/transform that transforms the texts
            target_transform (callable, optional): A function/transform that takes in the target and transforms it.
            root (str): Root directory of dataset where the dataset
                exists or will be downloaded.
        """
        try:
            from torchtext.data.functional import to_map_style_dataset
        except ImportError:
            raise ImportError("torchtext not fond. Run pip install torchtext")
        self.ds = self._create(root, split='train' if train else 'test')
        self.ds = to_map_style_dataset(self.ds)
        self.transform = transform
        self.target_transform = target_transform or self.default_target_transform

    def __getitem__(self, idx):
        target, text = self.ds[idx]
        if self.transform is not None:
            text = self.transform(text)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return text, target

    def texts(self):
        return (self.ds[i][1] for i in range(len(self.ds)))

    def __len__(self):
        return len(self.ds)

class AGNews(TorchtextClassificationDataset):
    """ AG News dataset

    Number of classes:
        4

    Classes:
        World, Sports, Business, Sci/Tech

    Size:
        - train: 120000
        - test:    7600
    """
    classes = [
       "World", 
       "Sports",
       "Business",
       "Sci/Tech"
    ]

    @staticmethod
    def default_target_transform(y):
        return y-1

    def _create(self, root, split):
        from torchtext import datasets
        return datasets.AG_NEWS(root=root, split=split)

class SogouNews(TorchtextClassificationDataset):
    """ SogouCA and SogouCS news dataset, in 5
    Note that the Chinese characters have been converted to Pinyin.

    Number of classes:
        5

    Classes:
        World, Sports, Business, Sci/Tech

    Size:
        - train: 450000
        - test:   60000
    """

    classes = [
        "sports",
        "finance",
        "entertainment",
        "automobile",
        "technology",
    ]

    @staticmethod
    def default_target_transform(y):
        return y-1

    def _create(self, root, split):
        from torchtext import datasets
        return datasets.SogouNews(root=root, split=split)

class DBpedia(TorchtextClassificationDataset):
    """ DBpedia (from "DB" for "database") is a project aiming to extract
    structured content from the information created in the Wikipedia project.

    Number of classes:
        14

    Classes:
        Company, EducationalInstitution, Artist, Athlete, OfficeHolder,
        MeanOfTransportation, Building, NaturalPlace, Village, Animal, Plant,
        Album, Film, WrittenWork

    Size:
        - train: 560000
        - test:   70000
    """

    classes = [
        "Company",
        "EducationalInstitution",
        "Artist",
        "Athlete",
        "OfficeHolder",
        "MeanOfTransportation",
        "Building",
        "NaturalPlace",
        "Village",
        "Animal",
        "Plant",
        "Album",
        "Film",
        "WrittenWork"
    ]

    @staticmethod
    def default_target_transform(y):
        return y-1

    def _create(self, root, split):
        from torchtext import datasets
        return datasets.DBpedia(root=root, split=split)

class YelpReviewPolarity(TorchtextClassificationDataset):
    """ Yelp Reviews Sentiment Analysis dataset

    Number of classes:
        2

    Classes:
        neg, pos

    Size:
        - train: 560000
        - test:   38000
    """
    classes = [
        'neg',
        'pos'
    ]

    @staticmethod
    def default_target_transform(y):
        return y-1

    def _create(self, root, split):
        from torchtext import datasets
        return datasets.YelpReviewPolarity(root=root, split=split)

class YelpReviewFull(TorchtextClassificationDataset):
    """ Yelp Full Reviews Sentiment Analysis dataset

    Number of classes:
        5

    Classes:
        1, 2, 3, 4, 5

    Size:
        - train: 650000
        - test:   50000
    """

    classes = [
        "1",
        "2",
        "3",
        "4",
        "5"
    ]

    @staticmethod
    def default_target_transform(y):
        return y-1

    def _create(self, root, split):
        from torchtext import datasets
        return datasets.YelpReviewFull(root=root, split=split)

class YahooAnswers(TorchtextClassificationDataset):
    """ Yahoo! Answers Topic Classification Dataset

    Number of classes:
        10

    Classes:
        Society & Culture, Science & Mathematics, Health, Education &
        Reference, Computers & Internet, Sports, Business & Finance,
        Entertainment & Music, Family & Relationships, Politics & Government

    Size:
        - train: 1400000
        - test:    60000
    """
    classes = [
        "Society & Culture",
        "Science & Mathematics",
        "Health",
        "Education & Reference",
        "Computers & Internet",
        "Sports",
        "Business & Finance",
        "Entertainment & Music",
        "Family & Relationships",
        "Politics & Government",
    ]

    @staticmethod
    def default_target_transform(y):
        return y-1

    def _create(self, root, split):
        from torchtext import datasets
        return datasets.YahooAnswers(root=root, split=split)

class AmazonReviewPolarity(TorchtextClassificationDataset):
    """ The Amazon reviews dataset consists of reviews from amazon

    Number of classes:
        2

    Classes:
        neg, pos

    Size:
        - train: 3600000
        - test:   400000
    """
    classes = ['neg', 'pos']

    @staticmethod
    def default_target_transform(y):
        return y-1

    def _create(self, root, split):
        from torchtext import datasets
        return datasets.AmazonReviewPolarity(root=root, split=split)

class IMDB(TorchtextClassificationDataset):
    """ Large Movie Review Dataset

    Number of classes:
        2

    Classes:
        neg, pos

    Size:
        - train: 25000
        - test: 25000
    """
    classes = ['neg', 'pos']

    @staticmethod
    def default_target_transform(y):
        return 0 if y == 'neg' else 1

    def _create(self, root, split):
        from torchtext import datasets
        return datasets.IMDB(root=root, split=split)
