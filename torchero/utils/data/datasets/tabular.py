from torchero.utils.data.datasets import Dataset

import pandas as pd

class TabularDataset(Dataset):
    """ Dataset for labeled text
    """
    @classmethod
    def from_json(
        cls,
        path,
        field_names,
        target_field_names=None,
        transform=None,
        target_transform=None
    ):
        """ Creates an instance from a json file (with list of dict fields scheme).

        Arguments:
            path (str or Path): Path of the json file
            field_name (str or int): If a string is passed the column name for the
                texts. If a int is passed the index of the column for the text
            target_file_name (str, int, list of str or list of str): If a
                string is passed the column name for the target. If a int is
                passed the index of the column for the target
            transforms (callable, optional): Transform functions for list of fields
            target_transform (callable, optional): Transform function for the list of target fields
        """
        squeeze = isinstance(field_names, (int, str))
        if squeeze:
            field_names = [field_names]
        if target_field_names is not None:
            squeeze_targets = isinstance(target_field_names, (int, str))
            if squeeze_targets:
                target_field_names = [target_field_names]
        else:
            target_field_names = []
        records = pd.read_json(jsonfile)
        data = records[field_names]
        if squeeze:
            data = data[0]
        if target_field_names is not None:
            target_data = records[target_field_names]
            if squeeze_targets:
                target_data = target_data[0]
        else:
            target_data = None
        return cls(
            data=data,
            target_data=target_data,
            transform=transform,
            target_transform=target_transform
        )

    @classmethod
    def from_csv(
        cls,
        path,
        columns,
        target_columns=None,
        delimiter=",",
        quotechar='"',
        has_header=True,
        column_names=None,
        compression='infer',
        transform=None,
        target_transform=None,
    ):
        """ Creates an instance from a csv file

        Arguments:
            path (str or Path): Path of the csv file
            columns (tuple of str or int): Columns names (or column indices to
                use) for the input data
            target_columns (tuple of str or int): Column names (or
                column-indices to use) for the target data
            delimiter (str): Character used to splits the csv fields
            quotechar (str): Character used to delimit the text strings
            has_header (bool): True if the csv contains a header. False,
                otherwise
            column_names (list): List of columns names
            compression (str): Compression method for the csv. Set to 'infer'
                to infer it from the extension
            transform (callable, optional): Transform functions for the row
            target_transform (callable, optional): Transform functions for the
                row target
        """
        squeeze = isinstance(columns, (int, str))
        if squeeze:
            columns = [columns]
        if target_columns is not None:
            squeeze_targets = isinstance(target_columns, (int, str))
            if squeeze_targets:
                target_columns = [target_columns]
        else:
            squeeze_targets = False
            target_columns = []
        records = pd.read_csv(path,
                              usecols=columns + target_columns,
                              delimiter=delimiter,
                              quotechar=quotechar,
                              names=column_names,
                              compression=compression)
        data = records[columns]
        if target_columns:
            target_data = records[target_columns]
        else:
            target_data = None
        return cls(data,
                   target_data,
                   transform,
                   target_transform,
                   squeeze_data=squeeze,
                   squeeze_targets=squeeze_targets)

    def __init__(
        self,
        data,
        target_data=None,
        transform=None,
        target_transform=None,
        squeeze_data=True,
        squeeze_targets=True,
    ):
        """ Constructor

        Arguments:
            data (iter-like, pd.DataFrame, pd.Series): Input samples.
            targets (iter-like, pd.DataFrame, pd.Series): Targets for each sample.
            transform (callable, optional): Transform functions for the input samples
            target_transform (callable, optional): Transform functions for the targets
            squeeze_data (bool): When set to True if the data is a single column every item of the
                dataset returns the column value instead of a tuple.
            squeeze_targets (bool): When set to True if the target is a single column every item of the
                dataset returns the value instead of a tuple.
        """
        self.data = pd.DataFrame(data).reset_index(drop=True)
        self.target_data = None
        if target_data is not None:
            self.target_data = pd.DataFrame(target_data).reset_index(drop=True)
            if len(self.target_data) != len(self.data):
                raise ValueError("Target data should contain the same number of rows as data")
        self.transform = transform
        self.target_transform = target_transform
        self.squeeze_data = squeeze_data
        self.squeeze_targets = squeeze_targets

    def __len__(self):
        return len(self.data)

    @property
    def names(self):
        """ Names of the columns for the input samples
        """
        return list(self.data.columns)

    def __getitem__(self, i):
        X = self.data.iloc[i]
        if self.squeeze_data and len(X) == 1:
            X = X[0]
        else:
            X = tuple(X)
        if self.transform is not None:
            X = self.transform(X)
        if self.target_data is not None:
            y = self.target_data.iloc[i]
            if self.squeeze_targets and len(y) == 1:
                y = y[0]
            else:
                y = tuple(y)
            if self.target_transform is not None:
                y= self.target_transform(y)
            return X, y
        else:
            return X

    def show_samples(self, k=10):
        sample = self.data.sample(k)
        if self.target_data is not None:
            sample = pd.concat([sample, self.target_data.loc[sample.index]], axis=1)
        return sample

