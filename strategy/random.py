from typing import List, Generator

import numpy as np
from sklearn.model_selection import train_test_split, KFold

from .base import AbstractSplittingStrategy


class RandomStrategy(AbstractSplittingStrategy):
    def __init__(self, test_size: float):
        super().__init__()
        self.test_size = test_size

    def split(self, data: np.ndarray) -> List[np.ndarray]:
        return train_test_split(data, test_size=self.test_size)

    def generates_many_splits(self) -> bool:
        return False

    def __str__(self):
        return f'Random (ts = {self.test_size:.2f})'


class CrossValidationRandomStrategy(AbstractSplittingStrategy):
    def __init__(self, n_folds: int):
        super().__init__()
        self.n_folds = n_folds

    def split(self, data: np.ndarray) -> Generator[List[np.ndarray], None, None]:
        kf = KFold(n_splits=self.n_folds)
        indexes = [idx for idx in kf.split(data)]

        def res() -> Generator[List[np.ndarray], None, None]:
            for idx in indexes:
                train_idx, test_idx = idx
                yield [data[train_idx, :], data[test_idx, :]]

        return res()

    def generates_many_splits(self) -> bool:
        return True

    def __str__(self):
        return f'CV (n = {self.n_folds})'
