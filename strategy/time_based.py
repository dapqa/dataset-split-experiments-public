from typing import List

import numpy as np

from strategy.base import AbstractSplittingStrategy


class TimeBasedSplittingStrategy(AbstractSplittingStrategy):
    def __init__(self, test_size: float, time_column_number: int = 3):
        super().__init__()
        self.test_size = test_size
        self.time_column_number = time_column_number

    def split(self, data: np.ndarray) -> List[np.ndarray]:
        row_count = len(data)
        test_row_count = round(row_count * self.test_size)
        train_row_count = row_count - test_row_count

        sort_idx = np.argsort(data[:, self.time_column_number])
        data = data[sort_idx]

        return [data[0:train_row_count, :], data[train_row_count:, :]]

    def generates_many_splits(self) -> bool:
        return False

    def __str__(self):
        return f'Time (ts = {self.test_size:.2f})'


class TemporalUserSplittingStrategy(AbstractSplittingStrategy):
    def __init__(self, test_size: float, time_column_number: int = 3):
        super().__init__()
        self.test_size = test_size
        self.time_column_number = time_column_number

    def split(self, data: np.ndarray) -> List[np.ndarray]:
        data = data[np.lexsort((data[:, 3], data[:, 0]))]

        train_idx = np.zeros(len(data), dtype='bool')
        test_idx = np.zeros(len(data), dtype='bool')
        prev_user_id = data[0][0]
        i_from = 0
        for i in range(len(data)):
            user_id = data[i][0]
            if user_id != prev_user_id:
                chunk_length = i - i_from
                chuck_test_size = round(chunk_length * self.test_size)
                chuck_train_size = chunk_length - chuck_test_size

                train_idx[i_from:i_from + chuck_train_size] = True
                test_idx[i_from + chuck_train_size:i_from + chunk_length] = True

                i_from = i
                prev_user_id = user_id

        chunk_length = len(data) - i_from
        chuck_test_size = round(chunk_length * self.test_size)
        chuck_train_size = chunk_length - chuck_test_size
        train_idx[i_from:i_from + chuck_train_size] = True
        test_idx[i_from + chuck_train_size:i_from + chunk_length] = True

        return [data[train_idx], data[test_idx]]

    def generates_many_splits(self) -> bool:
        return False

    def __str__(self):
        return f'Time-U (ts = {self.test_size:.2f})'

