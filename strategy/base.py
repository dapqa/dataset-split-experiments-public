from abc import ABCMeta, abstractmethod
from typing import Union, List, Generator

import numpy as np


class AbstractSplittingStrategy(metaclass=ABCMeta):
    @abstractmethod
    def split(self, data: np.ndarray) -> Union[List[np.ndarray], Generator[List[np.ndarray], None, None]]:
        pass

    @abstractmethod
    def generates_many_splits(self) -> bool:
        pass
