from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class BaseDetector(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def detect(self, image: np.ndarray) -> list:
        pass
