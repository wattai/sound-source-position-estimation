# -*- coding: utf-8 -*-

import abc

import numpy as np


class SoundSourceLocatorBase(abc.ABC):
    @abc.abstractmethod
    def fit_transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        pass
