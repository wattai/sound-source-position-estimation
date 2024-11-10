# -*- coding: utf-8 -*-
"""WIP."""

import abc

from pydantic import BaseModel
import numpy as np


def make_signals(num_sources: int, num_mics: int):
    pass


def calc_time_delays_from_position(
    r: float,
    theta: float,
    phi: float,
    c: float = 340.0,
) -> np.ndarray:
    """_summary_

    Args:
        r: Distance.
        theta: Azimuth angle.
        phi: Elevation angle.
        c: Speed of sound. Defaults to 340.0.

    Returns:
        Position.
    """
    positions = (
        r * np.cos(theta) * np.sin(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta),
    )
    tau = np.linalg.vector_norm(positions)
    return tau


class Signal(BaseModel):
    values: list[float]
    sampling_frequency: float


class BaseDevice(abc.ABC):
    pass


class BaseSource(BaseDevice):
    pass


class BaseMicrophone(BaseDevice):
    pass


class BaseEnvironment(abc.ABC):
    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    @abc.abstractmethod
    def sound_speed(self) -> float:
        pass

    @abc.abstractmethod
    def ring_sources(self) -> list[Signal]:
        pass


class Air(BaseEnvironment):
    def __init__(
        self,
        sources: list[BaseSource],
        microphones: list[BaseMicrophone],
        temperature: float = 20.0,
    ):
        self.sources = sources
        self.microphones = microphones
        self._temperature = temperature

    @property
    def sound_speed(self) -> float:
        return 331.3 + 0.6 * self._temperature

    def ring_sources(self) -> list[Signal]:
        return calc_received_signals(
            self.sources,
            self.microphones,
            sound_speed=self.sound_speed,
        )


def calc_received_signals(
    sources: list[BaseSource],
    microphones: list[BaseMicrophone],
    sound_speed: float,
) -> list[Signal]:
    pass
