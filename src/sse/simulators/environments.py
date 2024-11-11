# -*- coding: utf-8 -*-
"""WIP."""

import abc

from pydantic import BaseModel
import numpy as np
import soundfile as sf


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


class Position3D(BaseModel):
    r: float
    theta: float
    phi: float


class BaseDevice(abc.ABC):
    pass


class BaseSource(BaseDevice):
    @abc.abstructmethod
    def ring(self) -> Signal:
        pass


class Source(BaseSource):
    def __init__(
        self,
        position: Position3D,
        signal: Signal,
    ):
        self.signal = signal
        self.position = position

    def ring(self) -> Signal:
        return self.signal


def file_to_signals(filepath: str) -> list[Signal]:
    return sf.read(filepath)


class BaseSignalGenerator(abc.ABC):
    @abc.abstractmethod
    def generate(self, sampling_frequency: float) -> Signal:
        pass


class SineSignalGenerator(BaseSignalGenerator):
    def __init__(
        self,
        frequency: float,
    ):
        self.frequeny = frequency

    def generate(self, sampling_frequency: float) -> Signal:
        return np.sin()


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
