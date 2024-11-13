# -*- coding: utf-8 -*-
"""WIP."""

import abc

from pydantic import BaseModel
import numpy as np
import scipy as sp
import soundfile as sf


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
    @abc.abstractmethod
    def ring(self) -> Signal:
        pass


class Source(BaseSource):
    def __init__(
        self,
        position: Position3D,
        signal: Signal,
    ):
        self.position = position
        self.signal = signal

    def ring(self) -> Signal:
        return self.signal


def file_to_signals(filepath: str) -> list[Signal]:
    return sf.read(filepath)


class BaseSignalGenerator(abc.ABC):
    @abc.abstractmethod
    def generate(self, sampling_frequency: float, time_length: float) -> Signal:
        pass


class SineSignalGenerator(BaseSignalGenerator):
    def __init__(
        self,
        frequency: float,
    ):
        self.frequency = frequency

    def generate(self, sampling_frequency: float, time_length: float) -> Signal:
        time = make_time_sequence(sampling_frequency, time_length)
        return Signal(
            values=list(np.sin(2 * np.pi * self.frequency * time)),
            sampling_frequency=sampling_frequency,
        )


def make_time_sequence(sampling_frequency: float, time_length: float) -> np.ndarray:
    return np.linspace(
        0,
        int(sampling_frequency * time_length),
        int(sampling_frequency * time_length) - 1,
    )


class BaseMicrophone(BaseDevice):
    @abc.abstractmethod
    def record(self, signals: list[Signal]) -> Signal:
        pass


class Microphone(BaseMicrophone):
    def __init__(
        self,
        position: Position3D,
        sampling_frequency: float,
    ):
        self.position = position
        self.sampling_frequency = sampling_frequency

    def record(self, signals: list[Signal]) -> Signal:
        return overlap(signals, self.sampling_frequency)


def overlap(signals: list[Signal], sampling_frequency: float) -> Signal:
    resampled = [resample(s, sampling_frequency) for s in signals]
    len_longest = max(len(s.values) for s in signals)
    overlapped_array = np.zeros(len_longest)
    for signal in resampled:
        overlapped_array[: len(signal.values)] += np.array(signal.values)
    return Signal(
        values=list(overlapped_array),
        sampling_frequency=sampling_frequency,
    )


def resample(signal: Signal, new_sampling_frequency: float) -> Signal:
    # 変換するサンプル数を計算
    num_samples = int(
        round(len(signal.values) * (new_sampling_frequency / signal.sampling_frequency))
    )

    # 'num' を整数にキャスト
    return Signal(
        values=list(sp.signal.resample(signal.values, num_samples)),
        sampling_frequency=new_sampling_frequency,
    )


class BaseMedium(abc.ABC):
    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    @abc.abstractmethod
    def sound_speed(self) -> float:
        pass


class Air(BaseMedium):
    def __init__(
        self,
        temperature: float = 20.0,
    ):
        self._temperature = temperature

    @property
    def sound_speed(self) -> float:
        return 331.3 + 0.6 * self._temperature


class BaseObserver(abc.ABC):
    @abc.abstractmethod
    def ring_sources(self) -> list[Signal]:
        pass


class Observer(BaseObserver):
    def __init__(
        self,
        sources: list[BaseSource],
        microphones: list[BaseMicrophone],
        medium: BaseMedium,
    ):
        self.sources = sources
        self.microphones = microphones
        self.medium = medium

    def ring_sources(self) -> list[Signal]:
        return calc_received_signals(
            self.sources,
            self.microphones,
            sound_speed=self.medium.sound_speed,
        )


def calc_received_signals(
    sources: list[BaseSource],
    microphones: list[BaseMicrophone],
    sound_speed: float,
) -> list[Signal]:
    return [
        mic.record(
            [
                delay(
                    signal=source.ring(),
                    distance=calc_distance(mic.position, source.position),
                    sound_speed=sound_speed,
                )
                for source in sources
            ],
        )
        for mic in microphones
    ]


def calc_distance(p1: Position3D, p2: Position3D) -> float:
    return euclidean_distance(
        p1=(p1.r, p1.theta, p1.phi),
        p2=(p2.r, p2.theta, p2.phi),
    )


def polar_to_cartesian(r, theta, phi):
    """極座標をデカルト座標に変換"""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def euclidean_distance(p1, p2):
    """2つの点間のユークリッド距離を計算"""
    (r1, theta1, phi1) = p1
    (r2, theta2, phi2) = p2

    x1, y1, z1 = polar_to_cartesian(r1, theta1, phi1)
    x2, y2, z2 = polar_to_cartesian(r2, theta2, phi2)

    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return distance


def delay(signal: Signal, distance: float, sound_speed: float) -> Signal:
    # num_delay_points を整数にキャスト
    num_delay_points = int(round(signal.sampling_frequency * (distance / sound_speed)))

    # スライシングに整数を使用
    return Signal(
        values=np.pad(signal.values[num_delay_points:], (0, num_delay_points)),
        sampling_frequency=signal.sampling_frequency,
    )


# def delay(signal: Signal, distance: float, sound_speed: float) -> Signal:
#     num_delay_points = signal.sampling_frequency * (distance / sound_speed)
#     return Signal(
#         values=np.pad(signal.values[num_delay_points:], ((0, num_delay_points))),
#         sampling_frequency=signal.sampling_frequency,
#     )
