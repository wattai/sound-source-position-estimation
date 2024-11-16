# -*- coding: utf-8 -*-

import pytest

import numpy as np

from sse.music_v2 import MusicSoundSourceLocator
from sse.simulators.environments import (
    Observer,
    Air,
    Microphone,
    Source,
    Position3D,
    SineSignalGenerator,
)

SAMPLING_FREQUENCY = 16000  # [Hz]
SOUND_SPEED = Air().sound_speed  # [m/s]
GAP_WIDTH_BETWEEN_MICS = 5.0  # [m]
SIGNAL_TIME_LENGTH = 5.0  # [sec.]


def make_dummy_signals(
    theta: float,
    fs: float,
    d: float,
    time_length: float,
    medium=Air(),
) -> np.ndarray:
    """Return 2ch dummy signals.

    Args:
        theta: Which direction the signal comes from [rad].
        fs: Sampling frequency [Hz].
        d: Distance between mics [m].
        medium: Medium which sounds pass through.

    Returns:
        Sound waves shaped as: [num_samples, num_channels].
    """

    obs = Observer(
        sources=[
            Source(
                position=Position3D(r=100, theta=theta, phi=0),
                signal=SineSignalGenerator(frequency=3000.2).generate(
                    sampling_frequency=fs,
                    time_length=time_length,
                ),
            )
        ],
        microphones=[
            Microphone(
                position=Position3D(r=d / 2, theta=0, phi=0),
                sampling_frequency=fs,
            ),
            Microphone(
                position=Position3D(r=d / 2, theta=np.pi, phi=0),
                sampling_frequency=fs,
            ),
        ],
        medium=medium,
    )
    outs = obs.ring_sources()
    return np.c_[outs[0].values, outs[1].values]


class TestMusicSoundSourceLocator:
    @pytest.mark.parametrize("theta", [60])
    def test_fit_transform(self, theta: float):
        x = make_dummy_signals(
            theta=theta / 180 * np.pi,
            fs=SAMPLING_FREQUENCY,
            d=GAP_WIDTH_BETWEEN_MICS,
            time_length=10.0,
        )
        self.locator = MusicSoundSourceLocator(
            fs=SAMPLING_FREQUENCY,
            d=GAP_WIDTH_BETWEEN_MICS,
            N_theta=180 + 1,
        )
        predicted_theta = self.locator.fit_transform(X=x)
        print("predicted_theta (MUSIC): ", predicted_theta)


class TestCSPSoundSourceLocator:
    @pytest.mark.parametrize("theta", [30, 60, 90, 120, 150])
    def test_accuracy(
        self,
        theta: float,
        acceptable_error_in_deg: float = 5.0,
    ):
        x = make_dummy_signals(
            theta=theta / 180 * np.pi,
            fs=SAMPLING_FREQUENCY,
            d=GAP_WIDTH_BETWEEN_MICS,
            time_length=SIGNAL_TIME_LENGTH,
        )
        predicted_theta = estimate_theta_by_csp(
            x1=x[:, 0],
            x2=x[:, 1],
            fs=SAMPLING_FREQUENCY,
            c=SOUND_SPEED,
            d=GAP_WIDTH_BETWEEN_MICS,
        )
        print("predicted_theta (CSP): ", predicted_theta)
        assert (predicted_theta - theta) ** 2 < acceptable_error_in_deg


def estimate_theta_by_csp(
    x1: np.ndarray,
    x2: np.ndarray,
    fs: float = 16000,
    c: float = 343.3,
    d: float = 0.1,
) -> float:
    return tdoa2deg(calc_tdoa(x1, x2) / fs, c=c, d=d)


def calc_tdoa(x1: np.ndarray, x2: np.ndarray) -> float:
    assert len(x1) == len(x2)
    estimated_delay = calc_csp_coefs(x1=x1, x2=x2).argmax() - len(x1)
    return estimated_delay


def calc_csp_coefs(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    phi = np.correlate(x2, x1, mode="full")
    PHI = np.fft.fft(phi)
    csp = np.fft.fft(PHI / np.abs(PHI)).real
    return csp


def tdoa2deg(
    tdoa: float,
    c: float = 343.3,
    d: float = 0.1,
) -> float:
    return np.rad2deg(np.arccos(np.clip(tdoa * c / d, -1, 1)))


def deg2tdoa(
    deg: float,
    c: float = 343.3,
    d: float = 0.1,
) -> float:
    return d * np.cos(np.deg2rad(deg)) / c
