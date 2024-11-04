# -*- coding: utf-8 -*-

import numpy as np

from sse.music_v2 import MusicSoundSourceLocator


def make_dummy_signals(
    theta=15.0,
    fs=16000,
    c=340,
    d=1.0,
) -> np.ndarray:
    """_summary_

    Args:
        theta (float, optional): _description_. Defaults to 15.0.
        fs (int, optional): _description_. Defaults to 16000.
        N_fft (int, optional): _description_. Defaults to 128.
        c: Sound speed [m/sec]. Defaults to 340.
        d: Distance between mics [m]. Defaults to 1.0.

    Returns:
        np.ndarray: _description_
    """
    tdoa = d * np.sin(np.deg2rad(theta)) / c
    # tdoa = d * np.cos(np.deg2rad(theta)) / c
    print("tdoa", tdoa)

    T = 0.2  # [sec]
    # width = N_fft // 2
    num_points_of_tdoa_width = int(tdoa * fs)  # point of TDOA.
    # t = np.linspace(0, N_fft + 2 * width - 1, N_fft + 2 * width) / fs
    t = np.linspace(0, int(fs * T - 1), int(fs * T)) / fs  # base time.
    # t1 = t[width + num_points_of_tdoa_width : width + N_fft + num_points_of_tdoa_width]
    # t2 = t[width : width + N_fft]
    t1 = t[num_points_of_tdoa_width:]
    t2 = t[:-num_points_of_tdoa_width]
    print("t1.shape", t1.shape)
    print("t2.shape", t2.shape)
    x1 = np.sin(2 * np.pi * 5000 * t1)[:, None]
    x2 = np.sin(2 * np.pi * 5000 * t2)[:, None]
    x = np.c_[x1, x2]
    # x = np.c_[x1 + np.random.randn(*x1.shape) * 0.05, x2 + np.random.randn(*x2.shape) * 0.05]

    # xs = np.random.randn(len(t))[:, None]
    # x1 = xs[num_points_of_tdoa_width:]
    # t2 = xs[:-num_points_of_tdoa_width]
    # x = np.c_[x1, x2]
    return x


class TestMusicSoundSourceLocator:
    def test_fit_transform(self):
        self.locator = MusicSoundSourceLocator(
            fs=16000,
            d=0.1,
            N_theta=180 + 1,
        )
        X = make_dummy_signals(
            theta=40.0,
            fs=16000,
            d=0.1,
        )
        predicted_theta = self.locator.fit_transform(X=X)
        print("predicted_theta", predicted_theta)
        # np.testing.assert_allclose(predicted_theta, 40.726257)
