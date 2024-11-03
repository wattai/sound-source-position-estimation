# -*- coding: utf-8 -*-

import numpy as np

from sse.music_v2 import MusicSoundSourceLocator


def make_dummy_signals(
    theta=15.0,
    fs=16000,
    N_fft=128,
    c=340,
    d=1.0,
) -> np.ndarray:
    tdoa = d * np.sin(np.deg2rad(theta)) / c
    # tdoa = d * np.cos(np.deg2rad(theta)) / c

    width = N_fft // 2
    # s = np.random.randn(N_fft + 2*width) # signal source.
    ptdoa = np.round(tdoa * fs).astype("i")  # point of TDOA.
    t = np.linspace(0, N_fft + 2 * width - 1, N_fft + 2 * width) / fs  # base time.
    t1 = t[width + ptdoa : width + N_fft + ptdoa]
    t2 = t[width : width + N_fft]
    x1 = np.sin(2 * np.pi * 500 * t1)[:, None]
    x2 = np.sin(2 * np.pi * 500 * t2)[:, None]
    x = np.c_[x1, x2]
    return x


class TestMusicSoundSourceLocator:
    def test_fit_transform(self):
        self.locator = MusicSoundSourceLocator(
            fs=48000,
            d=0.1,
            N_theta=180,
        )
        X = make_dummy_signals(
            theta=40.0,
            fs=48000,
            N_fft=2048,
            d=0.1,
        )
        predicted_theta = self.locator.fit_transform(X=X)
        np.testing.assert_allclose(predicted_theta, 42.737430)
