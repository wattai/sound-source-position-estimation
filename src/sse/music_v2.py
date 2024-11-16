# -*- coding: utf-8 -*-

import numpy as np

from sse.base import SoundSourceLocatorBase


class MusicSoundSourceLocator(SoundSourceLocatorBase):
    """Music Sound Source Locator.

    References:
        https://www.fujipress.jp/main/wp-content/themes/Fujipress/pdf_subscribed.php

    """

    def __init__(
        self,
        fs: float,
        d: float,
        N_theta: int = 180,
    ):
        """Implementation of MUSIC method.

        Args:
            fs: Sampling frequency [Hz].
            d: distance between mics.
            N_theta (int, optional): _description_. Defaults to 180.
        """

        self.fs = fs  # sampling frequency [Hz].
        self.Ts = 1.0 / fs  # sampling Term [sec].
        self.d = d  # mic. distance [m].
        self.k = None  # data index series [point].
        self.L = None  # data length [point].
        self.c = 340

        self.N_theta = N_theta
        self.N_ch = None

        self.minvec = None
        self.thetas = None

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        num_sources: int = 1,
    ) -> "MusicSoundSourceLocator":
        # x: input sound signal
        # x.shape = (sample, N_ch)

        self.L = len(X)
        self.k = np.arange(self.L // 2)
        self.N_ch = X.shape[1]

        self.window_width = 512
        print("X.shape", X.shape)
        X_spectrogram = stft(X, window_width=self.window_width)  # [time, freq, ch]
        print("X_spectrogram.shape", X_spectrogram.shape)
        self.R = X_spectrogram.transpose(1, 2, 0) @ X_spectrogram.conj().transpose(
            1, 0, 2
        )
        self.eigvec, self.eigval, _ = np.linalg.svd(self.R)
        # self.eigval, self.eigvec = np.linalg.eig(self.R)
        print("self.eigvec.shape", self.eigvec.shape)
        self.minvec = self.eigvec[:, :, num_sources:].reshape(
            -1, self.N_ch, self.N_ch - num_sources
        )
        self.thetas = np.linspace(0, np.pi, self.N_theta)  # -90deg ~ +90deg
        # self.thetas = np.linspace(-np.pi / 2, np.pi / 2, self.N_theta)  # -90deg ~ +90deg
        self.S = self._calc_s_music(self.thetas)
        print("S.shape", self.S.shape)

        # import matplotlib.pyplot as plt
        # plt.pcolor(np.abs(self.S))
        # plt.show()
        # plt.plot(np.abs(self.S.mean(1)))
        # plt.show()

        return np.rad2deg(self.thetas[np.abs(self.S)[:, :].mean(1).argmax()])

    def _theta_hat(self):
        return np.rad2deg(self.thetas[(np.abs(self.S) ** 2).mean(0).argmax()])

    def _calc_s_music(self, thetas: list[float]):
        freqs = np.fft.fftfreq(n=self.window_width, d=self.Ts)
        print("freqs", freqs)
        print(
            "self._calc_alpha2(theta, freqs).shape",
            self._calc_alpha2(thetas[0], freqs).shape,
        )

        outs = []
        for theta in thetas:
            a = self._calc_alpha2(theta, freqs=freqs).reshape(-1, 1, self.N_ch)
            # print("a.shape", a.shape)
            # print("self.minvec.shape", self.minvec.shape)
            # print("np.linalg.vector_norm(a, axis=1, ord=2).shape", np.linalg.vector_norm(a, axis=1, ord=2).shape)
            upper = np.abs((a.conj() @ self.minvec))[:, 0, :] ** 2
            lower = np.linalg.vector_norm(a, axis=2, ord=2)
            # print("upper.shape", upper.shape)
            # print("lower.shape", lower.shape)
            outs.append(1 / np.sum(upper / lower, axis=1))
        return np.array(outs)

    def _calc_alpha2(self, theta, freqs: list[float]) -> np.ndarray:
        """Calculate array vectors that the shape is [num_freqs, num_mics]."""
        ma = LinearMicrophoneArray(
            d=self.d,
            c=self.c,
            num_mics=self.N_ch,
        )
        return ma.calc_array_vectors(
            theta=theta,
            freqs=freqs,
        )


class LinearMicrophoneArray:
    def __init__(
        self,
        d: float = 0.1,
        c: float = 340,
        num_mics: int = 2,
    ):
        self.d = d
        self.c = c
        self.num_mics = num_mics

    def calc_array_vectors(
        self,
        theta: float,
        freqs: list[float],
    ) -> np.ndarray:
        """Calculate array vectors that the shape is [num_freqs, num_mics]."""
        return array_vectors_of_liner_microphone_array(
            theta=theta,
            freqs=freqs,
            d=self.d,
            c=self.c,
            num_mics=self.num_mics,
        )


def array_vectors_of_liner_microphone_array(
    theta: float,
    freqs: list[float],
    d: float = 0.1,
    c: float = 340,
    num_mics: int = 2,
) -> np.ndarray:
    """Define array vectors for lined microphones.

    Args:
        theta: Angle which indicates where the signal comes from.
        freqs: Assumed signal frequencies.
        d: distance between microphones. Defaults to 0.1.
        c: Speed of sound. Defaults to 340.
        num_mics: The numbers of micorophones in the array. Defaults to 2.

    Returns:
        Array vectors that the shape is [num_freqs, num_mics].

    """
    tau = [(i * d * np.cos(theta)) / c for i in range(num_mics)]
    return np.array(
        [[np.exp(1j * 2 * np.pi * f * t) for t in tau] for f in freqs],
    )


def stft(x: np.ndarray, window_width: int = 512, shift_width: int = 256):
    X = []
    for idx in range(0, len(x) - window_width, shift_width):
        print(x[idx : idx + window_width].shape)
        X.append(np.fft.fft(x[idx : idx + window_width], axis=0))
    return np.array(X)
