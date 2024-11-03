# -*- coding: utf-8 -*-

import numpy as np

from sse.base import SoundSourceLocatorBase


class MusicSoundSourceLocator(SoundSourceLocatorBase):
    """Music Sound Source Locator."""

    def __init__(
        self,
        fs: float,
        d: float,
        c: float = 340.0,
        N_theta: int = 180,
    ):
        """Implementation of MUSIC method.

        Args:
            fs: Sampling frequency [Hz].
            d: distance between mics.
            c: sound speed. Defaults to 340.0.
            N_theta (int, optional): _description_. Defaults to 180.
        """

        self.fs = fs  # sampling frequency [Hz].
        self.T = 1.0 / fs  # sampling Term [sec].
        self.d = d  # mic. distance [m].
        self.k = None  # data index series [point].
        self.L = None  # data length [point].
        self.c = c  # sound speed [m/sec].
        self.N_theta = N_theta
        self.N_ch = None

        self.minvec = None
        self.thetas = None

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> "MusicSoundSourceLocator":
        # x: input sound signal
        # x.shape = (sample, N_ch)

        self.L = len(X) // 1
        self.k = np.arange(self.L // 2)
        self.N_ch = X.shape[1]

        winfunc = np.hanning(len(X)).reshape(-1, 1)

        X_freq = np.fft.fft(winfunc * X, axis=0)[: self.L // 2, :]
        self.R = (
            X_freq.conj().reshape(-1, self.N_ch, 1) @ X_freq.reshape(-1, 1, self.N_ch)
        ) / (X_freq.shape[0] / self.fs)

        self.eigval, self.eigvec = np.linalg.eig(self.R)
        self.minvec = self.eigvec[:, :, -1].reshape(-1, self.N_ch, 1)

        self.thetas = np.linspace(-np.pi / 2, np.pi / 2, self.N_theta)
        self.S = self._calc_s_music(self.thetas)
        return self._theta_hat()

    def _theta_hat(self):
        return np.rad2deg(self.thetas[(np.abs(self.S) ** 2).mean(0).argmax()])

    def _calc_s_music(self, thetas):
        return np.array(
            [
                np.sum(self._calc_alpha(theta).conj() * self._calc_alpha(theta), axis=0)
                / np.abs(
                    self.minvec.conj().transpose(0, 2, 1)
                    @ self._calc_alpha(theta).transpose(1, 0).reshape(-1, self.N_ch, 1)
                ).squeeze()
                ** 2
                for theta in thetas
            ]
        ).T * np.sqrt(
            self.eigval[:, 0][:, None]
        )  # / (self.eigval[:, -1][:, None]+1e-8) )

    def _calc_alpha(self, theta):
        return np.array(
            [
                np.ones(len(self.k)),
                np.exp(
                    2j
                    * np.pi
                    * self.d
                    * np.sin(theta)
                    / (self.c * self.T)
                    * ((self.k) / (len(self.k) * 2))
                ),
            ]
        )
