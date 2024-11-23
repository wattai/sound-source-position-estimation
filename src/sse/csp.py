# -*- coding: utf-8 -*-

import numpy as np

from sse.base import SoundSourceLocatorBase


def calc_csp_coeffs(x):
    phi = np.correlate(x[:, 0], x[:, 1], mode="full")
    PHI = np.fft.fft(phi)
    csp = np.fft.fft(PHI / np.abs(PHI)).real
    return csp


def calc_tdoa(x):
    estimated_delay = calc_csp_coeffs(x).argmax() - (len(x[:, 0]))
    return estimated_delay


def tdoa2deg(tdoa, c=340, d=0.1):
    return np.rad2deg(np.arccos(np.clip(tdoa * c / d, -1, 1)))


def deg2tdoa(deg, c=340, d=0.1):
    return d * np.cos(np.deg2rad(deg)) / c


class CSPSoundSourceLocator(SoundSourceLocatorBase):
    def __init__(self):
        pass

    def fit_transform(self, X: np.ndarray, fs, c, d) -> float:
        # X: input sound signal
        # X.shape = (sample, N_ch)
        theta_hat = tdoa2deg(calc_tdoa(X) / fs, c=c, d=d)
        return theta_hat

    def calc_likehood_map(self):
        pass
        # csp = CSP(x)
        # N_csp = len(csp)
        # csp /= N_csp
        # t_delay = np.linspace(-N_csp // 2, N_csp // 2, N_csp) / fs
        # theta_delay = tdoa2deg(t_delay, c=c, d=d)

        # plt.figure()
        # plt.subplot(211)
        # plt.title("Based on CSP.")
        # plt.plot(t1, x1, linestyle="-", label="1ch")
        # plt.plot(t1, x2, linestyle="-.", label="2ch")
        # plt.legend(loc="upper right")
        # plt.xlabel("time [sec]")
        # plt.ylabel("amp. [a.u.]")
        # plt.xlim(t1.min(), t1.max())
        # plt.grid(linestyle="--")

        # plt.subplot(212)
        # plt.plot(
        #     (90 - theta_delay),
        #     10 * np.log10((csp**2) / (csp**2).max()),
        #     marker="D",
        #     markersize=5,
        # )
        # plt.xlabel("theta delay [deg]")
        # plt.ylabel("CSP log power [dB]")
        # plt.xlim(90 - theta_delay.max(), 90 - theta_delay.min())
        # plt.grid(linestyle="--")
        # plt.tight_layout()
        # plt.show()
        # print("theta: %.3f, theta_hat: %.3f" % (90 - theta, 90 - theta_hat))
