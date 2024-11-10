# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(7)


class MUSIC:
    def __init__(self, fs, d, c=340.0, N_theta=180):
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

    def alpha(self, theta):
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

    def S_MN(self, thetas):
        return np.array(
            [
                1.0
                / np.abs(
                    self.minvec.conj().transpose(0, 2, 1)
                    @ self.alpha(theta).transpose(1, 0).reshape(-1, self.N_ch, 1)
                ).squeeze()
                ** 2
                for theta in thetas
            ]
        ).T * np.sqrt(self.eigval[:, 0][:, None] / (self.eigval[:, -1][:, None] + 1e-8))

    def S_MUSIC(self, thetas):
        return np.array(
            [
                np.sum(self.alpha(theta).conj() * self.alpha(theta), axis=0)
                / np.abs(
                    self.minvec.conj().transpose(0, 2, 1)
                    @ self.alpha(theta).transpose(1, 0).reshape(-1, self.N_ch, 1)
                ).squeeze()
                ** 2
                for theta in thetas
            ]
        ).T * np.sqrt(
            self.eigval[:, 0][:, None]
        )  # / (self.eigval[:, -1][:, None]+1e-8) )

    def fit_transform(self, x):
        # x: input sound signal
        # x.shape = (sample, N_ch)

        self.L = len(x) // 1
        self.k = np.arange(self.L // 2)
        self.N_ch = x.shape[1]

        winfunc = np.hanning(len(x)).reshape(-1, 1)
        X = np.fft.fft(winfunc * x, axis=0)[: self.L // 2, :]
        self.R = (X.conj().reshape(-1, self.N_ch, 1) @ X.reshape(-1, 1, self.N_ch)) / (
            X.shape[0] / self.fs
        )

        self.eigval, self.eigvec = np.linalg.eig(self.R)
        self.minvec = self.eigvec[:, :, -1].reshape(-1, self.N_ch, 1)

        self.thetas = np.linspace(-np.pi / 2, np.pi / 2, self.N_theta)
        # self.S = self.S_MN(self.thetas)
        self.S = self.S_MUSIC(self.thetas)
        return self.S

    def theta_hat(self):
        return np.rad2deg(self.thetas[(np.abs(self.S) ** 2).mean(0).argmax()])


def CSP(x):
    phi = np.correlate(x[:, 0], x[:, 1], mode="full")
    PHI = np.fft.fft(phi)
    csp = np.fft.fft(PHI / np.abs(PHI)).real
    return csp


def TDOA(x):
    estimated_delay = CSP(x).argmax() - (len(x[:, 0]))
    return estimated_delay


def tdoa2deg(tdoa, c=340, d=0.1):
    return np.rad2deg(np.arccos(np.clip(tdoa * c / d, -1, 1)))


def deg2tdoa(deg, c=340, d=0.1):
    return d * np.cos(np.deg2rad(deg)) / c


def simu_csp(theta=120, fs=16000, N_fft=128, c=340, d=1.0):
    tdoa = deg2tdoa(theta, c=c, d=d)
    width = N_fft // 2
    t = np.linspace(0, N_fft + 2 * width - 1, N_fft + 2 * width) / fs  # base time.
    s = np.random.randn(N_fft + 2 * width)  # signal source.
    ptdoa = np.round(tdoa * fs).astype("i")  # point of TDOA.
    t1 = t[width + ptdoa : width + ptdoa + N_fft]
    # t2 = t[width : width+N]
    x1 = s[width + ptdoa : width + ptdoa + N_fft] + 0.5 * np.random.randn(len(t1))
    x2 = s[width : width + N_fft] + 0.5 * np.random.randn(len(t1))
    x = np.c_[x1, x2]

    theta_hat = tdoa2deg(TDOA(x) / fs, c=c, d=d)
    csp = CSP(x)
    N_csp = len(csp)
    csp /= N_csp
    t_delay = np.linspace(-N_csp // 2, N_csp // 2, N_csp) / fs
    theta_delay = tdoa2deg(t_delay, c=c, d=d)

    plt.figure()
    plt.subplot(211)
    plt.title("Based on CSP.")
    plt.plot(t1, x1, linestyle="-", label="1ch")
    plt.plot(t1, x2, linestyle="-.", label="2ch")
    plt.legend(loc="upper right")
    plt.xlabel("time [sec]")
    plt.ylabel("amp. [a.u.]")
    plt.xlim(t1.min(), t1.max())
    plt.grid(linestyle="--")

    plt.subplot(212)
    plt.plot(
        (90 - theta_delay),
        10 * np.log10((csp**2) / (csp**2).max()),
        marker="D",
        markersize=5,
    )
    plt.xlabel("theta delay [deg]")
    plt.ylabel("CSP log power [dB]")
    plt.xlim(90 - theta_delay.max(), 90 - theta_delay.min())
    plt.grid(linestyle="--")
    plt.tight_layout()
    plt.show()
    print("theta: %.3f, theta_hat: %.3f" % (90 - theta, 90 - theta_hat))


def simu_music(theta=120, fs=16000, N_fft=128, c=340, d=1.0, N_theta=90 + 1):
    tdoa = d * np.cos(np.deg2rad(theta)) / c
    width = N_fft // 2
    # s = np.random.randn(N_fft + 2*width) # signal source.
    ptdoa = np.round(tdoa * fs).astype("i")  # point of TDOA.
    t = np.linspace(0, N_fft + 2 * width - 1, N_fft + 2 * width) / fs  # base time.
    t1 = t[width + ptdoa : width + N_fft + ptdoa]
    t2 = t[width : width + N_fft]
    x1 = (
        1 * np.sin(2 * np.pi * 500 * t1)[:, None]
        + 0.01 * np.random.randn(len(t1))[:, None]
    )
    x2 = (
        1 * np.sin(2 * np.pi * 500 * t2)[:, None]
        + 0.01 * np.random.randn(len(t2))[:, None]
    )
    # x1 = s[width+ptdoa : width+ptdoa+N_fft][:, None] + 0.01*np.random.randn(len(t1))[:, None]
    # x2 = s[width : width+N_fft][:, None] + 0.01*np.random.randn(len(t2))[:, None]
    x = np.c_[x1, x2]

    est = MUSIC(fs=fs, d=d, c=c, N_theta=N_theta)
    S = est.fit_transform(x)

    # eps = 1e-8
    plt.figure()
    plt.subplot(211)
    plt.title("Based on MUSIC.")
    X, Y = np.meshgrid(np.rad2deg(est.thetas), np.linspace(0, fs // 2 - 1, len(x) // 2))
    # X, Y = np.meshgrid(np.rad2deg(est.thetas), np.linspace(0, fs // 2 - 1, len(x) // 2))
    plt.pcolor(X, Y, (np.abs(S)), cmap="jet")
    plt.colorbar()
    plt.xlim(-90, 90)
    plt.grid(linestyle="--")
    plt.ylabel("frequency [Hz]")
    plt.xlabel("angle [deg]")

    plt.subplot(212)
    P = 10 * np.log10((np.abs(S) ** 2).mean(0) / (np.abs(S) ** 2).mean(0).max())
    plt.plot(np.rad2deg(est.thetas), P, marker="D", markersize=5)
    plt.xlim(-90, 90)
    plt.grid(linestyle="--")
    plt.ylabel("log-power spec. [dB]")
    plt.xlabel("angle [deg]")
    plt.tight_layout()

    plt.show()

    print("thata: %.3f, theta_hat: %.3f" % (90 - theta, est.theta_hat()))


if __name__ == "__main__":
    fs = 48000  # sampling frequency. [Hz]
    c = 340  # wave speed. [m/sec]
    d = 0.1  # width for mic. array. [m]

    theta = 75  # 0~180 [deg]
    N_fft = 2048
    simu_csp(theta=90 - theta, fs=fs, N_fft=N_fft, c=c, d=d)
    simu_music(theta=90 - theta, fs=fs, N_fft=N_fft, c=c, d=d, N_theta=180 + 1)
