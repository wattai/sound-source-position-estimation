# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 12:07:55 2016

@author: wattai
"""

import numpy as np
import scipy as sp

# CSP(Cross-power Spectrum Phase analysis:白色化相互相関法)を用いたDOA計算
def doa_csp(x, fs, v_wave, dist_mic):
    # x: 2ch signal data
    # fs: sampling rate [Hz]
    # v_wave: velocity of the wave [m/s]
    # dist_mic: distance between mics [m]
    
    nearest_dir = 0
    if x.shape[1] != 2:
        print("Input is NOT 2ch !!")
    else:
        # hamming窓付きFFT
        X = np.fft.fft((x.T * np.hamming(len(x))).T, axis=0)
        # 振幅で正規化
        X /= np.abs(X)
        # 相互相関関数を求める
        S = (X[:, 0].conj() * X[:, 1]) / (np.abs(X[:, 0]) * np.abs(X[:, 1]))
        # CSP係数を求める
        CSP = np.fft.ifft(S , axis=0)
        # CSP係数が最大となるサンプル差（位相差）を求める
        delta = np.argmax(CSP, axis=0)
        # 到達時間差(DOA)を求める
        tau = delta / fs
        # 音源が,どちらのマイクに近いか推定
        if delta < len(x)/2: nearest_dir = -1
        elif delta > len(x)/2: nearest_dir = 1
        # 到達時間差(DOA)修正
        if nearest_dir == False:
            tau = -1. * (len(x)/fs - tau)
        # 音源方向推定
        z = v_wave * tau / dist_mic
        if z > 1: z = 0 # clipping upto 1
        elif z < -1: z = -0  # clipping downto -1
        theta = np.rad2deg(np.arcsin(z))
            
    return tau, nearest_dir, theta, CSP

    
import matplotlib.pyplot as plt
if __name__ == "__main__":
    
    fs = 44100
    T = 2
    f1 = 500
    f2 = 300
    d1 = 1.00060
    d2 = 1.00055
    t = np.linspace(0, fs*T-1, fs*T)/fs
    x = np.zeros([fs*T, 2])
    x[:, 0] += np.random.randn(fs*T) * 0.005
    x[:, 1] += np.random.randn(fs*T) * 0.005
    t1 = np.linspace(0, fs*(T-d1)-1, fs*(T-d1)) / fs
    t2 = np.linspace(0, fs*(T-d2)-1, fs*(T-d2)) / fs
    x[int(np.ceil(fs*d1)):, 0] += np.exp(-t1*10) * np.sin(2*np.pi*f1*(t1)) 
    x[int(np.ceil(fs*d2)):, 1] += np.exp(-t2*10) * np.sin(2*np.pi*f1*(t2)) 
    
    # frame 毎に音源方向推定
    result = np.array([0, 0, 0])
    N_frame = 1024
    r = np.arange(0, len(x)-N_frame, int(N_frame/2))
    #for i in range(len(x)-N_frame):
    for i in r:
        x_frame = x[i:int(i+N_frame), :]
        result = np.c_[result, doa_csp(x_frame, fs, v_wave = 340, dist_mic = 8*10e-3)[:3]]
    
    tau, nearest_dir, theta, CSP = doa_csp(x, fs, v_wave = 340, dist_mic = 8*10e-3)
    print(tau, nearest_dir, theta)
    plt.plot(t, x[:, 0])
    plt.plot(t, x[:, 1])
    