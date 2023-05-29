# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 15:35:24 2020

@author: raulm
"""

# Imports
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import librosa
from tqdm import tqdm
# Import helper functions
from helper_functions import awgn, sliding_window, sliding_window_rec, Yule_Walker, kalman_ite

# KALMAN FILTER TO DENOISE VOICE SIGNAL


# file_path = './input.wav'
# signal, Fs = librosa.load(file_path)
# signal = signal.reshape((-1, 1))
# Signal length

signal_number = 1

# Load signals
if (signal_number == 1):
    signal = scipy.io.loadmat('fcno01fz.mat')
    signal = np.array(signal["fcno01fz"])
else:
    signal = scipy.io.loadmat('fcno02fz.mat')
    signal = np._ArrayComplex_co(signal["fcno02fz"])


Fs = 8000  # 8 kHz sample frequency

N = signal.shape[0]

# time vector
t = np.arange(0, N/Fs, 1/Fs)
t = np.reshape(t, (len(t), 1))

# frequency vector
freq = np.linspace(-Fs/2, Fs/2, N)
freq = np.reshape(freq, (len(freq), 1))

# Yules-Walker and AR parameters
p = 16  # AR order
ite_kalman = 10  # We apply YW-KALMAN a few times to each slice

# Noise addition
SNR = 10  # dB
noisy_signal, wg_noise = awgn(signal, SNR)

# Plot noisy signal and original signal
plt.figure()
plt.grid()
plt.title('Original signal vs Noisy signal')
plt.plot(t, signal)
plt.plot(t, noisy_signal, 'r--', linewidth=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(['Original signal', 'Noisy signal'])

# Noise variance estimation (using silence )
varNoise = np.var(noisy_signal[1:4500])

# Sliced signal
signal_sliced_windowed, padding = sliding_window(signal, Fs)

# Save after filtering
signal_sliced_windowed_filtered = np.zeros((signal_sliced_windowed.shape))


for ite_slice in tqdm(range(signal_sliced_windowed.shape[1])):

    # Slice n
    slice_signal = signal_sliced_windowed[:, ite_slice:ite_slice+1].T

    # On fait YW-KALMAN plusieurs tours pour chaque morceau
    for ite in range(ite_kalman):
        # YW
        a, var_bruit = Yule_Walker(slice_signal, p)
#        ar, variance, coeff_reflection = aryule(slice_signal, p)

        # Save
        signal_filtered = np.zeros((1, signal_sliced_windowed.shape[0]))

        # Phi et H
        Phi = np.concatenate((np.zeros((p-1, 1)), np.eye(p-1)), axis=1)
        Phi = np.concatenate((Phi, -np.fliplr(a[1:].T)), axis=0)

        H = np.concatenate((np.zeros((p-1, 1)), np.ones((1, 1))), axis=0).T

        # Q, R and Po
        Q = var_bruit*np.eye(p)
        R = varNoise
        P = 10000*np.eye(p)

        # Initialisation vecteur d'etat
        x = np.zeros((p, 1))

        for jj in range(signal_sliced_windowed.shape[0]):
            y = slice_signal[0][jj]  # Observation
            [x, P] = kalman_ite(x, P, y, Q, R, Phi, H)
            signal_filtered[0][jj] = x[-1]

        slice_signal = signal_filtered
    signal_sliced_windowed_filtered[:,
                                    ite_slice:ite_slice+1] = signal_filtered.T


# Reconstruct signal
signal_reconstructed = sliding_window_rec(
    signal_sliced_windowed_filtered, Fs, padding)

# Plot reconstructed signal and original signal
plt.figure()
plt.grid()
plt.title('Noisy signal vs Reconstructed signal')
plt.plot(t, noisy_signal)
plt.plot(t, signal_reconstructed, 'r--', linewidth=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(['Noisy signal', 'Reconstructed signal'])
plt.show()

# Play Sound
sd.play(signal, Fs, blocking=True)
sd.play(wg_noise, Fs, blocking=True)
sd.play(noisy_signal, Fs, blocking=True)
sd.play(signal_reconstructed, Fs, blocking=True)
