# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:57:57 2020

@author: raulm
"""
import numpy as np
from scipy.linalg import toeplitz


def awgn(signal, SNR_db):
    """
    This function adds white gaussian noise (WGN)
    to the input signal at the desired SNR(db)
    """
    # Signal power
    S = np.sum(signal**2)

    # 1 random draw
    wg_noise = np.random.randn(len(signal), 1)

    # Noise power
    N = np.sum(wg_noise**2)

    # WGN
    wg_noise = np.sqrt(10**(-SNR_db/10)*(S/N))*wg_noise

    # Debug
#    snr_db_computed = 10*np.log10(np.sum(signal**2)/np.sum(wg_noise**2))
#    print(snr_db_computed)
    noisy_signal = signal.reshape(wg_noise.shape) + wg_noise

    return noisy_signal, wg_noise


def sliding_window(signal, Fs):
    """
    This function splits the signal into pseudo-stationary slices 
    with a 50 % overlapping
    """

    # 50% overlapping
    stationary_samples = int(30e-3*Fs)  # parole stationnaire ~= 30 ms
    half_stationary = int(stationary_samples/2)

    # Addition de padding avant reshape
    padding = half_stationary - np.remainder(len(signal), half_stationary)
    signal_padding = np.concatenate((signal, np.zeros((padding, 1))))

    # Reshape
    signal_sliced_half_stationary = np.reshape(np.concatenate((signal_padding, np.zeros(
        (half_stationary, 1)))), (half_stationary, -1), order='F')  # recouvrement 50%

    signal_sliced_half_stationary_delayed = np.reshape(np.concatenate((np.zeros(
        (half_stationary, 1)), signal_padding)), (half_stationary, -1), order='F')  # recouvrement 50%

    signal_sliced = np.concatenate(
        (signal_sliced_half_stationary_delayed, signal_sliced_half_stationary), axis=0)

    # Enleve ce qu'on a ajoute artifitiallement
    signal_sliced = signal_sliced[:, 1:-1]

    # Windowing
    ham_window = np.hamming(stationary_samples)
    ham_window = ham_window.reshape((len(ham_window), 1))
    signal_sliced_windowed = ham_window @ np.ones(
        (1, signal_sliced.shape[1]))*signal_sliced

    return signal_sliced_windowed, padding


def sliding_window_rec(signal_sliced_windowed, Fs, padding):
    """
    This function reconstructs the signal slices 
    to form the whole signal
    """

    # 50% overlapping
    stationary_samples = int(30e-3*Fs)  # parole stationnaire ~= 30 ms
    half_stationary = int(stationary_samples/2)

    x = signal_sliced_windowed.shape[1]
    signal_reconstructed = np.zeros((half_stationary*(x+1), x))

    # Unwindowing
    ham_window = np.hamming(stationary_samples)
    ham_window = ham_window.reshape((len(ham_window), 1))
    unwindow = 1/(ham_window[0:half_stationary]+ham_window[half_stationary:])
    unwindow = unwindow.reshape((len(unwindow), 1))
    unwindow_2 = np.repeat(unwindow, 2)
    unwindow_2 = unwindow_2.reshape((len(unwindow_2), 1))

    for ind in range(x):
        a = np.zeros((half_stationary*ind, 1))
        b = signal_sliced_windowed[:, ind:ind+1]*unwindow_2
        c = np.zeros((half_stationary*(x-ind-1), 1))
        signal_reconstructed[:, ind:ind+1] = np.concatenate((a, b, c), axis=0)

    signal_reconstructed2 = np.sum(signal_reconstructed, axis=1)
    signal_reconstructed2 = signal_reconstructed2.reshape(
        (len(signal_reconstructed2), 1))
    signal_reconstructed = signal_reconstructed2[0:-padding]

    return signal_reconstructed


def Yule_Walker(x, order):
    """
    This function computer Yule-Walker equations
    """
    rxx = np.correlate(x[0, :], x[0, :], "full")
    rxx = rxx.reshape((len(rxx), 1))
    zero = x.shape[1]

    rxx_vector = rxx[zero+1:zero+order+1]
    # rxx_vector = rxx_vector.reshape((len(rxx_vector),1))
    Rxx = toeplitz(rxx[zero:zero+order])

    a = np.concatenate(
        (np.ones(((1, 1))), np.linalg.inv(Rxx)@rxx_vector), axis=0)

    row = np.arange(0, order+1) + zero
    Rxx_row = rxx[row]

    # Estimation var bruit
    var_bruit = 0

    for pp in range(len(a)):
        var_bruit += a[pp]*Rxx_row[pp]

    return a, int(var_bruit)


def kalman_ite(x, P, y, Q, R, Phi, H):

    # Prediction for state vector and error covariance:
    x = Phi@x  # State vector estimation
    P = Phi@P@Phi.T + Q  # Error covariance matrix a priori

    # Compute Kalman gain factor:
    K = (P@H.T)@np.linalg.inv(H@P@H.T + R)  # Kalman gain

    # Correction based on observation:
    x = x + K@(y-H@x)  # State vector corrected estimation
    P = P - K@H@P  # Error covariance matrix a posteriori
    return x, P
