import os
import sys
import time
import pandas as pd
import numpy as np
from numba import jit


import math
from array import *

import numpy as np
from numba import jit
from math import factorial, log, floor
from sklearn.neighbors import KDTree
from scipy.signal import periodogram, welch
from scipy import stats
from statsmodels.tsa import stattools

from numpy.fft import fft, ifft, fft2, ifft2, fftshift



# Forked from FATS

def Meanvariance(data):
    magnitude = data[0]
    std = np.std(magnitude)
    try:
        mean = np.mean(magnitude)
    
        return std / mean
    except: return 0


def Amplitude(data):
    magnitude = data[0]
    N = len(magnitude)
    try:
        sorted_mag = np.sort(magnitude)

        return (np.median(sorted_mag[-math.ceil(0.05 * N):]) -
                np.median(sorted_mag[0:math.ceil(0.05 * N)])) / 2.0
    except: return 0
    # return sorted_mag[10]



def Rcs( data):
    """Range of cumulative sum"""
    magnitude = data[0]
    try:
        sigma = np.std(magnitude)
        if sigma==0:
            return 0
    
        N = len(magnitude)
        m = np.mean(magnitude)
        s = np.cumsum(magnitude - m) * 1.0 / (N * sigma)
        R = np.max(s) - np.min(s)
        return R
    except: return 0


def Autocor_length( data, lags=1):
    #from statsmodels.tsa import stattools
    nlags = lags
    magnitude = data[0]
    
    try:
        if stattools.acf(magnitude, nlags=nlags,fft=True,missing='none')!=0:
            AC = stattools.acf(magnitude, nlags=nlags,fft=True,missing='none')
            k = next((index for index, value in
                        enumerate(AC) if value < np.exp(-1)), None)

            while k is None:
                nlags = nlags + 1 #100
                AC = stattools.acf(magnitude, nlags=nlags,fft=True,missing='none')
                k = next((index for index, value in
                            enumerate(AC) if value < np.exp(-1)), None)

            return k
        else: return 0
    except: return 0
    



def Con(data, consecutiveStar=3):
    #    Index introduced for selection of variable starts from OGLE database.


    #To calculate Con, we counted the number of three consecutive measurements
    #that are out of 2sigma range, and normalized by N-2
    # https://arxiv.org/pdf/1506.00010.pdf


    magnitude = data[0]
    N = len(magnitude)
    try:
        if N < consecutiveStar:
            return 0
        sigma = np.std(magnitude)
        m = np.mean(magnitude)
        count = 0

        for i in range(N - consecutiveStar + 1):
            flag = 0
            for j in range(consecutiveStar):
                if(magnitude[i + j] > m + 2 * sigma or magnitude[i + j] < m - 2 * sigma):
                    flag = 1
                else:
                    flag = 0
                    break
            if flag:
                count = count + 1
        return count * 1.0 / (N - consecutiveStar + 1)
    except: return 0


def Color(data):

#"""Average color for each MACHO lightcurve
#mean(B1) - mean(B2)
#"""

    magnitude = data[0]
    magnitude2 = data[2]
    try: return np.mean(magnitude) - np.mean(magnitude2)
    except: return 0





def Beyond1Std(data):
#    ['magnitude', 'error']
#"""Percentage of points beyond one st. dev. from the weighted
#(by photometric errors) mean
#"""
    magnitude = data[0]        
    n = len(magnitude)
    error = np.full((n),0.01)
    try:
        weighted_mean = np.average(magnitude, weights=1 / error ** 2)

        # Standard deviation with respect to the weighted mean

        var = sum((magnitude - weighted_mean) ** 2)
        std = np.sqrt((1.0 / (n - 1)) * var)

        count = np.sum(np.logical_or(magnitude > weighted_mean + std,
                                        magnitude < weighted_mean - std))

        return float(count) / n
    except: return 0



def SmallKurtosis(data):
#"""Small sample kurtosis of the magnitudes.

#See http://www.xycoon.com/peakedness_small_sample_test_1.htm
#"""
    magnitude = data[0]
    n = len(magnitude)
    mean = np.mean(magnitude)
    std = np.std(magnitude)
    

    try:
        S = sum(((magnitude - mean) / std) ** 4)
        c1 = float(n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
        c2 = float(3 * (n - 1) ** 2) / ((n - 2) * (n - 3)) 
        return c1 * S - c2
    except: return 0




def Std(data):
    #   """Standard deviation of the magnitudes"""
    magnitude = data[0]
    try:return np.std(magnitude)
    except: return 0


def Skew(data):
    #"""Skewness of the magnitudes"""
    from scipy import stats
    magnitude = data[0]
    try:return stats.skew(magnitude)
    except: return 0



def StetsonJ(data):
    #"""Stetson (1996) variability index, a robust standard deviation"""
    #Data = ['magnitude', 'time', 'error', 'magnitude2', 'error2']
    aligned_magnitude = data[1]
    aligned_magnitude2 = data[2]
    aligned_error = np.full((len(aligned_magnitude)),0.01)  
    #aligned_error = np.random.normal(loc=1, scale =0.0001, size=len(aligned_magnitude))
    aligned_error2 = aligned_error #np.full((len(aligned_magnitude2)),0.001)
    #aligned_error2 = np.random.normal(loc=1, scale =0.0001, size=len(aligned_magnitude2))
    N = len(aligned_magnitude)

    try:
        mean_mag = (np.sum(aligned_magnitude/(aligned_error*aligned_error)) /
                    np.sum(1.0 / (aligned_error * aligned_error)))

        mean_mag2 = (np.sum(aligned_magnitude2 / (aligned_error2*aligned_error2)) /
                        np.sum(1.0 / (aligned_error2 * aligned_error2)))

        sigmap = (np.sqrt(N * 1.0 / (N - 1)) *
                    (aligned_magnitude[:N] - mean_mag) /
                    aligned_error)
        sigmaq2 = (np.sqrt(N * 1.0 / (N - 1)) *
                    (aligned_magnitude2[:N] - mean_mag2) /
                    aligned_error2)
        sigma_i = sigmap * sigmaq2

        J = (1.0 / len(sigma_i) * np.sum(np.sign(sigma_i) *
                np.sqrt(np.abs(sigma_i))))
        return J
    except: return 0





#############################
#import numpy as np
#from numpy.fft import fft, ifft, fft2, ifft2, fftshift
 
def cross_correlation_using_fft(x, y):
    try:
        f1 = fft(x)
        f2 = fft(np.flipud(y))
        cc = np.real(ifft(f1 * f2))
        return fftshift(cc)
    except: return 0
 
# shift &lt; 0 means that y starts 'shift' time steps before x # shift &gt; 0 means that y starts 'shift' time steps after x
def compute_shift(x, y):
    try:
        assert len(x) == len(y)
        c = cross_correlation_using_fft(x, y)
        assert len(c) == len(x)
        zero_index = int(len(x) / 2) - 1
        shift = zero_index - np.argmax(c)
        return shift
    except: return 0





def MaxSlope( data):
    #Examining successive (time-sorted) magnitudes, the maximal first difference
    #(value of delta magnitude over delta time)
    #Data = ['magnitude', 'time']
    magnitude = data[0]
    time = data[1]
    try:
        slope = np.abs(magnitude[1:] - magnitude[:-1]) / (time[1:] - time[:-1])
        #np.max(slope)

        return np.max(slope)
    except: return 0





def LinearTrend( data):
    #['magnitude', 'time']
    magnitude = data[0]
    time = data[1]
    try:
        regression_slope = stats.linregress(time, magnitude)[0]

        return regression_slope
    except: return 0


def Eta_color( data):
    #['magnitude', 'time', 'magnitude2']
    aligned_magnitude = data[0]
    aligned_magnitude2 = data[2]
    aligned_time = data[1]
    N = len(aligned_magnitude)
    B_Rdata = aligned_magnitude - aligned_magnitude2

    try:
        w = 1.0 / np.power(aligned_time[1:] - aligned_time[:-1], 2)
        w_mean = np.mean(w)

        N = len(aligned_time)
        sigma2 = np.var(B_Rdata)

        S1 = sum(w * (B_Rdata[1:] - B_Rdata[:-1]) ** 2)
        S2 = sum(w)

        eta_B_R = (w_mean * np.power(aligned_time[N - 1] -
                    aligned_time[0], 2) * S1 / (sigma2 * S2 * N ** 2))

        return eta_B_R
    except: return 0


 
def Eta_e( data):
    #['magnitude', 'time']
    magnitude = data[0]
    time = data[1]
    try:
        w = 1.0 / np.power(np.subtract(time[1:], time[:-1]), 2)
        w_mean = np.mean(w)

        N = len(time)
        sigma2 = np.var(magnitude)

        S1 = sum(w * (magnitude[1:] - magnitude[:-1]) ** 2)
        S2 = sum(w)

        eta_e = (w_mean * np.power(time[N - 1] -
                    time[0], 2) * S1 / (sigma2 * S2 * N ** 2))

        return eta_e
    except: return 0



###################################
###################################
"""EntroPy is a Python 3 package providing several time-efficient algorithms for computing the complexity of one-dimensional time series. 
It can be used for example to extract features from EEG signals.
EntroPy was created and is maintained by Raphael Vallat.
https://github.com/raphaelvallat/entropy
Several functions of EntroPy were borrowed from:

MNE-features: https://github.com/mne-tools/mne-features
pyEntropy: https://github.com/nikdon/pyEntropy
pyrem: https://github.com/gilestrolab/pyrem
nolds: https://github.com/CSchoel/nolds
All the credit goes to the author of these excellent packages."""

#import numpy as np
#from numba import jit
#from math import log, floor

#from .utils import _linear_regression



def petrosian_fd(x):
    """Petrosian fractal dimension.

    Parameters
    ----------
    x : list or np.array
        One dimensional time series

    Returns
    -------
    pfd : float
        Petrosian fractal dimension

    Notes
    -----
    The Petrosian algorithm can be used to provide a fast computation of
    the FD of a signal by translating the series into a binary sequence.

    The Petrosian fractal dimension of a time series :math:`x` is defined by:

    .. math:: \\frac{log_{10}(N)}{log_{10}(N) +
        log_{10}(\\frac{N}{N+0.4N_{\\Delta}})}

    where :math:`N` is the length of the time series, and
    :math:`N_{\\Delta}` is the number of sign changes in the binary sequence.

    Original code from the pyrem package by Quentin Geissmann.

    References
    ----------
    .. [1] A. Petrosian, Kolmogorov complexity of finite sequences and
        recognition of different preictal EEG patterns, in , Proceedings of the
        Eighth IEEE Symposium on Computer-Based Medical Systems, 1995,
        pp. 212-217.

    .. [2] Goh, Cindy, et al. "Comparison of fractal dimension algorithms for
        the computation of EEG biomarkers for dementia." 2nd International
        Conference on Computational Intelligence in Medicine and Healthcare
        (CIMED2005). 2005.

    Examples
    --------
    Petrosian fractal dimension.

        >>> import numpy as np
        >>> from entropy import petrosian_fd
        >>> np.random.seed(123)
        >>> x = np.random.rand(100)
        >>> print(petrosian_fd(x))
            1.0505
    """
    n = len(x)
    # Number of sign changes in the first derivative of the signal
    diff = np.ediff1d(x)
    N_delta = (diff[1:-1] * diff[0:-2] < 0).sum()
    try: p_fd = np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * N_delta)))
    except: return 0
    return p_fd


def katz_fd(x):
    """Katz Fractal Dimension.

    Parameters
    ----------
    x : list or np.array
        One dimensional time series

    Returns
    -------
    kfd : float
        Katz fractal dimension

    Notes
    -----
    The Katz Fractal dimension is defined by:

    .. math:: FD_{Katz} = \\frac{log_{10}(n)}{log_{10}(d/L)+log_{10}(n)}

    where :math:`L` is the total length of the time series and :math:`d`
    is the Euclidean distance between the first point in the
    series and the point that provides the furthest distance
    with respect to the first point.

    Original code from the mne-features package by Jean-Baptiste Schiratti
    and Alexandre Gramfort.

    References
    ----------
    .. [1] Esteller, R. et al. (2001). A comparison of waveform fractal
            dimension algorithms. IEEE Transactions on Circuits and Systems I:
            Fundamental Theory and Applications, 48(2), 177-183.

    .. [2] Goh, Cindy, et al. "Comparison of fractal dimension algorithms for
            the computation of EEG biomarkers for dementia." 2nd International
            Conference on Computational Intelligence in Medicine and Healthcare
            (CIMED2005). 2005.

    Examples
    --------
    Katz fractal dimension.

        >>> import numpy as np
        >>> from entropy import katz_fd
        >>> np.random.seed(123)
        >>> x = np.random.rand(100)
        >>> print(katz_fd(x))
            5.1214
    """
    try:
        x = np.array(x)
        dists = np.abs(np.ediff1d(x))
        ll = dists.sum()
        mean = dists.mean()
        if mean==0: return 0
        ln = np.log10(np.divide(ll, mean))
        aux_d = x - x[0]
        d = np.max(np.abs(aux_d[1:]))    
        return np.divide(ln, np.add(ln, np.log10(np.divide(d, ll))))
    except: return 0
    


@jit('float64(float64[:], int32)')
def _higuchi_fd(x, kmax):
    """Utility function for `higuchi_fd`.
    """
    n_times = x.size
    lk = np.empty(kmax)
    x_reg = np.empty(kmax)
    y_reg = np.empty(kmax)
    for k in range(1, kmax + 1):
        lm = np.empty((k,))
        for m in range(k):
            ll = 0
            n_max = floor((n_times - m - 1) / k)
            n_max = int(n_max)
            for j in range(1, n_max):
                ll += abs(x[m + j * k] - x[m + (j - 1) * k])
            ll /= k
            ll *= (n_times - 1) / (k * n_max)
            lm[m] = ll
        # Mean of lm
        m_lm = 0
        for m in range(k):
            m_lm += lm[m]
        m_lm /= k
        lk[k - 1] = m_lm
        x_reg[k - 1] = log(1. / k)
        y_reg[k - 1] = log(m_lm)
    higuchi, _ = _linear_regression(x_reg, y_reg)
    return higuchi


def higuchi_fd(x, kmax=10):
    """Higuchi Fractal Dimension.

    Parameters
    ----------
    x : list or np.array
        One dimensional time series
    kmax : int
        Maximum delay/offset (in number of samples).

    Returns
    -------
    hfd : float
        Higuchi Fractal Dimension

    Notes
    -----
    Original code from the mne-features package by Jean-Baptiste Schiratti
    and Alexandre Gramfort.

    The `higuchi_fd` function uses Numba to speed up the computation.

    References
    ----------
    .. [1] Higuchi, Tomoyuki. "Approach to an irregular time series on the
        basis of the fractal theory." Physica D: Nonlinear Phenomena 31.2
        (1988): 277-283.

    Examples
    --------
    Higuchi Fractal Dimension

        >>> import numpy as np
        >>> from entropy import higuchi_fd
        >>> np.random.seed(123)
        >>> x = np.random.rand(100)
        >>> print(higuchi_fd(x))
            2.051179
    """
    x = np.asarray(x, dtype=np.float64)
    kmax = int(kmax)
    try: hg_fd = _higuchi_fd(x, kmax)
    except: return 0
    return hg_fd



#import numpy as np
#from numba import jit
#from math import factorial, log
#from sklearn.neighbors import KDTree
#from scipy.signal import periodogram, welch

#from .utils import _embed

#all = ['perm_entropy', 'spectral_entropy', 'svd_entropy', 'app_entropy',
#        'sample_entropy']

###########Entropy
def perm_entropy(x, order=3, delay=1, normalize=False):
    """Permutation Entropy.

    Parameters
    ----------
    x : list or np.array
        One-dimensional time series of shape (n_times)
    order : int
        Order of permutation entropy
    delay : int
        Time delay
    normalize : bool
        If True, divide by log2(order!) to normalize the entropy between 0
        and 1. Otherwise, return the permutation entropy in bit.

    Returns
    -------
    pe : float
        Permutation Entropy

    Notes
    -----
    The permutation entropy is a complexity measure for time-series first
    introduced by Bandt and Pompe in 2002 [1]_.

    The permutation entropy of a signal :math:`x` is defined as:

    .. math:: H = -\\sum p(\\pi)log_2(\\pi)

    where the sum runs over all :math:`n!` permutations :math:`\\pi` of order
    :math:`n`. This is the information contained in comparing :math:`n`
    consecutive values of the time series. It is clear that
    :math:`0 ≤ H (n) ≤ log_2(n!)` where the lower bound is attained for an
    increasing or decreasing sequence of values, and the upper bound for a
    completely random system where all :math:`n!` possible permutations appear
    with the same probability.

    The embedded matrix :math:`Y` is created by:

    .. math:: y(i)=[x_i,x_{i+delay}, ...,x_{i+(order-1) * delay}]

    .. math:: Y=[y(1),y(2),...,y(N-(order-1))*delay)]^T


    References
    ----------
    .. [1] Bandt, Christoph, and Bernd Pompe. "Permutation entropy: a
            natural complexity measure for time series." Physical review letters
            88.17 (2002): 174102.

    Examples
    --------
    1. Permutation entropy with order 2

        >>> from entropy import perm_entropy
        >>> x = [4, 7, 9, 10, 6, 11, 3]
        >>> # Return a value in bit between 0 and log2(factorial(order))
        >>> print(perm_entropy(x, order=2))
            0.918
    2. Normalized permutation entropy with order 3

        >>> from entropy import perm_entropy
        >>> x = [4, 7, 9, 10, 6, 11, 3]
        >>> # Return a value comprised between 0 and 1.
        >>> print(perm_entropy(x, order=3, normalize=True))
            0.589
    """
    x = np.array(x)
    ran_order = range(order)
    hashmult = np.power(order, ran_order)
    # Embed x and sort the order of permutations
    try:sorted_idx = _embed(x, order=order, delay=delay).argsort(kind='quicksort')
    except: return 0          
    # Associate unique integer to each permutations
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
    # Return the counts
    _, c = np.unique(hashval, return_counts=True)
    # Use np.true_divide for Python 2 compatibility
    p = np.true_divide(c, c.sum())
    pe = -np.multiply(p, np.log2(p)).sum()
    if normalize:
        pe /= np.log2(factorial(order))
    return pe


def spectral_entropy(x, sf, method='fft', nperseg=None, normalize=True):
    """Spectral Entropy.

    Parameters
    ----------
    x : list or np.array
        One-dimensional time series of shape (n_times)
    sf : float
        Sampling frequency
    method : str
        Spectral estimation method ::

        'fft' : Fourier Transform (via scipy.signal.periodogram)
        'welch' : Welch periodogram (via scipy.signal.welch)

    nperseg : str or int
        Length of each FFT segment for Welch method.
        If None, uses scipy default of 256 samples.
    normalize : bool
        If True, divide by log2(psd.size) to normalize the spectral entropy
        between 0 and 1. Otherwise, return the spectral entropy in bit.

    Returns
    -------
    se : float
        Spectral Entropy

    Notes
    -----
    Spectral Entropy is defined to be the Shannon Entropy of the Power
    Spectral Density (PSD) of the data:

    .. math:: H(x, sf) =  -\\sum_{f=0}^{f_s/2} PSD(f) log_2[PSD(f)]

    Where :math:`PSD` is the normalised PSD, and :math:`f_s` is the sampling
    frequency.

    References
    ----------
    .. [1] Inouye, T. et al. (1991). Quantification of EEG irregularity by
        use of the entropy of the power spectrum. Electroencephalography
        and clinical neurophysiology, 79(3), 204-210.

    Examples
    --------
    1. Spectral entropy of a pure sine using FFT

        >>> from entropy import spectral_entropy
        >>> import numpy as np
        >>> sf, f, dur = 100, 1, 4
        >>> N = sf * duration # Total number of discrete samples
        >>> t = np.arange(N) / sf # Time vector
        >>> x = np.sin(2 * np.pi * f * t)
        >>> print(np.round(spectral_entropy(x, sf, method='fft'), 2)
            0.0

    2. Spectral entropy of a random signal using Welch's method

        >>> from entropy import spectral_entropy
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> x = np.random.rand(3000)
        >>> print(spectral_entropy(x, sf=100, method='welch'))
            9.939

    3. Normalized spectral entropy

        >>> print(spectral_entropy(x, sf=100, method='welch', normalize=True))
            0.995
    """
    x = np.array(x)
    # Compute and normalize power spectrum
    if method == 'fft':
       try: _, psd = periodogram(x, sf)
       except: return 0
    elif method == 'welch':
        try: _, psd = welch(x, sf, nperseg=nperseg)
        except: return 0
    try:
        if psd.sum()==0: return 0
        psd_norm = np.divide(psd, psd.sum())
        if psd_norm==0: return 0
        se = -np.multiply(psd_norm, np.log2(psd_norm)).sum()
    except: return 0
    if normalize:
        se /= np.log2(psd_norm.size)
    return se


def svd_entropy(x, order=3, delay=1, normalize=True):
    """Singular Value Decomposition entropy.

    Parameters
    ----------
    x : list or np.array
        One-dimensional time series of shape (n_times)
    order : int
        Order of permutation entropy
    delay : int
        Time delay
    normalize : bool
        If True, divide by log2(order!) to normalize the entropy between 0
        and 1. Otherwise, return the permutation entropy in bit.

    Returns
    -------
    svd_e : float
        SVD Entropy

    Notes
    -----
    SVD entropy is an indicator of the number of eigenvectors that are needed
    for an adequate explanation of the data set. In other words, it measures
    the dimensionality of the data.

    The SVD entropy of a signal :math:`x` is defined as:

    .. math::
        H = -\\sum_{i=1}^{M} \\overline{\\sigma}_i log_2(\\overline{\\sigma}_i)

    where :math:`M` is the number of singular values of the embedded matrix
    :math:`Y` and :math:`\\sigma_1, \\sigma_2, ..., \\sigma_M` are the
    normalized singular values of :math:`Y`.

    The embedded matrix :math:`Y` is created by:

    .. math:: y(i)=[x_i,x_{i+delay}, ...,x_{i+(order-1) * delay}]

    .. math:: Y=[y(1),y(2),...,y(N-(order-1))*delay)]^T

    Examples
    --------
    1. SVD entropy with order 2

        >>> from entropy import svd_entropy
        >>> x = [4, 7, 9, 10, 6, 11, 3]
        >>> # Return a value in bit between 0 and log2(factorial(order))
        >>> print(svd_entropy(x, order=2))
            0.762

    2. Normalized SVD entropy with order 3

        >>> from entropy import svd_entropy
        >>> x = [4, 7, 9, 10, 6, 11, 3]
        >>> # Return a value comprised between 0 and 1.
        >>> print(svd_entropy(x, order=3, normalize=True))
            0.687
    """
    try:
        x = np.array(x)
        mat = _embed(x, order=order, delay=delay)
        W = np.linalg.svd(mat, compute_uv=False)
        # Normalize the singular values
        W /= sum(W)
        svd_e = -np.multiply(W, np.log2(W)).sum()
        if normalize:
            svd_e /= np.log2(order)
        return svd_e
    except: return 0


def _app_samp_entropy(x, order, metric='chebyshev', approximate=True):
    """Utility function for `app_entropy`` and `sample_entropy`.
    """
    try:
        _all_metrics = KDTree.valid_metrics
        if metric not in _all_metrics:
            raise ValueError('The given metric (%s) is not valid. The valid '
                                'metric names are: %s' % (metric, _all_metrics))
        phi = np.zeros(2)
        r = 0.2 * np.std(x, axis=-1, ddof=1)

        # compute phi(order, r)
        _emb_data1 = _embed(x, order, 1)
        if approximate:
            emb_data1 = _emb_data1
        else:
            emb_data1 = _emb_data1[:-1]
        count1 = KDTree(emb_data1, metric=metric).query_radius(emb_data1, r,
                                                                count_only=True
                                                                ).astype(np.float64)
        # compute phi(order + 1, r)
        emb_data2 = _embed(x, order + 1, 1)
        count2 = KDTree(emb_data2, metric=metric).query_radius(emb_data2, r,
                                                                count_only=True
                                                                ).astype(np.float64)
        if approximate:
            phi[0] = np.mean(np.log(count1 / emb_data1.shape[0]))
            phi[1] = np.mean(np.log(count2 / emb_data2.shape[0]))
        else:
            phi[0] = np.mean((count1 - 1) / (emb_data1.shape[0] - 1))
            phi[1] = np.mean((count2 - 1) / (emb_data2.shape[0] - 1))
        return phi
    except: return 0


@jit('f8(f8[:], i4, f8)', nopython=True)
def _numba_sampen(x, mm=2, r=0.2):
    """
    Fast evaluation of the sample entropy using Numba.
    """
    n = x.size
    n1 = n - 1
    mm += 1
    mm_dbld = 2 * mm

    # Define threshold
    r *= x.std()

    # initialize the lists
    run = [0] * n
    run1 = run[:]
    r1 = [0] * (n * mm_dbld)
    a = [0] * mm
    b = a[:]
    p = a[:]

    for i in range(n1):
        nj = n1 - i

        for jj in range(nj):
            j = jj + i + 1
            if abs(x[j] - x[i]) < r:
                run[jj] = run1[jj] + 1
                m1 = mm if mm < run[jj] else run[jj]
                for m in range(m1):
                    a[m] += 1
                    if j < n1:
                        b[m] += 1
            else:
                run[jj] = 0
        for j in range(mm_dbld):
            run1[j] = run[j]
            r1[i + n * j] = run[j]
        if nj > mm_dbld - 1:
            for j in range(mm_dbld, nj):
                run1[j] = run[j]

    m = mm - 1

    while m > 0:
        b[m] = b[m - 1]
        m -= 1

    b[0] = n * n1 / 2
    a = np.array([float(aa) for aa in a])
    b = np.array([float(bb) for bb in b])
    p = np.true_divide(a, b)
    return -log(p[-1])


def app_entropy(x, order=2, metric='chebyshev'):
    """Approximate Entropy.

    Parameters
    ----------
    x : list or np.array
        One-dimensional time series of shape (n_times)
    order : int (default: 2)
        Embedding dimension.
    metric : str (default: chebyshev)
        Name of the metric function used with
        :class:`~sklearn.neighbors.KDTree`. The list of available
        metric functions is given by: ``KDTree.valid_metrics``.

    Returns
    -------
    ae : float
        Approximate Entropy.

    Notes
    -----
    Original code from the mne-features package.

    Approximate entropy is a technique used to quantify the amount of
    regularity and the unpredictability of fluctuations over time-series data.

    Smaller values indicates that the data is more regular and predictable.

    The value of :math:`r` is set to :math:`0.2 * \\text{std}(x)`.

    Code adapted from the mne-features package by Jean-Baptiste Schiratti
    and Alexandre Gramfort.

    References
    ----------
    .. [1] Richman, J. S. et al. (2000). Physiological time-series analysis
            using approximate entropy and sample entropy. American Journal of
            Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.

    1. Approximate entropy with order 2.

        >>> from entropy import app_entropy
        >>> import numpy as np
        >>> np.random.seed(1234567)
        >>> x = np.random.rand(3000)
        >>> print(app_entropy(x, order=2))
            2.075
    """
    try:
        phi = _app_samp_entropy(x, order=order, metric=metric, approximate=True)
        return np.subtract(phi[0], phi[1])
    except: return 0


def sample_entropy(x, order=2, metric='chebyshev'):
    """Sample Entropy.

    Parameters
    ----------
    x : list or np.array
        One-dimensional time series of shape (n_times)
    order : int (default: 2)
        Embedding dimension.
    metric : str (default: chebyshev)
        Name of the metric function used with KDTree. The list of available
        metric functions is given by: `KDTree.valid_metrics`.

    Returns
    -------
    se : float
        Sample Entropy.

    Notes
    -----
    Sample entropy is a modification of approximate entropy, used for assessing
    the complexity of physiological time-series signals. It has two advantages
    over approximate entropy: data length independence and a relatively
    trouble-free implementation. Large values indicate high complexity whereas
    smaller values characterize more self-similar and regular signals.

    Sample entropy of a signal :math:`x` is defined as:

    .. math:: H(x, m, r) = -log\\frac{C(m + 1, r)}{C(m, r)}

    where :math:`m` is the embedding dimension (= order), :math:`r` is
    the radius of the neighbourhood (default = :math:`0.2 * \\text{std}(x)`),
    :math:`C(m + 1, r)` is the number of embedded vectors of length
    :math:`m + 1` having a Chebyshev distance inferior to :math:`r` and
    :math:`C(m, r)` is the number of embedded vectors of length
    :math:`m` having a Chebyshev distance inferior to :math:`r`.

    Note that if metric == 'chebyshev' and x.size < 5000 points, then the
    sample entropy is computed using a fast custom Numba script. For other
    metric types or longer time-series, the sample entropy is computed using
    a code from the mne-features package by Jean-Baptiste Schiratti
    and Alexandre Gramfort (requires sklearn).

    References
    ----------
    .. [1] Richman, J. S. et al. (2000). Physiological time-series analysis
            using approximate entropy and sample entropy. American Journal of
            Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.

    Examples
    --------
    1. Sample entropy with order 2.

        >>> from entropy import sample_entropy
        >>> import numpy as np
        >>> np.random.seed(1234567)
        >>> x = np.random.rand(3000)
        >>> print(sample_entropy(x, order=2))
            2.192

    2. Sample entropy with order 3 using the Euclidean distance.

        >>> from entropy import sample_entropy
        >>> import numpy as np
        >>> np.random.seed(1234567)
        >>> x = np.random.rand(3000)
        >>> print(sample_entropy(x, order=3, metric='euclidean'))
            2.725
    """
    x = np.asarray(x, dtype=np.float64)
    if metric == 'chebyshev' and x.size < 5000:
        try: return _numba_sampen(x, mm=order, r=0.2)
        except: return 0
    else:
        phi = _app_samp_entropy(x, order=order, metric=metric,
                                approximate=False)
        try: return -np.log(np.divide(phi[1], phi[0]))
        except: return 0


    #utils entropy
def _embed(x, order=3, delay=1):
    """Time-delay embedding.

    Parameters
    ----------
    x : 1d-array, shape (n_times)
        Time series
    order : int
        Embedding dimension (order)
    delay : int
        Delay.

    Returns
    -------
    embedded : ndarray, shape (n_times - (order - 1) * delay, order)
        Embedded time-series.
    """
    N = len(x)
    if order * delay > N:
        print("Error: order * delay should be lower than x.size") #raise ValueError
        return 0
    if delay < 1:
        print("Delay has to be at least 1.")
        return 0
    if order < 2:
        print("Order has to be at least 2.")
        return 0
    Y = np.zeros((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[i * delay:i * delay + Y.shape[1]]
    return Y.T


@jit('UniTuple(float64, 2)(float64[:], float64[:])', nopython=True)
def _linear_regression(x, y):
    """Fast linear regression using Numba.

    Parameters
    ----------
    x, y : ndarray, shape (n_times,)
        Variables

    Returns
    -------
    slope : float
        Slope of 1D least-square regression.
    intercept : float
        Intercept
    """
    n_times = x.size
    sx2 = 0
    sx = 0
    sy = 0
    sxy = 0
    for j in range(n_times):
        sx2 += x[j] ** 2
        sx += x[j]
        sxy += x[j] * y[j]
        sy += y[j]
    den = n_times * sx2 - (sx ** 2)
    num = n_times * sxy - sx * sy
    slope = num / den
    intercept = np.mean(y) - slope * np.mean(x)
    return slope, intercept


@jit('i8[:](f8, f8, f8)', nopython=True)
def _log_n(min_n, max_n, factor):
    """
    Creates a list of integer values by successively multiplying a minimum
    value min_n by a factor > 1 until a maximum value max_n is reached.

    Used for detrended fluctuation analysis (DFA).

    Function taken from the nolds python package
    (https://github.com/CSchoel/nolds) by Christopher Scholzel.

    Parameters
    ----------
    min_n (float):
        minimum value (must be < max_n)
    max_n (float):
        maximum value (must be > min_n)
    factor (float):
        factor used to increase min_n (must be > 1)

    Returns
    -------
    list of integers:
        min_n, min_n * factor, min_n * factor^2, ... min_n * factor^i < max_n
        without duplicates
    """
    max_i = int(floor(log(1.0 * max_n / min_n) / log(factor)))
    ns = [min_n]
    for i in range(max_i + 1):
        n = int(floor(min_n * (factor ** i)))
        if n > ns[-1]:
            ns.append(n)
    return np.array(ns, dtype=np.int64)

