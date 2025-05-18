import numpy as np
from sklearn.discriminant_analysis import _cov
from mne.filter import filter_data
from scipy.signal import welch


def channel_covariance(X: np.ndarray) -> np.ndarray:
    """ Compute the covariance matrix over channels.

    Args:
        X (np.ndarray): Input data of shape (n_samples, n_channels, n_times).

    Returns:
        np.ndarray: Covariance matrix of shape (n_samples, n_channels, n_channels).
    """
    cov = [_cov(x.T, shrinkage='auto') for x in X]
    cov = np.array(cov)  # shape (n_samples, n_channels, n_channels)
    return cov


def power_spectra(X: np.ndarray, sfreq=250, f_max=40, chan_aggr='mean') -> np.ndarray:
    """ Compute the power spectra using Welch's method.

    Args:
        X (np.ndarray): Input data of shape (n_samples, n_channels, n_times).
        sfreq (int, optional): Sampling frequency in Hz. Defaults to 250.
        f_max (int, optional): Maximum frequency to consider. Defaults to 40.
        chan_aggr (str, optional): Channel aggregation method ('mean', 'median', 'flatten'). Defaults to 'mean'.

    Returns:
        tuple: Tuple containing:
            - np.ndarray: PSD of shape (n_samples, n_frequencies) or (n_samples, n_channels, n_frequencies).
            - np.ndarray: Frequencies corresponding to the PSD.
    """
    psds = []
    freqs = None

    # compute frequency components for each time-series
    for sample in X:
        sample_psds = []
        for channel in sample:
            f, pxx = welch(channel, sfreq, nperseg=sfreq*2)
            if freqs is None:
                freqs = f[f <= f_max]
            sample_psds.append(pxx[f <= f_max])
        psds.append(sample_psds)
    psds = np.array(psds)  # shape (n_samples, n_channels, n_frequencies)

    if chan_aggr == 'mean':
        psds = np.mean(psds, axis=1)  # shape (n_samples, n_frequencies)
    elif chan_aggr == 'median':
        psds = np.median(psds, axis=1)  # shape (n_samples, n_frequencies)
    elif chan_aggr == 'flatten':
        psds = psds.reshape(psds.shape[0], -1)  # shape (n_samples, n_channels * n_frequencies)

    return psds, freqs


def split_fbands(X: np.ndarray, fbands=[(4,8), (8,13), (13,30), (30,40)], sfreq=250) -> np.ndarray:
    """ Split signal into frequency bands.

    Args:
        X (np.ndarray): Input data of shape (n_samples, n_channels, n_times).
        fbands (list, optional): List of frequency bands. Defaults to [(4,8), (8,13), (13,30), (30,40)].

    Returns:
        np.ndarray: Array of shape (n_samples, n_bands, n_channels, n_frequencies) containing the filtered data.
    """
    
    X_bands = []
    for i in range(X.shape[0]):
        x_band = []
        for j in range(X.shape[1]):
            x_band.append(filter_data(X[i, j], sfreq=sfreq, l_freq=None, h_freq=None))
        X_bands.append(x_band)

    return np.array(X_bands)

# TODO add statistical features, e.g. mne_features

# TODO add time-frequency features



# OUR FEATURES ------------------------------------------------------------------------- 

def select_channels(X: np.ndarray, channels) -> np.ndarray:
    """ Select only given channels from 

    Args:
        X (np.ndarray): Input data of shape (n_samples, n_channels, n_times).
        channels (int list): Channel indices to select (i think the channels are called 'ch1' until 'ch8', and refer to ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8'])

    Returns:
        np.ndarray: Array of shape (n_samples, n_selected_channels, n_times)
    """

    return X[:, channels, :]

def split_time_bands(X: np.ndarray,
                     time_bands=[(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
                    ) -> np.ndarray:
    """ Split time frames into given sections and compute the average over these as features

    Args:
        X (np.ndarray): Input data of shape (n_samples, n_channels, n_times).
        time_bands: List of tuples containing start and end of the time band in percent

    Returns:
        np.ndarray: Array of shape (n_samples, n_channels, n_time_bands)
    """
    n_samples, n_channels, n_times = X.shape

    # Convert percents to integer indices
    index_tuples = [
        (int(np.floor(start * n_times)), int(np.ceil(end * n_times)))
        for start, end in time_bands
    ]

    # For each band, slice and compute mean over the time axis
    band_means = []
    for start_idx, end_idx in index_tuples:
        # X[:, :, start_idx:end_idx] has shape (n_samples, n_channels, band_length)
        band_mean = X[:, :, start_idx:end_idx].mean(axis=2)
        # band_mean has shape (n_samples, n_channels)
        band_means.append(band_mean)

    # Stack along a new third dimension → (n_samples, n_channels, n_time_bands)
    return np.stack(band_means, axis=2)

def flatten_channels(features_per_channel: np.ndarray) -> np.ndarray:
    """ Transform input of shape (n_samples, n_channels, n_fatures) to (n_samples, n_features*n_channels) """
    n_samples, n_channels, n_features = features_per_channel.shape
    return features_per_channel.reshape((n_samples, n_features*n_channels))
    
