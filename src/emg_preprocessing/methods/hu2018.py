"""Offline Hu 2018 baseline, with explicit numerical filter conventions."""
import numpy as np
from scipy.signal import bilinear_zpk, butter, sosfiltfilt, zpk2sos


def design_filters(config):
    fs = config['sample_rate_hz']
    bp, lp = config['bandpass'], config['lowpass']
    bandpass = butter(bp['prototype_order'], [bp['low_hz'], bp['high_hz']],
                      btype='bandpass', fs=fs, output='sos')
    # H(s) = wn**2 / (s + wn)**2: critically damped, NOT Butterworth.
    # Prewarp the declared single-pass -3 dB cutoff before bilinear transform.
    warped = 2 * fs * np.tan(np.pi * lp['cutoff_hz'] / fs)
    wn = warped / np.sqrt(np.sqrt(2) - 1)
    z, p, k = bilinear_zpk([], [-wn, -wn], wn**2, fs=fs)
    return bandpass, zpk2sos(z, p, k)


def preprocess(bipolar, config):
    """Invalid/constant channels remain NaN and carry an explicit flag."""
    x = np.asarray(bipolar, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(config['channels']):
        raise ValueError('Expected samples x selected channels')
    padlen = int(round(config['zero_phase']['pad_seconds'] * config['sample_rate_hz']))
    if len(x) <= padlen:
        raise ValueError(f'Record has {len(x)} samples; needs more than padlen={padlen}')
    flags = np.zeros(x.shape[1], dtype=np.uint8)
    finite = np.isfinite(x).all(axis=0)
    flags[~finite] |= 1
    constant = np.zeros(x.shape[1], dtype=bool)
    constant[finite] = np.ptp(x[:, finite], axis=0) == 0
    flags[constant] |= 2
    good = flags == 0
    bandpass, lowpass = design_filters(config)
    filtered = np.full_like(x, np.nan)
    envelope = np.full_like(x, np.nan)
    if good.any():
        filtered[:, good] = sosfiltfilt(bandpass, x[:, good], axis=0,
                                       padtype=config['zero_phase']['padtype'], padlen=padlen)
        envelope[:, good] = sosfiltfilt(lowpass, np.abs(filtered[:, good]), axis=0,
                                       padtype=config['zero_phase']['padtype'], padlen=padlen)
        if not np.isfinite(filtered[:, good]).all() or not np.isfinite(envelope[:, good]).all():
            raise ValueError('Nonfinite filter output')
    # Odd extension can create small negative endpoint values after rectification.
    negatives = np.sum(envelope < 0, axis=0)
    envelope = np.maximum(envelope, 0)
    edge_valid = np.ones(len(x), dtype=bool)
    margin = int(np.ceil(config['edge_margin_s'] * config['sample_rate_hz']))
    if margin:
        edge_valid[:margin] = False
        edge_valid[-margin:] = False
    return filtered, envelope, flags, edge_valid, negatives
