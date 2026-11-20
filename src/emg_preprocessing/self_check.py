"""Numerical checks for frequency response, phase, envelope, and bad channels."""
import json

import numpy as np
from scipy.signal import sosfreqz

from .methods.hu2018 import design_filters, preprocess
from .pipeline import DEFAULT_CONFIG, validate_config


def run_checks():
    config = json.loads(DEFAULT_CONFIG.read_text(encoding='utf-8'))
    validate_config(config)
    bandpass, lowpass = design_filters(config)
    _, response = sosfreqz(lowpass, worN=np.array([2.0]), fs=2000)
    np.testing.assert_allclose(abs(response[0]), 1 / np.sqrt(2), rtol=1e-8)
    _, response = sosfreqz(bandpass, worN=np.array([2., 100., 900.]), fs=2000)
    effective = abs(response) ** 2
    assert effective[0] < 1e-6 and effective[1] > .99 and effective[2] < .001
    time = np.arange(32000) / 2000
    amplitude = 1 + .25 * np.sin(2 * np.pi * .5 * time)
    carrier = np.sin(2 * np.pi * 100 * time)
    x = np.tile((amplitude * carrier)[:, None], (1, len(config['channels'])))
    x[:, 0] = np.nan
    x[:, 1] = 0
    filtered, envelope, flags, edge_valid, _ = preprocess(x, config)
    np.testing.assert_array_equal(flags[:2], [1, 2])
    assert np.all(flags[2:] == 0) and np.isnan(envelope[:, :2]).all()
    interior = (time > 3) & (time < 13)
    assert np.corrcoef(envelope[interior, 2], amplitude[interior])[0, 1] > .999
    np.testing.assert_allclose(envelope[interior, 2].mean(), 2 / np.pi, rtol=.015)
    phase_error = np.arctan2(np.dot(filtered[interior, 2], np.cos(2 * np.pi * 100 * time[interior])),
                             np.dot(filtered[interior, 2], carrier[interior]))
    assert abs(phase_error) < .001
    assert edge_valid.sum() == len(time) - 4000
    assert (envelope[:, 2:] >= 0).all()
    try:
        preprocess(x[:100], config)
    except ValueError:
        pass
    else:
        raise AssertionError('Short record must be rejected explicitly')
    print('PASS: cutoff, band rejection, zero phase, envelope, channel flags, edge mask, short-record rejection')
