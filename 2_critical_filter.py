import numpy as np

import nifty5 as ift

np.random.seed(42)

position_space = ift.RGSpace(2*(256,))
harmonic_space = position_space.get_default_codomain()
HT = ift.HarmonicTransformOperator(harmonic_space, target=position_space)
power_space = ift.PowerSpace(harmonic_space)

A = ift.SLAmplitude(
    **{
        'target': power_space,
        'n_pix': 64,  # 64 spectral bins
        # Smoothness of spectrum
        'a': 10,  # relatively high variance of spectral curvature
        'k0': .2,  # quefrency mode below which cepstrum flattens
        # Power-law part of spectrum
        'sm': -4,  # preferred power-law slope
        'sv': .6,  # low variance of power-law slope
        'im': -2,  # y-intercept mean, in-/decrease for more/less contrast
        'iv': 2.  # y-intercept variance
    })
