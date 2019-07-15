# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

import helpers as h
import nifty5 as ift

seeds = [123, 42, 42]
name = ['bernoulli', 'gauss', 'poisson']
for mode in [0, 1, 2]:
    np.random.seed(seeds[mode])

    position_space = ift.RGSpace([256, 256])
    harmonic_space = position_space.get_default_codomain()
    power_space = ift.PowerSpace(harmonic_space)
    HT = ift.HarmonicTransformOperator(harmonic_space, target=position_space)

    # Set up an amplitude operator for the field
    dct = {
        'target': power_space,
        'n_pix': 64,  # 64 spectral bins
        # Spectral smoothness (affects Gaussian process part)
        'a': 10,  # relatively high variance of spectral curvature
        'k0': .2,  # quefrency mode below which cepstrum flattens
        # Power-law part of spectrum:
        'sm': -4,  # preferred power-law slope
        'sv': .6,  # low variance of power-law slope
        'im': -3,  # y-intercept mean, in-/decrease for more/less contrast
        'iv': 2.  # y-intercept variance
    }
    A = ift.SLAmplitude(**dct)
    correlated_field = ift.CorrelatedField(position_space, A)

    dct = {}
    if mode == 0:
        signal = correlated_field.sigmoid()
        R = h.checkerboard_response(position_space)
    elif mode == 1:
        signal = correlated_field.exp()
        R = h.radial_tomography_response(position_space, lines_of_sight=256)
        N = ift.ScalingOperator(5., R.target)
        dct['N'] = N
    elif mode == 2:
        signal = correlated_field.exp()
        R = h.exposure_response(position_space)
    h.plot_prior_samples_2d(5, signal, R, correlated_field, A, name[mode],
                            **dct)
    signal_response = R @ signal
    if mode == 0:
        signal_response = signal_response.clip(1e-5, 1 - 1e-5)
        data, ground_truth = h.generate_bernoulli_data(signal_response)
        likelihood = ift.BernoulliEnergy(data) @ signal_response
    elif mode == 1:
        data, ground_truth = h.generate_gaussian_data(signal_response, N)
        likelihood = ift.GaussianEnergy(data, N) @ signal_response
    elif mode == 2:
        data, ground_truth = h.generate_poisson_data(signal_response)
        likelihood = ift.PoissonianEnergy(data) @ signal_response

    # Solve inference problem
    ic_sampling = ift.GradientNormController(iteration_limit=100)
    ic_newton = ift.GradInfNormController(
        name='Newton', tol=1e-6, iteration_limit=50)
    minimizer = ift.NewtonCG(ic_newton)
    H = ift.StandardHamiltonian(likelihood, ic_sampling)
    initial_mean = ift.MultiField.full(H.domain, 0.)
    mean = initial_mean
    N_samples = 5 if mode in [0, 2] else 10
    for _ in range(5):
        # Draw new samples and minimize KL
        KL = ift.MetricGaussianKL(mean, H, N_samples)
        KL, convergence = minimizer(KL)
        mean = KL.position
    N_posterior_samples = 30
    KL = ift.MetricGaussianKL(mean, H, N_posterior_samples)
    h.plot_reconstruction_2d(data, ground_truth, KL, signal, R, A, name[mode])
