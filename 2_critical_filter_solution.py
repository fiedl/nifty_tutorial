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

import nifty5 as ift
from helpers import (checkerboard_response, generate_gaussian_data,
                     plot_prior_samples_2d, plot_reconstruction_2d)

np.random.seed(42)

position_space = ift.RGSpace(2*(256,))
harmonic_space = position_space.get_default_codomain()
HT = ift.HarmonicTransformOperator(harmonic_space, target=position_space)
power_space = ift.PowerSpace(harmonic_space)

# Set up generative model
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
signal = ift.CorrelatedField(position_space, A)
R = checkerboard_response(position_space)

data_space = R.target
signal_response = R @ signal

# Set up likelihood and generate data from the model
N = ift.ScalingOperator(0.1, data_space)
data, ground_truth = generate_gaussian_data(signal_response, N)

plot_prior_samples_2d(5, signal, R, signal, A, 'gauss', N=N)

likelihood = ift.GaussianEnergy(
    mean=data, inverse_covariance=N.inverse)(signal_response)

# Solve inference problem
ic_sampling = ift.GradientNormController(iteration_limit=100)
ic_newton = ift.GradInfNormController(
    name='Newton', tol=1e-6, iteration_limit=30)
minimizer = ift.NewtonCG(ic_newton)
H = ift.StandardHamiltonian(likelihood, ic_sampling)

initial_mean = ift.MultiField.full(H.domain, 0.)
mean = initial_mean

# Draw five samples and minimize KL, iterate 10 times
for _ in range(10):
    KL = ift.MetricGaussianKL(mean, H, 5)
    KL, convergence = minimizer(KL)
    mean = KL.position

# Draw posterior samples and plot
N_posterior_samples = 30
KL = ift.MetricGaussianKL(mean, H, N_posterior_samples)
plot_reconstruction_2d(data, ground_truth, KL, signal, R, A, 'criticalfilter')
