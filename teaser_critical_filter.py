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
from helpers import plot_WF, power_plot, generate_mysterious_data

np.random.seed(42)

position_space = ift.RGSpace(256)
harmonic_space = position_space.get_default_codomain()

HT = ift.HarmonicTransformOperator(harmonic_space, target=position_space)

power_space = ift.PowerSpace(harmonic_space)

# Set up an amplitude operator for the field
# We want to set up a model for the amplitude spectrum with some magic numbers
dct = {
    'target': power_space,
    'n_pix': 64,  # 64 spectral bins
    # Spectral smoothness (affects Gaussian process part)
    'a': 10,  # relatively high variance of spectral curvature
    'k0': .2,  # quefrency mode below which cepstrum flattens
    # Power-law part of spectrum:
    'sm': -4,  # preferred power-law slope
    'sv': .6,  # low variance of power-law slope
    'im':  -6,  # y-intercept mean, in-/decrease for more/less contrast
    'iv': 2.   # y-intercept variance
}
A = ift.SLAmplitude(**dct)

correlated_field = ift.CorrelatedField(position_space, A)

### SETTING UP SPECIFIC SCENARIO ####

R = ift.GeometryRemover(position_space)
data_space = R.target

signal_response = R(correlated_field)


# Set up likelihood and load data
N = ift.ScalingOperator(0.1, data_space)

data, ground_truth = generate_mysterious_data(position_space)
data = ift.from_global_data(data_space, data)

likelihood = ift.GaussianEnergy(mean=data,
                                inverse_covariance=N.inverse)(signal_response)


#### SOLVING PROBLEM ####
ic_sampling = ift.GradientNormController(iteration_limit=100)
ic_newton = ift.GradInfNormController(
    name='Newton', tol=1e-6, iteration_limit=30)
minimizer = ift.NewtonCG(ic_newton)

H = ift.StandardHamiltonian(likelihood, ic_sampling)

initial_mean = ift.MultiField.full(H.domain, 0.)
mean = initial_mean

# number of samples used to estimate the KL
N_samples = 10

# Draw new samples to approximate the KL ten times
for i in range(10):
    # Draw new samples and minimize KL
    KL = ift.MetricGaussianKL(mean, H, N_samples)
    KL, convergence = minimizer(KL)
    mean = KL.position

# Draw posterior samples and plotting
N_posterior_samples = 10
KL = ift.MetricGaussianKL(mean, H, N_posterior_samples)

# Plotting the reconstruction result
ground_truth = ift.from_global_data(position_space, ground_truth)
posterior_samples = [correlated_field(KL.position+samp) for samp in KL.samples]
mean = 0.*posterior_samples[0]
for p in posterior_samples:
    mean = mean + p/len(posterior_samples)
plot_WF('unknown_power', ground_truth, data, m=mean, samples=posterior_samples)

# Plotting the reconstruction of the power spectrum
mysterious_spectrum = lambda k: 5/((7**2 - k**2)**2 + 3**2*k**2)
ground_truth_spectrum = ift.from_global_data(power_space, mysterious_spectrum(power_space.k_lengths))
posterior_power_samples = [A.force(KL.position+samp)**2 for samp in KL.samples]
power_mean = 0.*posterior_power_samples[0]
for p in posterior_power_samples:
    power_mean = power_mean + p/len(posterior_power_samples)
power_plot('power_reconstruction', ground_truth_spectrum, power_mean, posterior_power_samples)

