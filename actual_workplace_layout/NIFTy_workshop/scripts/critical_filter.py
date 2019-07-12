import numpy as np
import nifty5 as ift
from just_plot import *
from responses import *
from generate_data import *


np.random.seed(42)

position_space = ift.RGSpace([256,256])
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
    'im':  -2,  # y-intercept mean, in-/decrease for more/less contrast
    'iv': 2.   # y-intercept variance
}
A = ift.SLAmplitude(**dct)

correlated_field = ift.CorrelatedField(position_space, A)
# interactive plotting
# plotting correlated_field(ift.from_random('normal',correlated_field.target))

### SETTING UP SPECIFIC SCENARIO ####

R = checkerboard_response(position_space)
# R = ift.GeometryRemover(position_space)

data_space = R.target
signal = correlated_field
signal_response = R(correlated_field)


# Set up likelihood and draw data from the model
N = ift.ScalingOperator(0.1, data_space)
data, ground_truth = generate_gaussian_data(signal_response, N)

plot_prior_samples_2d(5, signal, R, correlated_field, A, 'gauss', N=N)

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
N_samples = 5

# Draw new samples to approximate the KL five times
for i in range(10):
    # Draw new samples and minimize KL
    KL = ift.MetricGaussianKL(mean, H, N_samples)
    KL, convergence = minimizer(KL)
    mean = KL.position

# Draw posterior samples and plotting
N_posterior_samples = 30
KL = ift.MetricGaussianKL(mean, H, N_posterior_samples)
plot_reconstruction_2d(data, ground_truth, KL, signal, R, A)

