from responses import exposure_response
from generate_data import generate_poisson_data
from plotting_aachen import plot_prior_samples_2d, plot_reconstruction_2d
import nifty5 as ift
import numpy as np


np.random.seed(42)

position_space = ift.RGSpace([256,256])
harmonic_space = position_space.get_default_codomain()
power_space = ift.PowerSpace(harmonic_space)

# Building model
HT = ift.HarmonicTransformOperator(harmonic_space, target=position_space)

# Set up an amplitude operator for the field
# We want to set up a model for the amplitude spectrum with some magic numbers
dct = {
    'target': power_space,
    'n_pix': 64,  # 64 spectral bins
    # Spectral smoothness (affects Gaussian process part)
    'a': 10,  # relatively high variance of spectral curvature
    'k0': .3,  # quefrency mode below which cepstrum flattens
    # Power-law part of spectrum:
    'sm': -3,  # preferred power-law slope
    'sv': .6,  # low variance of power-law slope
    'im':  -4,  # y-intercept mean, in-/decrease for more/less contrast
    'iv': 2.   # y-intercept variance
}
A = ift.SLAmplitude(**dct)
correlated_field = ift.CorrelatedField(position_space, A)

### SETTING UP SPECIFIC SCENARIO ####

signal = correlated_field.exp()

R = exposure_response(position_space)
signal_response = R(signal)

data_space = R.target
data, ground_truth = generate_poisson_data(signal_response)

likelihood = ift.PoissonianEnergy(data)(signal_response)

### PLOT PRIOR SAMPLES ###
plot_prior_samples_2d(5, signal, R, correlated_field, A, 'poisson')


######## SOLVING PROBLEM ########
# Minimization parameters
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
for i in range(5):
    # Draw new samples and minimize KL
    KL = ift.MetricGaussianKL(mean, H, N_samples)
    KL, convergence = minimizer(KL)
    mean = KL.position

### PLOT RESULTS ###
N_posterior_samples = 30
KL = ift.MetricGaussianKL(mean, H, N_posterior_samples)
plot_reconstruction_2d(data, ground_truth, KL, signal, R, A)



