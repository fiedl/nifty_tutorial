import numpy as np

import nifty5 as ift


np.random.seed(42)

space = ift.RGSpace(256)
harmonic_space = space.get_default_codomain()

HT = ift.HartleyOperator(harmonic_space, target=space)
R = ift.GeometryRemover(space)
data_space = R.target
N = ift.ScalingOperator(0.1, data_space)

data = np.load('data_2.npy')
data = ift.from_global_data(data_space, data) 
power_space = ift.PowerSpace(harmonic_space)

# Set up an amplitude operator for the field
dct = {
    'target': power_space,
    'n_pix': 64,  # 64 spectral bins
    # Spectral smoothness (affects Gaussian process part)
    'a': 10,  # relatively high variance of spectral curbvature
    'k0': .2,  # quefrency mode below which cepstrum flattens
    # Power-law part of spectrum:
    'sm': -4,  # preferred power-law slope
    'sv': .6,  # low variance of power-law slope
    'im':  -6,  # y-intercept mean, in-/decrease for more/less contrast
    'iv': 2.   # y-intercept variance
}
A = ift.SLAmplitude(**dct)

correlated_field = ift.CorrelatedField(space, A)

signal_response = R(correlated_field)

# Minimization parameters
ic_sampling = ift.GradientNormController(iteration_limit=60)
ic_newton = ift.GradInfNormController(
    name='Newton', tol=1e-6, iteration_limit=20)
minimizer = ift.NewtonCG(ic_newton)

# Set up likelihood and information Hamiltonian
likelihood = ift.GaussianEnergy(mean=data, covariance=N)(signal_response)
H = ift.StandardHamiltonian(likelihood, ic_sampling)

initial_mean = ift.MultiField.full(H.domain, 0.)
mean = initial_mean

# number of samples used to estimate the KL
N_samples = 3

# Draw new samples to approximate the KL five times
for i in range(5):
    # Draw new samples and minimize KL
    KL = ift.MetricGaussianKL(mean, H, N_samples, mirror_samples=True)
    KL, convergence = minimizer(KL)
    mean = KL.position

# Draw posterior samples
N_posterior_samples = 10
KL = ift.MetricGaussianKL(mean, H, N_posterior_samples)
sc = ift.StatCalculator()
sc_pow = ift.StatCalculator()
sky_samples = []
powers = []
for sample in KL.samples:
    tmp = correlated_field(sample + KL.position)
    sc.add(tmp)
    power_samp = A.force(sample + KL.position)**2
    sc_pow.add(power_samp)
    sky_samples += [tmp]
    powers += [power_samp]

from plotting_aachen import plot,power_plot
mock = np.load('signal_2.npy')
mock = ift.from_global_data(space,mock)
plot('results_signal',sc.mean,data,mock,sky_samples)



from generate_data import mystery_spec
actual_pow = ift.PS_field(A.target[0], mystery_spec)
power_plot('results_power',actual_pow,sc_pow.mean,powers)
