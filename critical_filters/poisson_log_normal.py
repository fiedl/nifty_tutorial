import sys
sys.path.append('../')
import numpy as np
import nifty5 as ift
from responses import exposure_response
from generate_data import generate_Poisson_data


np.random.seed(42)

position_space = ift.RGSpace([256,256])

harmonic_space = position_space.get_default_codomain()

HT = ift.HarmonicTransformOperator(harmonic_space, target=position_space)
R = exposure_response(position_space)
data_space = R.target

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
# interactive plotting
# plotting correlated_field(ift.from_random('normal',correlated_field.target))

R = exposure_response(position_space)
signal = correlated_field.exp()
signal_response = R(signal)


data_space = R.target

data = generate_Poisson_data(signal_response)

# Minimization parameters
ic_sampling = ift.GradientNormController(iteration_limit=60)
ic_newton = ift.GradInfNormController(
    name='Newton', tol=1e-6, iteration_limit=20)
minimizer = ift.NewtonCG(ic_newton)

# Set up likelihood and information Hamiltonian

likelihood = ift.PoissonianEnergy(data)(signal_response)
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

# Draw posterior samples
N_posterior_samples = 10
KL = ift.MetricGaussianKL(mean, H, N_posterior_samples)
sc = ift.StatCalculator()
sky_samples = []
for sample in KL.samples:
    tmp = correlated_field(sample + KL.position)
    sc.add(tmp)
    sky_samples += [tmp]

# Plotting
plot = ift.Plot()
plot.add([sc.mean, R.adjoint(data)]+sky_samples, alpha = [1., 1.] + [0.4]*N_posterior_samples, ymin = -1.4459574, ymax = 1.956656)
plot.output(xsize=10, ysize=6 , name='results_signal.pdf')

from generate_data import mystery_spec
actual_pow = ift.PS_field(A.target[0], mystery_spec)
powers = [A.force(s + KL.position)**2 for s in KL.samples]
plot = ift.Plot()
plot.add(
    [A.force(KL.position)**2,
                actual_pow] + powers,
    title="Sampled Posterior Power Spectrum",
    alpha = [1,1] + [0.3]*N_posterior_samples)
plot.output(name='results_power.pdf')
