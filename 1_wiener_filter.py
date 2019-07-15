import numpy as np

import nifty5 as ift
from just_plot import plot_WF

np.random.seed(42)

# want to implement: m = Dj = (S^{-1} + R^T N^{-1} R)^{-1} R^T N^{-1} d

position_space = ift.RGSpace(256)

R = ift.GeometryRemover(position_space)
data_space = R.target
data = np.load('../data_1.npy')
data = ift.from_global_data(data_space, data)

ground_truth = np.load('../signal_1.npy')
ground_truth = ift.from_global_data(position_space, ground_truth)
plot_WF('data', ground_truth, data)

N = ift.ScalingOperator(0.1, data_space)

harmonic_space = position_space.get_default_codomain()
HT = ift.HartleyOperator(harmonic_space, target=position_space)


def prior_spectrum(k):
    return 1/(10. + k**2.5)


S_h = ift.create_power_operator(harmonic_space, prior_spectrum)
S = HT @ S_h @ HT.adjoint

D_inv = S.inverse + R.adjoint @ N.inverse @ R
j = (R.adjoint @ N.inverse)(data)

IC = ift.GradientNormController(iteration_limit=100, tol_abs_gradnorm=1e-7)
D = ift.InversionEnabler(D_inv.inverse, IC, approximation=S)

m = D(j)

plot_WF('result', ground_truth, data, m)

S = ift.SandwichOperator.make(HT.adjoint, S_h)
D = ift.WienerFilterCurvature(R, N, S, IC, IC).inverse
N_samples = 10
samples = [D.draw_sample() + m for i in range(N_samples)]

plot_WF('result', ground_truth, data, m=m, samples=samples)
