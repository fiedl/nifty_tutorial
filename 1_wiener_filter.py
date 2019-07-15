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
from helpers import plot_WF

np.random.seed(42)

# Want to implement: m = Dj = (S^{-1} + R^T N^{-1} R)^{-1} R^T N^{-1} d

position_space = ift.RGSpace(256)

# Generate data and signal
harmonic_space = position_space.get_default_codomain()
HT = ift.HartleyOperator(harmonic_space, target=position_space)
N = ift.ScalingOperator(0.1, position_space)
S_k = ift.create_power_operator(harmonic_space, lambda k: 1/(10. + k**2.5))
s = HT(S_k.draw_sample())
d = s + N.draw_sample()
np.save('data.npy', d.to_global_data())
np.save('signal.npy', s.to_global_data())
# End generate data and signal

R = ift.GeometryRemover(position_space)
data_space = R.target
data = np.load('data.npy')
data = ift.from_global_data(data_space, data)

ground_truth = np.load('signal.npy')
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
