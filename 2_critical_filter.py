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
