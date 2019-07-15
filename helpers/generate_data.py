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


def generate_gaussian_data(signal_response, noise_covariance):
    ground_truth = ift.from_random('normal', signal_response.domain)
    return signal_response(
        ground_truth) + noise_covariance.draw_sample(), ground_truth


def generate_poisson_data(signal_response):
    ground_truth = ift.from_random('normal', signal_response.domain)
    rate = signal_response(ground_truth).val
    data = np.random.poisson(rate)
    return ift.from_global_data(signal_response.target, data), ground_truth


def generate_bernoulli_data(signal_response):
    ground_truth = ift.from_random('normal', signal_response.domain)
    rate = signal_response(ground_truth).val
    data = np.random.binomial(1, rate)
    return ift.from_global_data(signal_response.target, data), ground_truth


def mystery_spec(k):
    return 5/((7**2 - k**2)**2 + 3**2*k**2)


def prior_spec(k):
    return 1/(10. + k**2.5)


if __name__ == '__main__':
    space = ift.RGSpace(256)
    harmonic_space = space.get_default_codomain()

    HT = ift.HartleyOperator(harmonic_space, target=space)
    N = ift.ScalingOperator(0.1, space)
    S_k = ift.create_power_operator(harmonic_space, mystery_spec)
    s = HT(S_k.draw_sample())
    d = s + N.draw_sample()
    np.save('../data_2.npy', d.to_global_data())
    np.save('../signal_2.npy', s.to_global_data())

    S_k = ift.create_power_operator(harmonic_space, prior_spec)
    s = HT(S_k.draw_sample())
    d = s + N.draw_sample()
    np.save('../data_1.npy', d.to_global_data())
    np.save('../signal_1.npy', s.to_global_data())
