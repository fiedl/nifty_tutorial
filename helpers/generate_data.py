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


def generate_gaussian_data(signal_response, noise_covariance):
    ground_truth = ift.from_random('normal', signal_response.domain)
    d = signal_response(ground_truth) + noise_covariance.draw_sample()
    return d, ground_truth


def generate_poisson_data(signal_response):
    ground_truth = ift.from_random('normal', signal_response.domain)
    rate = signal_response(ground_truth).to_global_data()
    d = np.random.poisson(rate)
    return ift.from_global_data(signal_response.target, d), ground_truth


def generate_bernoulli_data(signal_response):
    ground_truth = ift.from_random('normal', signal_response.domain)
    rate = signal_response(ground_truth).to_global_data()
    d = np.random.binomial(1, rate)
    return ift.from_global_data(signal_response.target, d), ground_truth


def generate_wf_data(domain, spectrum):
    harmonic_space = domain.get_default_codomain()
    HT = ift.HartleyOperator(harmonic_space, target=domain)
    N = ift.ScalingOperator(0.1, domain)
    S_k = ift.create_power_operator(harmonic_space, spectrum)
    s = HT(S_k.draw_sample()).to_global_data()
    d = s + N.draw_sample().to_global_data()
    return d, s


def generate_mysterious_data(domain):
    return generate_wf_data(domain, lambda k: 5/((7**2 - k**2)**2 + 3**2*k**2))
