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

from itertools import product

import numpy as np
import pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import nifty5 as ift


def plot_WF(name, mock, d, m=None, samples=None):
    plt.figure(figsize=(15, 8))
    dist = mock.domain[0].distances[0]
    npoints = mock.domain[0].shape[0]
    xcoord = np.arange(npoints, dtype=np.float64)*dist
    plt.plot(xcoord, d.to_global_data(), 'kx', label='data')
    plt.plot(xcoord, mock.to_global_data(), 'b-', label='ground truth')
    if m is not None:
        plt.plot(xcoord, m.to_global_data(), 'k-', label='reconstruction')
    plt.title('reconstructed signal')
    plt.ylabel('value')
    plt.xlabel('position')
    if samples is not None:
        std = 0
        for s in samples:
            std = std + (s - m)**2
        std = std/len(samples)
        std = std.to_global_data()
        std = np.sqrt(std)
        md = m.to_global_data()
        plt.fill_between(
            xcoord,
            md - std,
            md + std,
            alpha=0.3,
            color='k',
            label='standard deviation')
    plt.legend()
    x1, x2, y1, y2 = plt.axis()
    ymin = np.min(d.to_global_data()) - 0.1
    ymax = np.max(d.to_global_data()) + 0.1
    xmin = np.min(xcoord)
    xmax = np.max(xcoord)
    plt.axis((xmin, xmax, ymin, ymax))
    plt.savefig('{}.png'.format(name), dpi=300)
    plt.close('all')


def power_plot(name, s, m, samples=None):
    plt.figure(figsize=(15, 8))
    ks = s.domain[0].k_lengths
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(ks, s.to_global_data(), 'b-', label='ground truth')
    plt.plot(ks, m.to_global_data(), 'k-', label='reconstruction')
    plt.title('reconstructed power-spectrum')
    plt.ylabel('power')
    plt.xlabel('harmonic mode')
    if samples is not None:
        for i in range(len(samples)):
            if i == 0:
                lgd = 'samples'
            else:
                lgd = None
            plt.plot(
                ks, samples[i].to_global_data(), 'k-', alpha=0.3, label=lgd)
    plt.legend()
    plt.savefig('{}.png'.format(name), dpi=300)
    plt.close('all')


def plot_prior_samples_2d(n_samps,
                          signal,
                          R,
                          correlated_field,
                          A,
                          likelihood,
                          N=None):
    samples, pspecmin, pspecmax = [], np.inf, 0
    pspec = A*A
    for _ in range(n_samps):
        ss = ift.from_random('normal', signal.domain)
        samples.append(ss)
        foo = pspec.force(ss).to_global_data()
        pspecmin = min([min(foo), pspecmin])
        pspecmax = max([max(foo), pspecmin])

    fig, ax = plt.subplots(nrows=n_samps, ncols=5, figsize=(2*5, 2*n_samps))
    for ii, sample in enumerate(samples):
        cf = correlated_field(sample)
        signal_response = R @ signal
        sg = signal(sample)
        sr = (R.adjoint @ R @ signal)(sample)
        if likelihood == 'gauss':
            data = signal_response(sample) + N.draw_sample()
        elif likelihood == 'poisson':
            rate = signal_response(sample).to_global_data()
            data = ift.from_global_data(signal_response.target,
                                        np.random.poisson(rate))
        elif likelihood == 'bernoulli':
            rate = signal_response(sample).to_global_data()
            data = ift.from_global_data(signal_response.target,
                                        np.random.binomial(1, rate))
        else:
            raise ValueError('likelihood type not implemented')
        data = R.adjoint(data + 0.)

        As = pspec.force(sample)
        ax[ii, 0].plot(As.domain[0].k_lengths, As.to_global_data())
        ax[ii, 0].set_ylim(pspecmin, pspecmax)
        ax[ii, 0].set_yscale('log')
        ax[ii, 0].set_xscale('log')
        ax[ii, 0].get_xaxis().set_visible(False)

        ax[ii, 1].imshow(cf.to_global_data(), aspect='auto')
        ax[ii, 1].get_xaxis().set_visible(False)
        ax[ii, 1].get_yaxis().set_visible(False)

        ax[ii, 2].imshow(sg.to_global_data(), aspect='auto')
        ax[ii, 2].get_xaxis().set_visible(False)
        ax[ii, 2].get_yaxis().set_visible(False)

        ax[ii, 3].imshow(sr.to_global_data(), aspect='auto')
        ax[ii, 3].get_xaxis().set_visible(False)
        ax[ii, 3].get_yaxis().set_visible(False)

        ax[ii, 4].imshow(data.to_global_data(), cmap='viridis', aspect='auto')
        ax[ii, 4].get_xaxis().set_visible(False)
        ax[ii, 4].yaxis.tick_right()
        ax[ii, 4].get_yaxis().set_visible(False)

        if ii == 0:
            ax[0, 0].set_title('power spectrum')
            ax[0, 1].set_title('correlated field')
            ax[0, 2].set_title('signal')
            ax[0, 3].set_title('signal Response')
            ax[0, 4].set_title('synthetic data')

        ax[n_samps - 1, 0].get_xaxis().set_visible(True)
    plt.tight_layout()
    plt.savefig('prior_samples_{}.png'.format(likelihood))
    plt.close('all')


def plot_reconstruction_2d(data, ground_truth, KL, signal, R, A, name):
    sc = ift.StatCalculator()
    sky_samples, pspec_samples = [], []
    for sample in KL.samples:
        tmp = signal(sample + KL.position)
        sc.add(tmp)
        sky_samples.append(tmp)
        pspec_samples.append(A.force(sample)**2)

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(4*3, 4*2))
    im = []
    im.append(ax[0, 0].imshow(
        signal(ground_truth).to_global_data(), aspect='auto'))
    ax[0, 0].set_title('true signal')

    im.append(ax[0, 1].imshow(
        R.adjoint(R(sc.mean)).to_global_data(), aspect='auto'))
    ax[0, 1].set_title('signal response')

    im.append(ax[0, 2].imshow(R.adjoint(data).to_global_data(), aspect='auto'))
    ax[0, 2].set_title('data')

    im.append(ax[1, 0].imshow(sc.mean.to_global_data(), aspect='auto'))
    ax[1, 0].set_title('posterior mean')

    im.append(ax[1, 1].imshow(
        ift.sqrt(sc.var).to_global_data(), aspect='auto'))
    ax[1, 1].set_title('standard deviation')

    for ss in pspec_samples:
        ax[1, 2].plot(
            ss.domain[0].k_lengths, ss.to_global_data(), color='lightgrey')

    amp_mean = sum(pspec_samples)/len(pspec_samples)
    ax[1, 2].plot(
        amp_mean.domain[0].k_lengths,
        amp_mean.to_global_data(),
        color='black',
        label='reconstruction')
    ax[1, 2].plot(
        amp_mean.domain[0].k_lengths,
        A.force(ground_truth).to_global_data()**2,
        color='b',
        label='ground truth')
    ax[1, 2].legend()
    ax[1, 2].set_yscale('log')
    ax[1, 2].set_xscale('log')
    ax[1, 2].set_title('power spectra')

    for c, (i, j) in enumerate(product(range(2), range(3))):
        if i != 1 or j != 2:
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
            divider = make_axes_locatable(ax[i, j])
            ax_cb = divider.new_horizontal(size='5%', pad=0.05)
            fig1 = ax[i, j].get_figure()
            fig1.add_axes(ax_cb)
            ax[i, j].figure.colorbar(
                im[c], ax=ax[i, j], cax=ax_cb, orientation='vertical')
    plt.tight_layout()
    plt.savefig('reconstruction{}.png'.format(name))
    plt.close('all')
