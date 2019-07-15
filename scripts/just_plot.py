from itertools import product

import numpy as np
import pylab as plt

import nifty5 as ift


def plot_WF(name, mock, d, m=None, samples=None):
    plt.figure(figsize=(15, 8))
    dist = mock.domain[0].distances[0]
    npoints = mock.domain[0].shape[0]
    xcoord = np.arange(npoints, dtype=np.float64)*dist
    plt.plot(xcoord, d.to_global_data(), 'kx', label="data")
    plt.plot(xcoord, mock.to_global_data(), 'b-', label="ground truth")
    if m is not None:
        plt.plot(xcoord, m.to_global_data(), 'k-', label="reconstruction")
    plt.title('reconstructed signal')
    plt.ylabel('value')
    plt.xlabel('position')
    if samples is not None:
        std = 0.
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
            color="k",
            label=r"standard deviation")
    plt.legend()
    x1, x2, y1, y2 = plt.axis()
    ymin = np.min(d.to_global_data()) - 0.1
    ymax = np.max(d.to_global_data()) + 0.1
    xmin = np.min(xcoord)
    xmax = np.max(xcoord)
    plt.axis((xmin, xmax, ymin, ymax))
    plt.savefig(name + ".png", dpi=300)
    plt.close('all')


def plot_CF(KL, correlated_field, A, data):
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
    ground_truth = np.load('../signal_2.npy')
    ground_truth = ift.from_global_data(correlated_field.target, ground_truth)
    from generate_data import mystery_spec
    actual_pow = ift.PS_field(A.target[0], mystery_spec)

    power_plot('critical_power', actual_pow, sc_pow.mean, powers)
    plot_WF('critical_filter', ground_truth, data, sc.mean, sky_samples)
    plt.close('all')


def power_plot(name, s, m, samples=None):
    plt.figure(figsize=(15, 8))
    ks = s.domain[0].k_lengths
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(ks, s.to_global_data(), 'b-', label="ground truth")
    plt.plot(ks, m.to_global_data(), 'k-', label="reconstruction")
    plt.title('reconstructed power-spectrum')
    plt.ylabel('power')
    plt.xlabel('harmonic mode')
    if samples is not None:
        for i in range(len(samples)):
            if i == 0:
                lgd = "samples"
            else:
                lgd = None
            plt.plot(
                ks, samples[i].to_global_data(), 'k-', alpha=0.3, label=lgd)
    plt.legend()
    plt.savefig(name + ".png", dpi=300)
    plt.close('all')


def plot_prior_samples_2d(n_samps,
                          signal,
                          R,
                          correlated_field,
                          A,
                          likelihood,
                          N=None):
    fig, ax = plt.subplots(
        nrows=n_samps, ncols=5, figsize=(
            2*5,
            2*n_samps,
        ))
    for s in range(n_samps):
        sample = ift.from_random('normal', signal.domain)
        cf = correlated_field(sample)
        signal_response = R(signal)
        sg = signal(sample)
        sr = R.adjoint(R(signal(sample)))
        pow = A.force(sample)
        if likelihood == 'gauss':
            data = signal_response(sample) + N.draw_sample()
        elif likelihood == 'poisson':
            rate = signal_response(sample).val
            data = ift.from_global_data(signal_response.target,
                                        np.random.poisson(rate))
        elif likelihood == 'bernoulli':
            rate = signal_response(sample).val
            data = ift.from_global_data(signal_response.target,
                                        np.random.binomial(1, rate))
        else:
            raise ValueError('likelihood type not implemented')
        data = R.adjoint(data + 0.)

        ax[s, 0].plot(pow.domain[0].k_lengths, pow.val)
        ax[s, 0].set_yscale('log')
        ax[s, 0].set_xscale('log')
        ax[s, 0].get_xaxis().set_visible(False)

        ax[s, 1].imshow(cf.val, aspect='auto')
        ax[s, 1].get_xaxis().set_visible(False)
        ax[s, 1].get_yaxis().set_visible(False)

        ax[s, 2].imshow(sg.val, aspect='auto')
        ax[s, 2].get_xaxis().set_visible(False)
        ax[s, 2].get_yaxis().set_visible(False)

        ax[s, 3].imshow(sr.val, aspect='auto')
        ax[s, 3].get_xaxis().set_visible(False)
        ax[s, 3].get_yaxis().set_visible(False)

        ax[s, 4].imshow(data.val, cmap='viridis', aspect='auto')
        ax[s, 4].get_xaxis().set_visible(False)
        ax[s, 4].yaxis.tick_right()
        ax[s, 4].get_yaxis().set_visible(False)

        if s == 0:
            ax[0, 0].set_title('power-spectrum')
            ax[0, 1].set_title('correlated field')
            ax[0, 2].set_title('signal')
            ax[0, 3].set_title('signal Response')
            ax[0, 4].set_title('synthetic data')

        ax[n_samps - 1, 0].get_xaxis().set_visible(True)

    plt.tight_layout()
    plt.savefig('prior_samples_' + likelihood + '.png')
    plt.close('all')


def plot_reconstruction_2d(data, ground_truth, KL, signal, R, A):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    sc = ift.StatCalculator()
    sky_samples = []
    amp_samples = []
    for sample in KL.samples:
        tmp = signal(sample + KL.position)
        pow = A.force(sample)
        sc.add(tmp)
        sky_samples += [tmp]
        amp_samples += [pow]

    fig, ax = plt.subplots(
        nrows=2, ncols=3, figsize=(
            4*3,
            4*2,
        ))
    im = list()
    im.append(ax[0, 0].imshow(signal(ground_truth).val, aspect='auto'))
    ax[0, 0].set_title('true signal')

    im.append(ax[0, 1].imshow(R.adjoint(R(sc.mean)).val, aspect='auto'))
    ax[0, 1].set_title('signal response')

    im.append(ax[0, 2].imshow(R.adjoint(data).val, aspect='auto'))
    ax[0, 2].set_title('data')

    im.append(ax[1, 0].imshow(sc.mean.val, aspect='auto'))
    ax[1, 0].set_title('posterior mean')

    im.append(ax[1, 1].imshow(ift.sqrt(sc.var).val, aspect='auto'))
    ax[1, 1].set_title('standard deviation')

    for s in amp_samples:
        ax[1, 2].plot(s.domain[0].k_lengths, s.val, color='lightgrey')

    amp_mean = sum(amp_samples)/(len(amp_samples))
    ax[1, 2].plot(
        amp_mean.domain[0].k_lengths,
        amp_mean.val,
        color='black',
        label='reconstruction')
    ax[1, 2].plot(
        amp_mean.domain[0].k_lengths,
        A.force(ground_truth).val,
        color='b',
        label='ground truth')
    ax[1, 2].legend()
    ax[1, 2].set_yscale('log')
    ax[1, 2].set_xscale('log')
    ax[1, 2].set_title('power-spectra')

    c = 0
    for i, j in product(range(2), range(3)):
        if not (i == 1 and j == 2):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
            divider = make_axes_locatable(ax[i, j])
            ax_cb = divider.new_horizontal(size="5%", pad=0.05)
            fig1 = ax[i, j].get_figure()
            fig1.add_axes(ax_cb)
            ax[i, j].figure.colorbar(
                im[c], ax=ax[i, j], cax=ax_cb, orientation='vertical')
        c += 1

    plt.tight_layout()
    plt.savefig('reconstruction.png')
    plt.close('all')
