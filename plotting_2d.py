import nifty5 as ift
import numpy as np
import matplotlib.pylab as pl


def plot_prior_samples_2d(n_samps, signal, response, correlated_field, amplitude, likelihood, noise_covariance=None):
    fig, ax = pl.subplots(nrows=n_samps, ncols=5, figsize=(2*5, 2*n_samps, ))
    for s in range(n_samps):
        sample = ift.from_random('normal', signal.domain)
        cf = correlated_field(sample)
        signal_response = response(signal)
        sg = signal(sample)
        sr = response.adjoint(response(signal(sample)))
        pow = amplitude.force(sample)
        if likelihood == 'gauss':
            synthetic_position = ift.from_random('normal', signal_response.domain)
            data = signal_response(synthetic_position) + noise_covariance.draw_sample()
        elif likelihood == 'poisson':
            synthetic_position = ift.from_random('normal', signal_response.domain)
            rate = signal_response(synthetic_position).val
            data = ift.from_global_data(signal_response.target, np.random.poisson(rate))
        elif likelihood == 'bernoulli':
            synthetic_position = ift.from_random('normal', signal_response.domain)
            rate = signal_response(synthetic_position).val
            data = ift.from_global_data(signal_response.target, np.random.binomial(1, rate))
        else:
            raise ValueError('likelihood type not implemented')
        data = response.adjoint(data)

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
            ax[0, 0].set_title('Power spectrum')
            ax[0, 1].set_title('Correlated field')
            ax[0, 2].set_title('Signal')
            ax[0, 3].set_title('Signal Response')
            ax[0, 4].set_title('Synthetic data')

        ax[n_samps-1, 0].get_xaxis().set_visible(True)

    pl.tight_layout()
    pl.savefig('prior_samples_' + likelihood + '.png')
    return


def plot_reconstruction_2d(data, truth, KL, signal, response, amplitude):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    sc = ift.StatCalculator()
    sky_samples = []
    amp_samples = []
    for sample in KL.samples:
        tmp = signal(sample + KL.position)
        pow = amplitude.force(sample)
        sc.add(tmp)
        sky_samples += [tmp]
        amp_samples += [pow]

    pl.ion()
    fig, ax = pl.subplots(nrows=2, ncols=3, figsize=(4*3, 4*2,))
    im = list()
    im.append(ax[0, 0].imshow(truth.val, aspect='auto'))
    ax[0, 0].set_title('True Signal')

    im.append(ax[0, 1].imshow(response.adjoint(response(sc.mean)).val, aspect='auto'))
    ax[0, 1].set_title('Signal Response')

    im.append(ax[0, 2].imshow(response.adjoint(data).val, aspect='auto'))
    ax[0, 2].set_title('Data')

    im.append(ax[1, 0].imshow(sc.mean.val, aspect='auto'))
    ax[1, 0].set_title('Posterior mean')

    im.append(ax[1, 1].imshow(sc.var.val, aspect='auto'))
    ax[1, 1].set_title('Uncertainty')

    for s in amp_samples:
        ax[1, 2].plot(s.domain[0].k_lengths, s.val, color='lightgrey')

    amp_mean = sum(amp_samples)/(len(amp_samples))
    ax[1, 2].plot(amp_mean.domain[0].k_lengths, amp_mean.val, color='black')
    ax[1, 2].set_yscale('log')
    ax[1, 2].set_xscale('log')
    ax[1, 2].set_title('Power Spectra')

    from itertools import product
    c = 0
    for i, j in product(range(2), range(3)):
        if not (i == 1 and j == 2):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
            divider = make_axes_locatable(ax[i, j])
            ax_cb = divider.new_horizontal(size="5%", pad=0.05)
            fig1 = ax[i, j].get_figure()
            fig1.add_axes(ax_cb)
            ax[i, j].figure.colorbar(im[c], ax=ax[i, j], cax=ax_cb, orientation='vertical')
        c += 1

    pl.tight_layout()
    pl.savefig('reconstruction.png')
    return
