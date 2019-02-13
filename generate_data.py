import numpy as np
import nifty5 as ift

np.random.seed(42)


def generate_gaussian_data(signal_response, noise_covariance):
    ground_truth = ift.from_random('normal', signal_response.domain)
    return signal_response(ground_truth) + noise_covariance.draw_sample(), ground_truth


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

if __name__ == '__main__':
    def prior_spec(k):
        return 1/(10.+k**2)

    m = 7
    b = 3
    a = 5
    def mystery_spec(k):
        return a/((m**2-k**2)**2 + b**2*k**2)

    space = ift.RGSpace(256)
    harmonic_space = space.get_default_codomain()

    HT = ift.HartleyOperator(harmonic_space, target=space)
    N = ift.ScalingOperator(0.1, space)
    S_k = ift.create_power_operator(harmonic_space, mystery_spec)
    s = HT(S_k.draw_sample())
    d = s + N.draw_sample()
    np.save('data_2.npy', d.to_global_data())

    S_k = ift.create_power_operator(harmonic_space, prior_spec)
    s = HT(S_k.draw_sample())
    d = s + N.draw_sample()
    np.save('data_1.npy', d.to_global_data())

