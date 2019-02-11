import numpy as np
import nifty5 as ift

np.random.seed(42)

def prior_spec(k):
    return 1/(10.+k**4)

m = 7
b = 3
a = 5
def mystery_spec(k):
    return a/((m**2-k**2)**2 + b**2*k**2)

space = ift.RGSpace(256)
harmonic_space = space.get_default_codomain()

HT = ift.HartleyOperator(harmonic_space, target=space)
N = ift.ScalingOperator(0.01, space)
S_k = ift.create_power_operator(harmonic_space, mystery_spec)
s = HT(S_k.draw_sample())
d = s + N.draw_sample()
np.save('data_2.npy', d.to_global_data())

S_k = ift.create_power_operator(harmonic_space, prior_spec)
s = HT(S_k.draw_sample())
d = s + N.draw_sample()
np.save('data_1.npy', d.to_global_data())
