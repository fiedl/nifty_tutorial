import numpy as np

import nifty5 as ift
from scripts.generate_data import *
from scripts.just_plot import *
from scripts.responses import *

np.random.seed(42)

position_space = ift.RGSpace([256, 256])
harmonic_space = position_space.get_default_codomain()
HT = ift.HarmonicTransformOperator(harmonic_space, target=position_space)
power_space = ift.PowerSpace(harmonic_space)
A = ift.SLAmplitude(
    target=power_space, n_pix=64, a=10, k0=.2, sm=-4, sv=.6, im=-2, iv=2)
