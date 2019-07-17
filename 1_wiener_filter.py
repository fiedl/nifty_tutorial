import numpy as np

import nifty5 as ift
from helpers import generate_wf_data, plot_WF

np.random.seed(42)

# Want to implement: m = Dj = (S^{-1} + R^T N^{-1} R)^{-1} R^T N^{-1} d
