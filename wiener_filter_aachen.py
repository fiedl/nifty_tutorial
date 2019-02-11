import nifty5 as ift
import numpy as np

np.random.seed(42)

# want to implement: m = Dj = (S^{-1} + R^T N^{-1} R)^{-1} R^T N^{-1} d

space = ift.RGSpace(256)
harmonic_space = space.get_default_codomain()

HT = ift.HartleyOperator(harmonic_space, target=space)
R = ift.GeometryRemover(space)
data_space = R.target
N = ift.ScalingOperator(0.1, data_space)

data = np.load('data_1.npy')
data = ift.from_global_data(data_space, data) 

def prior_spectrum(k):
    return 1/(10.+k**2)

S_h = ift.create_power_operator(harmonic_space, prior_spectrum)
S = HT @ S_h @ HT.adjoint

D_inv = S.inverse + R.adjoint @ N.inverse @ R
j = (R.adjoint @ N.inverse)(data)

IC = ift.GradientNormController(iteration_limit = 100, tol_abs_gradnorm=1e-7)
D = ift.InversionEnabler(D_inv.inverse, IC, approximation=S)

m = D(j)

S = ift.SandwichOperator.make(HT.adjoint, S_h)
D = ift.WienerFilterCurvature(R, N , S, IC, IC).inverse
N_samples = 10
samples = [D.draw_sample()+m for i in range(N_samples)]

plot = ift.Plot()
plot.add([m, R.adjoint(data)] + samples, alpha = [1,1]+[0.3]*N_samples)
plot.output(xsize = 10, ysize = 6, name='result.pdf')

