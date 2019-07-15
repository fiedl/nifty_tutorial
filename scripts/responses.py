import numpy as np

import nifty5 as ift


def checkerboard_response(position_space):
    '''Checkerboard mask for 2D mode'''
    mask = np.ones(position_space.shape)
    x, y = position_space.shape
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                mask[i*x//8:(i + 1)*x//8, j*y//8:(j + 1)*y//8] = 0
    mask = ift.from_global_data(position_space, mask)
    return ift.MaskOperator(mask)


def exposure_response(position_space):
    '''Structured exposure for 2D mode'''
    x_shape, y_shape = position_space.shape

    exposure = np.ones(position_space.shape)
    exposure[x_shape//3:x_shape//2, :] *= 2.
    exposure[x_shape*4//5:x_shape, :] *= .1
    exposure[x_shape//2:x_shape*3//2, :] *= 3.
    exposure[:, x_shape//3:x_shape//2] *= 2.
    exposure[:, x_shape*4//5:x_shape] *= .1
    exposure[:, x_shape//2:x_shape*3//2] *= 3.

    exposure = ift.Field.from_global_data(position_space, exposure)
    E = ift.makeOp(exposure)
    G = ift.GeometryRemover(E.target)
    return G @ E


def psf_response(position_space):
    C = ift.HarmonicSmoothingOperator(position_space, 0.01)
    G = ift.GeometryRemover(C.target)
    return G @ C


def radial_tomography_response(position_space, lines_of_sight=100):
    starts = list(np.random.uniform(0, 1, (lines_of_sight, 2)).T)
    ends = list(0.5 + 0*np.random.uniform(0, 1, (lines_of_sight, 2)).T)
    return ift.LOSResponse(position_space, starts=starts, ends=ends)


def random_tomography_response(position_space, lines_of_sight=100):
    starts = list(np.random.uniform(0, 1, (lines_of_sight, 2)).T)
    ends = list(np.random.uniform(0, 1, (lines_of_sight, 2)).T)
    return ift.LOSResponse(position_space, starts=starts, ends=ends)
