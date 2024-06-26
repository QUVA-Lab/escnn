import numpy as np

from scipy.ndimage import rotate
from scipy.ndimage import affine_transform


def rotate_array_2d(x, angle):
    k = 2 * angle / np.pi
    if k.is_integer():
        # Rotations by 180 and 270 degrees seem to be not perfect using `ndimage.rotate` and can therefore
        # make some tests fail.
        # For this reason, we use `np.rot90` to perform rotations by multiples of 90 degrees without interpolation
        return np.rot90(x, k, axes=(-2, -1))
    else:
        return rotate(x, angle * 180.0 / np.pi, (-2, -1), reshape=False, order=2)


def linear_transform_array_3d(x, trafo: np.ndarray, exact=True, order=2):
    assert trafo.shape == (3, 3)
    assert len(x.shape) > 2
    return linear_transform_array_nd(x, trafo, exact, order=order)
    

def linear_transform_array_nd(x, trafo: np.ndarray, exact=True, order=2):
    n = trafo.shape[0]
    assert trafo.shape == (n, n)
    assert len(x.shape) >= n

    # TODO : MAKE THIS EXPLICIT SOMEWHERE IN THE DOCS!!!
    # assume trafo matrix has [X, Y, Z, ....] order
    # but input tensor has [..., -Z, -Y, X] order
    trafo = trafo[::-1, ::-1].copy()
    trafo[:-1, :] *= -1
    trafo[:, :-1] *= -1

    # This seems necessary when the rotation is only involving a subset of the dimensions to avoid weird interpolation
    # artifacts (e.g. if a rotation is only in the XY plane but preserves the Z axis, small numerical errors might
    # still affect the Z axis causing equivariance unittests to fail)
    trafo = trafo.astype(np.float16)

    D = len(x.shape)
    at = np.abs(trafo)
    
    if exact and (
            np.isclose(at.sum(axis=0), 1).all() and
            np.isclose(at.sum(axis=1), 1).all() and
            (np.isclose(at, 1.) | np.isclose(at, 0.)).all()
    ):
        # if it is a permutation matrix we can perform this transformation without interpolation
        axs = np.around(trafo).astype(int) @ np.arange(1, n+1).reshape(n, 1)
        axs = axs.reshape(-1)
        
        stride = np.sign(axs).tolist()
        axs = np.abs(axs).tolist()
        
        axs = list(range(D - n)) + [D - n - 1 + a for a in axs]
        assert len(axs) == D, (len(axs), D)

        y = x.transpose(axs)
        
        stride = (Ellipsis,) + tuple([slice(None, None, s) for s in stride])
        y = y[stride]
        return y
    else:
        
        trafo = trafo.T

        t = np.eye(D)
        t[-n:, -n:] = trafo
        center = np.zeros(len(x.shape))
        center[-n:] = (np.asarray(x.shape[-n:]) - 1) / 2
        center[-n:] = -(trafo - np.eye(n)) @ center[-n:]

        # mode='grid-constant' is important to avoid strange boundary artifacts
        # see here: https://github.com/scipy/scipy/issues/9865#issuecomment-726993353
        # but it seems much more memory expensive
        # return affine_transform(x, t, offset=center, order=order, mode='grid-constant')
        return affine_transform(x, t, offset=center, order=order)


if __name__ == '__main__':

    # test that the exact rotation method produces the same results as the interpolation one
    # on all 48 origin-preserving isometries of the voxel grid
    
    import itertools
    x = np.random.randn(15, 15, 15)
    for p in itertools.permutations([0,1,2]):
        M = np.eye(3)[p, :]
        
        for s in itertools.product([-1, 1], repeat=3):
            rot = np.asarray(s).reshape(-1, 1) * M
            
            y1 = linear_transform_array_3d(x, rot, True)
            y2 = linear_transform_array_3d(x, rot, False, order=2)
            y3 = linear_transform_array_3d(x, rot, False, order=3)
            assert np.allclose(y2, y1), rot
            assert np.allclose(y3, y1), rot
            
    # test that the nd method is equivalent to the 2d one
    x = np.random.randn(3, 2, 11, 11)
    np.set_printoptions(suppress=True, precision=3, linewidth=100000)

    for _ in range(10):
        angle = np.random.rand()*2*np.pi
        
        rot = np.asarray([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])
    
        y1 = rotate_array_2d(x, angle)
        y2 = linear_transform_array_nd(x, rot)
        assert np.allclose(y2, y1), rot
