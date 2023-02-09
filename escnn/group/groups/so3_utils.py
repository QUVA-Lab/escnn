from lie_learn.representations.SO3.wigner_d import wigner_D_matrix

from scipy.spatial.transform import Rotation

import numpy as np

from ._numerical_thomson import thomson_sphere, thomson_so3, thomson_cube_sphere, thomson_cube_so3

import warnings


__all__ = [
    'PARAMETRIZATION',
    'PARAMETRIZATIONS',
    'IDENTITY',
    '_change_param',
    '_check_param',
    '_invert',
    '_combine',
    '_equal',
    '_hash',
    '_repr',
    '_wigner_d_matrix',
    '_character',
    '_change_of_basis_real2complex',
    '_grid',
    '_sphere_grid',
    'find_rotation_pole2point',
]

PARAMETRIZATION = "Q"

PARAMETRIZATIONS = [
    'Q',
    'MAT',
    'EV',
    
    # estrinsic rotations
    'xyz',
    'xzy',
    'yxz',
    'yzx',
    'zxy',
    'zyx',
    
    'xyx',
    'xzx',
    'yxy',
    'yzy',
    'zxz',
    'zyz',
    
    # intrinsic rotations
    'XYZ',
    'XZY',
    'YXZ',
    'YZX',
    'ZXY',
    'ZYX',
    
    'XYX',
    'XZX',
    'YXY',
    'YZY',
    'ZXZ',
    'ZYZ',
]


def _to_scipy_rot(element, param: str) -> Rotation:
    assert param in PARAMETRIZATIONS
    if param == 'Q':
        assert isinstance(element, np.ndarray)
        assert element.shape == (4,)
        assert np.isclose(np.linalg.norm(element), 1.)

        return Rotation.from_quat(element)
    elif param == 'MAT':
        assert isinstance(element, np.ndarray)
        assert element.shape == (3, 3)
        assert np.isclose(np.linalg.det(element), 1.)

        return Rotation.from_matrix(element)
    elif param == 'EV':
        return Rotation.from_rotvec(element)
    else:
        return Rotation.from_euler(param, element)


def _from_scipy_rot(element: Rotation, param: str) -> np.ndarray:
    assert param in PARAMETRIZATIONS
    if param == 'Q':
        return element.as_quat()
    elif param == 'MAT':
        return element.as_matrix()
    elif param == 'EV':
        return element.as_rotvec()
    else:
        with warnings.catch_warnings():
            # to prevent printing warnings about Gimbal-Lock
            warnings.simplefilter("ignore")

            return element.as_euler(param)


def _change_param(element, p_from: str, p_to: str):
    
    if p_from == p_to:
        return element
    
    rot = _to_scipy_rot(element, p_from)
    
    return _from_scipy_rot(rot, p_to)


def _check_param(element, param):
    try:
        _to_scipy_rot(element, param)
    except:
        return False
    
    return True


def _invert(element, param=PARAMETRIZATION):
    
    return _from_scipy_rot(
        _to_scipy_rot(element, param).inv(),
        param
    )


def _combine(e1, e2,
             param=PARAMETRIZATION,
             param1=None,
             param2=None
             ):
    r"""
        Compute `e1 * e2`.
        This corresponds to a first rotation by `e2` followed by a rotation by `e1`
    
    """
    
    if param1 is None:
        param1 = param
    if param2 is None:
        param2 = param

    return _from_scipy_rot(
        _to_scipy_rot(e1, param1) * _to_scipy_rot(e2, param2),
        param
    )


def _equal(e1, e2,
           param: str = PARAMETRIZATION,
           param1: str = None,
           param2: str = None,
           atol: float = 1e-7,
           rtol: float = 1e-5
           ) -> bool:
    
    if param1 is None:
        param1 = param
    if param2 is None:
        param2 = param
    
    if not isinstance(e1, np.ndarray):
        e1 = np.asarray(e1).reshape(1, -1)
    if not isinstance(e2, np.ndarray):
        e2 = np.asarray(e2).reshape(1, -1)
    
    # convert the rotations to quaternion
    q1 = _to_scipy_rot(e1, param1).as_quat()
    q2 = _to_scipy_rot(e2, param2).as_quat()
    
    # quaternions are a double cover of SO(3)
    # as both q and -q correspond to the same element of SO(3)
    # therefore, we need to check both cases
    return np.allclose(q1, q2, rtol=rtol, atol=atol) or np.allclose(-q1, q2, rtol=rtol, atol=atol)


def _hash(element, param: str = PARAMETRIZATION):
    element = np.around(_change_param(element, param, 'MAT').reshape(-1), 4)
    return hash(tuple(element))


def _repr(element, param: str = PARAMETRIZATION) -> str:
    element = _change_param(element, param, 'MAT')
    return '\n'.join(element.__repr__()[7:-2].split('\n       '))
    # element = _change_param(element, param, 'Q')
    # return '\n'.join(element.__repr__()[6:-1].split('\n       '))


def _wigner_d_matrix(element, l, param=PARAMETRIZATION, field: str = 'real'):
    wigner_d_param = 'ZYZ'
    
    if param != wigner_d_param:
        element = _change_param(element, p_from=param, p_to=wigner_d_param)
    
    return wigner_D_matrix(l, element[0, ...], element[1, ...], element[2, ...], field=field)


def _character(element, l, param=PARAMETRIZATION):
    if l == 0:
        return 1.
    else:
        element = _change_param(element, p_from=param, p_to='Q')
        theta = 2 * np.arctan2(
            np.linalg.norm(element[..., :-1]),
            element[..., -1]
        )
        sin = np.sin(.5 * theta)
        if not np.isclose(sin, 0.):
            c = np.sin((l + .5) * theta) / np.sin(0.5 * theta)
        else:
            # c[np.sin(.5*theta) == 0.] = (2*l + 1)
            c = (2 * l + 1)
        
        return c


def _change_of_basis_real2complex(d: int):
    # implements the change of basis matrix from the Wigner D Matrices parameterized as
    # ('real', 'quantum', 'centered', 'cs') to ('complex', 'quantum', 'centered', 'cs')
    # in a way that is compatible with lie_learn.representations.SO3.irrep_bases.change_of_basis_matrix

    assert d >= 0, d
    D = 2*d+1

    cob = np.zeros((D, D), dtype=complex)
    for f in range(1, d+1):
        cob[d-f, d-f] = -1j
        cob[d+f, d-f] = 1j if (f%2==0) else -1j

        cob[d-f, d+f] = 1.
        cob[d+f, d+f] = 1 if (f%2==0) else -1.

    cob *= np.sqrt(2) / 2.

    cob[d, d] = 1.

    return cob


IDENTITY = _change_param(np.array([0., 0., 0., 1.]), p_from='Q', p_to=PARAMETRIZATION)


def find_rotation_pole2point(point: np.ndarray):

    assert point.shape == (3,)

    point = point / np.linalg.norm(point)

    rot = np.zeros((3, 3))

    rot[:, 2] = point

    y = np.array([0., 0., 1.])

    if np.isclose(np.abs(np.dot(y, point)), 1.):
        y = np.array([1., 0., 0.])

    rot[:, 1] = y - np.dot(y, point) * point
    rot[:, 1] /= np.linalg.norm(rot[:, 1])
    rot[:, 0] = np.cross(rot[:, 2], rot[:, 1])
    det = np.linalg.det(rot)
    rot[:, 0] *= det

    assert np.isclose(np.linalg.det(rot), 1.), rot

    return rot


#############################################
# GRIDS
#############################################

# Sphere grids

def _random_sphere_samples(n: int, rnd=None):
    assert n > 0
    rng = np.random.RandomState(rnd) if rnd is not None else np.random
    points = rng.randn(n, 3)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points


def _spherical_healpix_grid(N_side: int) -> np.ndarray:
    north_points = []

    # north polar cap
    for i in range(1, N_side):
        for j in range(1, 4 * i + 1):
            cos_theta = 1. - i ** 2 / (3 * N_side ** 2)
            phi = np.pi / (2 * i) * (j - 0.5)

            sin_theta = np.sqrt(1. - cos_theta ** 2)

            north_points.append(
                np.array([np.cos(phi) * sin_theta, np.sin(phi) * sin_theta, cos_theta])
            )

    # north equatorial belt
    for i in range(N_side, 2 * N_side + 1):
        for j in range(1, 4 * N_side + 1):
            cos_theta = 4. / 3. - 2 * i / (3 * N_side)
            s = (i - N_side + 1) % 2
            phi = np.pi / (2 * N_side) * (j - s / 2.)

            sin_theta = np.sqrt(1. - cos_theta ** 2)

            north_points.append(
                np.array([np.cos(phi) * sin_theta, np.sin(phi) * sin_theta, cos_theta])
            )

    points = []
    # add points on the south pole (symmetric to north pole wrt equator, i.e. z=0)
    for p in north_points:
        if p[2] > 0.:
            sp = p.copy()
            sp[2] *= -1
            points.append(sp)
    points += north_points

    N_pix = 12 * N_side ** 2
    assert len(points) == N_pix, (len(points), N_side, N_pix)

    points = np.stack(points, axis=0)

    return points


def _plato_sphere_samples(solid: str):
    solid = solid.split('_')
    if len(solid) == 1:
        solid = solid[0]
        points = None
    elif len(solid) == 2:
        points = solid[1]
        solid = solid[0]
    else:
        raise ValueError

    assert solid in ['ico', 'cube', 'tetra']
    assert points in [None, 'vertices', 'faces', 'edges']

    so3_grid = _grid(solid, parametrization='MAT')

    if points is None:
        origin = np.array([0., 0., 1.]).reshape(-1, 1)
    elif solid == 'cube' and points == 'vertices':
        origin = np.array([1., 1., 1.]).reshape(-1, 1)
    else:
        raise NotImplementedError

    # origin = np.array([1., 1., 1.]).reshape(-1, 1)
    origin /= np.linalg.norm(origin)

    class HashablePoint:

        def __init__(self, point: np.ndarray):
            self.p = point

        def __eq__(self, other):
            return np.allclose(self.p, other.p)

        def __hash__(self):
            element = np.around(self.p.reshape(-1), 4)
            return hash(tuple(element))

    points = set(
        HashablePoint(r @ origin) for r in so3_grid
    )
    points = np.stack([p.p.flatten() for p in points], axis=0)
    return points


def _fibonacci_sphere_samples(N: int):

    idx = np.arange(N, dtype=float)

    phi = np.pi * (3 - np.sqrt(5))

    y = 1. - 2*(idx / (N-1))

    radius = np.sqrt(1 - y**2)

    theta = phi * idx

    x = np.cos(theta) * radius
    z = np.sin(theta) * radius

    points = np.stack([x, y, z], axis=-1)

    return points


def _longlat_sphere_grid(N: int, M: int):

    points = []

    assert M > 1

    for la in range(M):

        if la == 0:
            points.append(np.array([0., 0., 1.]))
            continue

        if la == M-1:
            points.append(np.array([0., 0., -1.]))
            continue

        theta = la * np.pi / (M - 1)
        z = np.cos(theta)

        r = np.sqrt(1 - z ** 2)

        for lo in range(N):
            phi = lo * 2*np.pi / N

            x = np.cos(phi)*r
            y = np.sin(phi)*r

            point = np.array([x, y, z])
            points.append(point)

    points = np.stack(points, axis=0)

    return points


def _sphere_grid(type: str, *args, adj: np.ndarray = None, **kwargs) -> np.ndarray:
    if type == 'rand':
        elements = _random_sphere_samples(*args, **kwargs)
    elif type in ['ico', 'cube', 'tetra', 'cube_vertices']:
        elements = _plato_sphere_samples(type)
    elif type == 'healpix':
        elements = _spherical_healpix_grid(*args, **kwargs)
    elif type == 'fibonacci':
        elements = _fibonacci_sphere_samples(*args, **kwargs)
    elif type == 'longlat':
        elements = _longlat_sphere_grid(*args, **kwargs)
    elif type == 'thomson':
        elements = thomson_sphere(*args, **kwargs)
    elif type == 'thomson_cube':
        elements = thomson_cube_sphere(*args, **kwargs)
    else:
        raise ValueError(f'Grid type "{type}" not recognized')

    assert elements.shape[1] == 3

    if adj is not None:
        assert adj.shape == (3, 3)
        assert np.isclose(np.abs(np.linalg.det(adj)), 1.)
        elements = elements @ adj.T

    return elements

# SO(3) grids

def _so3_hopf_uniform_grid(N) -> Rotation:
    m1 = int(np.round(np.cbrt(N * np.pi)))
    N_side = int(np.round(
        np.sqrt(N / (8 * np.pi ** 2))
    ))
    
    assert m1 > 0
    assert N_side > 0
    
    sphere_points = _spherical_healpix_grid(N_side)
    so3_points = []
    
    for sp in sphere_points:
        for i in range(m1):
            # add +1/4 such that the same rotation is not sampled on the opposite point on the sphere
            
            theta = (i + .25) * 2 * np.pi / m1
            # theta = i * 2 * np.pi / m1
            
            q = np.array([sp[0], sp[1], sp[2], 0.])
            q *= np.sin(theta / 2.)
            q[3] = np.cos(theta / 2.)
            
            so3_points.append(
                Rotation.from_quat(q)
            )
    
    # Filter duplicates
    
    class HashableRot:
        
        def __init__(self, rot: Rotation):
            self.r = rot
        
        def __eq__(self, other):
            return _equal(self.r.as_quat(), other.r.as_quat(), 'Q')
        
        def __hash__(self):
            return _hash(self.r.as_quat(), 'Q')

    so3_points = set(HashableRot(r) for r in so3_points)
    so3_points = Rotation.from_quat(np.stack([r.r.as_quat().reshape(4) for r in so3_points], axis=0))
    # print(len(sphere_points), len(so3_points),  m1, N)
    
    return so3_points


def _so3_hopf_fibonacci_grid(N) -> Rotation:
    N_circle = int(np.round(np.cbrt(N * np.pi)))
    N_sphere = int(np.round(
        np.cbrt(N**2 / np.pi)
    ))

    assert N_circle > 0
    assert N_sphere > 0

    sphere_points = _fibonacci_sphere_samples(N_sphere)
    so3_points = []

    for sp in sphere_points:
        for i in range(N_circle):
            # add +1/4 such that the same rotation is not sampled on the opposite point on the sphere

            # theta = (i + .25) * 2 * np.pi / N_circle
            theta = i * 2 * np.pi / N_circle

            q = np.array([sp[0], sp[1], sp[2], 0.])
            q *= np.sin(theta / 2.)
            q[3] = np.cos(theta / 2.)

            so3_points.append(
                Rotation.from_quat(q)
            )

    # Filter duplicates

    class HashableRot:

        def __init__(self, rot: Rotation):
            self.r = rot

        def __eq__(self, other):
            return _equal(self.r.as_quat(), other.r.as_quat(), 'Q')

        def __hash__(self):
            return _hash(self.r.as_quat(), 'Q')

    so3_points = set(HashableRot(r) for r in so3_points)
    so3_points = Rotation.from_quat(np.stack([r.r.as_quat().reshape(4) for r in so3_points], axis=0))
    # print(len(sphere_points), len(so3_points),  m1, N)

    return so3_points


def _random_samples(N: int, seed=None):
    assert N > 0
    return Rotation.random(N, seed)


def _grid(type: str, *args, adj=IDENTITY, parametrization: str = PARAMETRIZATION, **kwargs) -> np.ndarray:
    if type == 'rand':
        elements = _random_samples(*args, **kwargs)
    elif type == 'tetra':
        elements = Rotation.create_group('T')
    elif type == 'cube':
        elements = Rotation.create_group('O')
    elif type == 'ico':
        elements = Rotation.create_group('I')
    elif type == 'hopf':
        elements = _so3_hopf_uniform_grid(*args, **kwargs)
    elif type == 'fibonacci':
        elements = _so3_hopf_fibonacci_grid(*args, **kwargs)
    elif type == 'thomson':
        elements = Rotation.from_quat(thomson_so3(*args, **kwargs))
    elif type == 'thomson_cube':
        elements = Rotation.from_quat(thomson_cube_so3(*args, **kwargs))
    else:
        raise ValueError(f'Grid type "{type}" not recognized')
    
    adj = _to_scipy_rot(adj, PARAMETRIZATION)
    elements = adj * elements * adj.inv()
    
    return _from_scipy_rot(elements, parametrization)


if __name__ == '__main__':
    # # plot helpix grid
    # grid = _spherical_healpix_grid(4)
    # grid = np.asarray(grid)
    
    N = 500
    # elements = _so3_hopf_uniform_grid(N)
    # elements = set(elements)
    # print(N, len(elements))
    
    # plot hoft grid
    elements = _grid('hopf', N=N, parametrization='Q')
    print(len(elements))
    elements = np.asarray(elements)
    
    grid = elements[:, :3]
    theta_2 = np.arccos(elements[:, 3]) * 2 - np.pi
    theta_2 = theta_2.reshape(-1, 1)
    grid *= theta_2
    
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(grid[:, 0], grid[:, 1], grid[:, 2], )
    
    plt.show()
