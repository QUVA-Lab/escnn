import unittest
from unittest import TestCase

import numpy as np

from escnn.group import *
from escnn.kernels import *


class TestSolutionsEquivariance(TestCase):
    
    def test_circular(self):

        radial = GaussianRadialProfile(
            radii=[0., 1., 2., 5, 10],
            sigma=[0.6, 1., 1.3, 2.5, 3.],
        )
        basis = CircularShellsBasis(
            L = 5, radial=radial,
        )
        basis.check_equivariance()

    def test_spherical(self):

        radial = GaussianRadialProfile(

            radii=[0., 1., 2., 5, 10],
            sigma=[0.6, 1., 1.3, 2.5, 3.],
        )
        basis = SphericalShellsBasis(
            L=5, radial=radial,
        )
        basis.check_equivariance()

    def test_circular_filter(self):

        radial = GaussianRadialProfile(
            radii=[0., 1., 2., 5, 10],
            sigma=[0.6, 1., 1.3, 2.5, 3.],
        )
        basis = CircularShellsBasis(
            L = 5, radial=radial,
            filter=lambda attr: attr['irrep:frequency'] < 2*attr['radius']
        )
        basis.check_equivariance()

    def test_spherical_filter(self):

        radial = GaussianRadialProfile(
            radii=[0., 1., 2., 5, 10],
            sigma=[0.6, 1., 1.3, 2.5, 3.],
        )
        basis = SphericalShellsBasis(
            L=5, radial=radial,
            filter=lambda attr: attr['irrep:frequency'] < 2 * attr['radius']
        )
        basis.check_equivariance()

    def test_ico_filter(self):
        group = ico_group()

        for sgid in [
            (False, 2),
            (False, 3),
            (False, 5),
        ]:
            r = 1.
            change_of_basis = np.eye(3) * r
            attributes = {'radius': r}
            basis = SparseOrbitBasisWithIcosahedralSymmetry(
                X=group.homspace(sgid),
                sigma=0.6, attributes=attributes,
                change_of_basis=change_of_basis
            )

            basis.check_equivariance()


if __name__ == '__main__':
    unittest.main()
