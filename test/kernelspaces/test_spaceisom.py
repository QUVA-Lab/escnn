import unittest
from unittest import TestCase

import numpy as np

from escnn.group import *
from escnn.kernels import *


class TestSpaceIsomophism(TestCase):
    
    def test_point(self):
        self.check_isom(PointRn(2, so2_group()))
        self.check_isom(PointRn(2, o2_group()))
        self.check_isom(PointRn(3, so3_group()))
        self.check_isom(PointRn(3, o3_group()))
        
    def test_circle_so2(self):
        self.check_isom(CircleSO2())
        
    def test_circle_o2(self):
        for axis in [0., np.pi/2., np.pi/3., np.pi/4]:
            self.check_isom(CircleO2(axis))
            
        for _ in range(10):
            axis = np.random.rand()*np.pi
            self.check_isom(CircleO2(axis))

    def test_sphere_so3(self):
        self.check_isom(SphereSO3())
        
    def test_sphere_o3(self):
        self.check_isom(SphereO3())

    def check_isom(self, S: SpaceIsomorphism):
        
        S._test_section_consistency()

        S._test_custom_basis_consistency()

        S._test_equivariance()
        
        
if __name__ == '__main__':
    unittest.main()
