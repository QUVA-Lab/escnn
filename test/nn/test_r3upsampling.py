import unittest
from unittest import TestCase

from escnn.nn import *
from escnn.gspaces import *

import numpy as np


class TestUpsampling(TestCase):

    def test_cyclic_even_trilinear(self):
        g = rot2dOnR3(8)
        self.check_upsampling_scale(g, "trilinear")
        self.check_upsampling_size(g, "trilinear")

    def test_cyclic_odd_trilinear(self):
        g = rot2dOnR3(9)
        self.check_upsampling_scale(g, "trilinear")
        self.check_upsampling_size(g, "trilinear")

    def test_dihedral_even_trilinear(self):
        g = conicalOnR3(8)
        self.check_upsampling_scale(g, "trilinear")
        self.check_upsampling_size(g, "trilinear")

    def test_dihedral_odd_trilinear(self):
        g = conicalOnR3(9)
        self.check_upsampling_scale(g, "trilinear")
        self.check_upsampling_size(g, "trilinear")

    def test_cube_trilinear(self):
        g = octaOnR3()
        self.check_upsampling_scale(g, "trilinear")
        self.check_upsampling_size(g, "trilinear")

    def test_ico_trilinear(self):
        g = icoOnR3()
        self.check_upsampling_scale(g, "trilinear")
        self.check_upsampling_size(g, "trilinear")

    def test_mirr_trilinear(self):
        g = mirOnR3()
        self.check_upsampling_scale(g, "trilinear")
        self.check_upsampling_size(g, "trilinear")

    def test_inv_trilinear(self):
        g = invOnR3()
        self.check_upsampling_scale(g, "trilinear")
        self.check_upsampling_size(g, "trilinear")

    def test_so2_trilinear(self):
        g = rot2dOnR3()
        self.check_upsampling_scale(g, "trilinear")
        self.check_upsampling_size(g, "trilinear")

    def test_dihedral_trilinear(self):
        g = dihedralOnR3()
        self.check_upsampling_scale(g, "trilinear")
        self.check_upsampling_size(g, "trilinear")

    def test_cone_trilinear(self):
        g = conicalOnR3()
        self.check_upsampling_scale(g, "trilinear")
        self.check_upsampling_size(g, "trilinear")

    def test_so3_trilinear(self):
        g = rot3dOnR3()
        self.check_upsampling_scale(g, "trilinear")
        self.check_upsampling_size(g, "trilinear")

    def test_o3_trilinear(self):
        g = flipRot3dOnR3()
        self.check_upsampling_scale(g, "trilinear")
        self.check_upsampling_size(g, "trilinear")


    #################################################################################
    # "NEAREST" method is not equivariant!! As a result, all the following tests fail

    def test_cyclic_even_nearest(self):
        g = rot2dOnR3(8)
        self.check_upsampling_scale(g, "nearest")
        self.check_upsampling_size(g, "nearest")

    def test_cyclic_odd_nearest(self):
        g = rot2dOnR3(9)
        self.check_upsampling_scale(g, "nearest")
        self.check_upsampling_size(g, "nearest")

    def test_dihedral_even_nearest(self):
        g = conicalOnR3(8)
        self.check_upsampling_scale(g, "nearest")
        self.check_upsampling_size(g, "nearest")

    def test_dihedral_odd_nearest(self):
        g = conicalOnR3(9)
        self.check_upsampling_scale(g, "nearest")
        self.check_upsampling_size(g, "nearest")

    def test_cube_nearest(self):
        g = octaOnR3()
        self.check_upsampling_scale(g, "nearest")
        self.check_upsampling_size(g, "nearest")

    def test_ico_nearest(self):
        g = icoOnR3()
        self.check_upsampling_scale(g, "nearest")
        self.check_upsampling_size(g, "nearest")

    def test_mirr_nearest(self):
        g = mirOnR3()
        self.check_upsampling_scale(g, "nearest")
        self.check_upsampling_size(g, "nearest")

    def test_inv_nearest(self):
        g = invOnR3()
        self.check_upsampling_scale(g, "nearest")
        self.check_upsampling_size(g, "nearest")

    def test_so2_nearest(self):
        g = rot2dOnR3()
        self.check_upsampling_scale(g, "nearest")
        self.check_upsampling_size(g, "nearest")

    def test_dihedral_nearest(self):
        g = dihedralOnR3()
        self.check_upsampling_scale(g, "nearest")
        self.check_upsampling_size(g, "nearest")

    def test_cone_nearest(self):
        g = conicalOnR3()
        self.check_upsampling_scale(g, "nearest")
        self.check_upsampling_size(g, "nearest")

    def test_so3_nearest(self):
        g = rot3dOnR3()
        self.check_upsampling_scale(g, "nearest")
        self.check_upsampling_size(g, "nearest")

    def test_o3_nearest(self):
        g = flipRot3dOnR3()
        self.check_upsampling_scale(g, "nearest")
        self.check_upsampling_size(g, "nearest")


#################################################################################

    def check_upsampling_scale(self, g, mode):
        for s in [2, 3]:
            print(f"\nScale: {s}\n")
            for r in g.representations.values():
                r1 = FieldType(g, [r])
                ul = R3Upsampling(r1, mode=mode, scale_factor=s)
                ul.check_equivariance()

    def check_upsampling_size(self, g, mode):
        # for s in [71, 129]:
        for s in [23, 51]:
            print(f"\nSize: {s}\n")
            for r in g.representations.values():
                r1 = FieldType(g, [r])
                ul = R3Upsampling(r1, mode=mode, size=s)
                ul.check_equivariance()

        
if __name__ == '__main__':
    unittest.main()
