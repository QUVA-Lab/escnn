import unittest
from unittest import TestCase

from escnn.gspaces import *
from escnn.group import *
import numpy as np


class TestRestrictGSpace(TestCase):
    
    ####################################################################################################################
    ###                         ########################################################################################
    ###     2D   isometries     ########################################################################################
    ###                         ########################################################################################
    ####################################################################################################################

    def test_restrict_rotations(self):
        
        space = rot2dOnR2(-1, maximum_frequency=10)
        
        subspace, mapping, _ = space.restrict(4)
        
        self.assertIsInstance(subspace, GSpace2D)
        self.assertEqual(subspace.fibergroup.order(), 4)
        
        self.check_restriction(space, 4)

    def test_restrict_rotations_to_trivial(self):
    
        space = rot2dOnR2(-1, maximum_frequency=10)
    
        subspace, mapping, _ = space.restrict(1)
    
        self.assertIsInstance(subspace, GSpace2D)
        self.assertEqual(subspace.fibergroup.order(), 1)

        self.check_restriction(space, 1)

    def test_restrict_flipsrotations(self):
    
        space = flipRot2dOnR2(-1, maximum_frequency=10)
        
        N=10
        for axis in range(13):
            axis = axis * np.pi / (13*N)
            assert axis <= np.pi / N
            subspace, mapping, _ = space.restrict((axis, N))
    
            self.assertIsInstance(subspace, GSpace2D)
            self.assertEqual(subspace.fibergroup.order(), 2 * N)
    
            self.check_restriction(space, (axis, N))
        
    def test_restrict_flipsrotations_to_rotations(self):
    
        space = flipRot2dOnR2(-1, maximum_frequency=10)
    
        subspace, mapping, _ = space.restrict((None, -1))
    
        self.assertIsInstance(subspace, GSpace2D)
        self.assertEqual(subspace.fibergroup.order(), -1)

        self.check_restriction(space, (None, -1))

    def test_restrict_flipsrotations_to_flips(self):
    
        space = flipRot2dOnR2(-1, maximum_frequency=10)

        for axis in range(13):
            axis = axis * np.pi/13.
            subspace, mapping, _ = space.restrict((axis, 1))
        
            self.assertIsInstance(subspace, GSpace2D)
            self.assertEqual(subspace.fibergroup.order(), 2)

            self.check_restriction(space, (axis, 1))
        
    def test_restrict_fliprotations_to_trivial(self):
    
        space = flipRot2dOnR2(-1, maximum_frequency=10)
    
        subspace, mapping, _ = space.restrict((None, 1))
    
        self.assertIsInstance(subspace, GSpace2D)
        self.assertEqual(subspace.fibergroup.order(), 1)
        
        self.check_restriction(space, (None, 1))

    def test_restrict_flips_to_trivial(self):
    
        space = flip2dOnR2()
    
        subspace, mapping, _ = space.restrict(1)
    
        self.assertIsInstance(subspace, GSpace2D)
        self.assertEqual(subspace.fibergroup.order(), 1)

        self.check_restriction(space, 1)

    ####################################################################################################################
    ###                         ########################################################################################
    ###     3D   isometries     ########################################################################################
    ###                         ########################################################################################
    ####################################################################################################################

    ####### SO(3) ######################################################################################################
    
    def test_restrict_3d_rotations_to_ico(self):
    
        space = rot3dOnR3(maximum_frequency=2)
    
        subspace, mapping, _ = space.restrict('ico')
    
        self.assertIsInstance(subspace, GSpace3D)
        self.assertIsInstance(subspace.fibergroup, Icosahedral)
    
        self.check_restriction(space, 'ico')

    def test_restrict_3d_rotations_to_octa(self):

        space = rot3dOnR3(maximum_frequency=2)

        subspace, mapping, _ = space.restrict('octa')

        self.assertIsInstance(subspace, GSpace3D)
        self.assertIsInstance(subspace.fibergroup, Octahedral)

        self.check_restriction(space, 'octa')

    def test_restrict_3d_rotations_to_2d_rotations(self):
    
        space = rot3dOnR3(maximum_frequency=2)
    
        for n in [-1, 1, 2, 4, 7]:
            sg_id = False, n
            
            subspace, mapping, _ = space.restrict(sg_id)
        
            self.assertIsInstance(subspace, GSpace3D)
            if n > 0:
                self.assertIsInstance(subspace.fibergroup, CyclicGroup)
            else:
                self.assertIsInstance(subspace.fibergroup, SO2)

            self.assertEqual(subspace.fibergroup.order(), n)

            self.check_restriction(space, sg_id)

    def test_restrict_3d_rotations_to_2d_rotations_reflections(self):
    
        space = rot3dOnR3(maximum_frequency=2)
    
        for n in [2, 4]:
            for axis in [0., np.pi/2, np.pi/4]:
                sg_id = axis, n
            
                subspace, mapping, _ = space.restrict(sg_id)
            
                self.assertIsInstance(subspace, GSpace3D)
                self.assertIsInstance(subspace.fibergroup, DihedralGroup)
                self.assertEqual(subspace.fibergroup.order(), 2*n)
            
                self.check_restriction(space, sg_id)

    def test_restrict_3d_rotations_to_2d_reflection(self):
    
        space = rot3dOnR3(maximum_frequency=2)
    
        for axis in [0., np.pi / 2, np.pi / 4]:
            sg_id = axis, 1
        
            subspace, mapping, _ = space.restrict(sg_id)
        
            self.assertIsInstance(subspace, GSpace3D)
            self.assertIsInstance(subspace.fibergroup, CyclicGroup)
            self.assertEqual(subspace.fibergroup.order(), 2)
        
            self.check_restriction(space, sg_id)

    ####### O(3) #######################################################################################################

    def test_restrict_3d_rotationinversion_so3(self):
        space = flipRot3dOnR3(maximum_frequency=2)
        sg_id = 'so3'
        
        subspace, mapping, _ = space.restrict(sg_id)

        self.assertIsInstance(subspace, GSpace3D)
        self.assertIsInstance(subspace.fibergroup, SO3)

        self.check_restriction(space, sg_id)

    def test_restrict_3d_rotationinversion_ico(self):
        space = flipRot3dOnR3(maximum_frequency=2)
        sg_id = False, 'ico'
    
        subspace, mapping, _ = space.restrict(sg_id)
    
        self.assertIsInstance(subspace, GSpace3D)
        self.assertIsInstance(subspace.fibergroup, Icosahedral)
    
        self.check_restriction(space, sg_id)

    def test_restrict_3d_rotationinversion_dih_o2(self):
        space = flipRot3dOnR3(maximum_frequency=2)

        for axis in [0., np.pi / 2, np.pi / 3, 2 * np.pi]:
            for n in [2, 4]:
                sg_id = (False, axis, n)

                subspace, mapping, _ = space.restrict(sg_id)

                self.assertIsInstance(subspace, GSpace3D)
                self.assertIsInstance(subspace.fibergroup, DihedralGroup)
                self.assertEqual(subspace.fibergroup.order(), 2*n)

                self.check_restriction(space, sg_id)

    def test_restrict_3d_rotationinversion_so2(self):
    
        space = flipRot3dOnR3(maximum_frequency=2)

        for n in [2, 4]:
            sg_id = (False, False, n)
            
            subspace, mapping, _ = space.restrict(sg_id)

            self.assertIsInstance(subspace, GSpace3D)
            self.assertIsInstance(subspace.fibergroup, CyclicGroup)
            self.assertEqual(subspace.fibergroup.order(), n)

            self.check_restriction(space, sg_id)

    def test_restrict_3d_rotationinversion_flip(self):

        space = flipRot3dOnR3(maximum_frequency=2)

        for axis in [0., np.pi / 2, np.pi / 3, 2 * np.pi]:
        
            sg_id = (False, 2*axis, 1)
            
            subspace, mapping, _ = space.restrict(sg_id)

            self.assertIsInstance(subspace, GSpace3D)
            self.assertIsInstance(subspace.fibergroup, CyclicGroup)

            self.check_restriction(space, sg_id)

    def test_restrict_3d_rotationinversion_cone_o2(self):
    
        space = flipRot3dOnR3(maximum_frequency=2)

        for n in [2, 4]:
            # Cone aligned along Z axis
            # i.e., rotation along Z axis
            # on XY plane, mirror wrt Y axis (i.e. flip along X axis)
            sg_id = ('cone', 0., n)
            subspace, mapping, _ = space.restrict(sg_id)

            self.assertIsInstance(subspace, GSpace3D)
            self.assertIsInstance(subspace.fibergroup, DihedralGroup)

            self.check_restriction(space, sg_id)

            # Cone aligned along Z axis
            # i.e., rotation along Z axis
            # on XY plane, mirror wrt X axis (i.e. flip along Y axis)
            sg_id = ('cone', np.pi, n)
            subspace, mapping, _ = space.restrict(sg_id)

            self.assertIsInstance(subspace, GSpace3D)
            self.assertIsInstance(subspace.fibergroup, DihedralGroup)

            self.check_restriction(space, sg_id)

            xyz = space.fibergroup.element(
                (
                    0,
                    np.array([
                        np.sqrt(1./3.) * np.sin(np.pi*2/3),
                        np.sqrt(1./3.) * np.sin(np.pi*2/3),
                        np.sqrt(1./3.) * np.sin(np.pi*2/3),
                        np.cos(np.pi*2/3),
                    ])
                ),
                'Q'
            )

            for adj in [xyz, xyz@xyz, xyz @ space.fibergroup.inversion, xyz @ xyz @ space.fibergroup.inversion]:
                sg_id = ('cone', 0., n, adj)
                subspace, mapping, _ = space.restrict(sg_id)
        
                self.assertIsInstance(subspace, GSpace3D)
                self.assertIsInstance(subspace.fibergroup, DihedralGroup)
        
                self.check_restriction(space, sg_id)

    def test_restrict_3d_rotationinversion_mir(self):
    
        space = flipRot3dOnR3(maximum_frequency=2)
    
        # mirror wrt Y axis (i.e. flip along X axis)
        sg_id = ('cone', 0., 1)
        subspace, mapping, _ = space.restrict(sg_id)
    
        self.assertIsInstance(subspace, GSpace3D)
        self.assertIsInstance(subspace.fibergroup, CyclicGroup)
    
        self.check_restriction(space, sg_id)
        
        # mirror wrt X axis (i.e. flip along Y axis)
        sg_id = ('cone', np.pi, 1)
        subspace, mapping, _ = space.restrict(sg_id)
    
        self.assertIsInstance(subspace, GSpace3D)
        self.assertIsInstance(subspace.fibergroup, CyclicGroup)
    
        self.check_restriction(space, sg_id)
    
        xyz = space.fibergroup.element(
            (
                0,
                np.array([
                    np.sqrt(1. / 3.) * np.sin(np.pi * 2 / 3),
                    np.sqrt(1. / 3.) * np.sin(np.pi * 2 / 3),
                    np.sqrt(1. / 3.) * np.sin(np.pi * 2 / 3),
                    np.cos(np.pi * 2 / 3),
                    ])
            ),
            'Q'
        )
    
        for adj in [xyz, xyz @ xyz, xyz @ space.fibergroup.inversion, xyz @ xyz @ space.fibergroup.inversion]:
            sg_id = ('cone', 0., 1, adj)
            subspace, mapping, _ = space.restrict(sg_id)
        
            self.assertIsInstance(subspace, GSpace3D)
            self.assertIsInstance(subspace.fibergroup, CyclicGroup)
        
            self.check_restriction(space, sg_id)

    def test_restrict_3d_rotationinversion_fullcylinder_c2xo2(self):

        space = flipRot3dOnR3(maximum_frequency=2)

        for n in [2, 4]:
            sg_id = (True, True, n)

            subspace, mapping, _ = space.restrict(sg_id)

            self.assertIsInstance(subspace, GSpace3D)
            self.assertIsInstance(subspace.fibergroup, DirectProductGroup)
            self.assertIsInstance(subspace.fibergroup.G1, CyclicGroup) and subspace.fibergroup.order() == 2
            self.assertIsInstance(subspace.fibergroup.G2, DihedralGroup) and subspace.fibergroup.order() == 2*n

            self.check_restriction(space, sg_id)

    def test_restrict_3d_rotationinversion_cylinder_c2xso2(self):

        space = flipRot3dOnR3(maximum_frequency=2)

        for n in [2, 4]:
            sg_id = (True, False, n)

            subspace, mapping, _ = space.restrict(sg_id)

            self.assertIsInstance(subspace, GSpace3D)
            self.assertIsInstance(subspace.fibergroup, DirectProductGroup)
            self.assertIsInstance(subspace.fibergroup.G1, CyclicGroup) and subspace.fibergroup.order() == 2
            self.assertIsInstance(subspace.fibergroup.G2, CyclicGroup) and subspace.fibergroup.order() == n

            self.check_restriction(space, sg_id)

    def test_restrict_3d_rotationinversion_inv(self):
    
        space = flipRot3dOnR3(maximum_frequency=2)
    
        # Inversion wrt the origin
        sg_id = (True, False, 1)
        
        subspace, mapping, _ = space.restrict(sg_id)

        self.assertIsInstance(subspace, GSpace3D)
        self.assertIsInstance(subspace.fibergroup, CyclicGroup)

        self.check_restriction(space, sg_id)

    ####################################################################################################################
    def check_restriction(self, space: GSpace, subgroup_id):
        subspace, parent, child = space.restrict(subgroup_id)
        
        # rho = space.trivial_repr
        irreps = space.fibergroup.irreps()
        for rho in irreps:
            sub_rho = rho.restrict(subgroup_id)

            assert sub_rho.group == subspace.fibergroup, (sub_rho.group, subspace.fibergroup, space.fibergroup, subgroup_id)
            
            x = np.random.randn(1, rho.size, 5, 5, 5)
            
            for e in subspace.testing_elements:
                
                y1 = space.featurefield_action(x, rho, parent(e))
                y2 = subspace.featurefield_action(x, sub_rho, e)

                self.assertTrue(np.allclose(y1, y2), msg=f"{space.name} -> {subgroup_id}: {parent(e)} -> {e}")
        

if __name__ == '__main__':
    unittest.main()
