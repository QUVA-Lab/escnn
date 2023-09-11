import unittest
from unittest import TestCase

from escnn.group import *

import numpy as np


class TestSubgroups(TestCase):
    
    def test_restrict_dihedral_odd_dihedral(self):
        N = 9
        dg = DihedralGroup(N)
        for rot in range(1, N):
            if N % rot == 0:
                for axis in range(int(N // rot)):
                    sg_id = (axis, rot)
                    self.check_restriction(dg, sg_id)
    
    def test_restrict_dihedral_odd_cyclic_odd(self):
        dg = DihedralGroup(9)
        sg_id = (None, 3)
        self.check_restriction(dg, sg_id)
    
    def test_restrict_dihedral_odd_flips(self):
        dg = DihedralGroup(11)
        for axis in range(11):
            sg_id = (axis, 1)
            self.check_restriction(dg, sg_id)
    
    def test_restrict_cyclic_odd_cyclic_odd(self):
        dg = CyclicGroup(9)
        sg_id = 3
        self.check_restriction(dg, sg_id)
            
    def test_restrict_dihedral_even_dihedral(self):
        N = 12
        dg = DihedralGroup(N)
        for rot in range(1, N):
            if N % rot == 0:
                for axis in range(int(N//rot)):
                    sg_id = (axis, rot)
                    self.check_restriction(dg, sg_id)

    def test_restrict_dihedral_even_cyclic_even(self):
        dg = DihedralGroup(12)
        sg_id = (None, 4)
        self.check_restriction(dg, sg_id)

    def test_restrict_dihedral_even_cyclic_odd(self):
        dg = DihedralGroup(12)
        sg_id = (None, 3)
        self.check_restriction(dg, sg_id)
        
    def test_restrict_dihedral_even_flips(self):
        dg = DihedralGroup(12)
        sg_id = (1, 1)
        self.check_restriction(dg, sg_id)

    def test_restrict_cyclic_even_cyclic_even(self):
        dg = CyclicGroup(8)
        sg_id = 2
        self.check_restriction(dg, sg_id)
        
    def test_restrict_cyclic_even_cyclic_odd(self):
        dg = CyclicGroup(10)
        sg_id = 5
        self.check_restriction(dg, sg_id)

    def test_restrict_dihedral_even_cyclic(self):
        dg = DihedralGroup(12)
        sg_id = (None, 12)
        self.check_restriction(dg, sg_id)
        
    def test_restrict_dihedral_odd_cyclic(self):
        dg = DihedralGroup(13)
        sg_id = (None, 13)
        self.check_restriction(dg, sg_id)
        
    def test_restrict_o2_flips(self):
        dg = O2(10)
        S = 7
        for axis in range(S):
            sg_id = (axis*2*np.pi/S, 1)
            
            self.check_restriction(dg, sg_id)

    def test_restrict_o2_dihedral_even(self):
        dg = O2(10)
        S = 7
        for axis in range(S):
            sg_id = (axis*2*np.pi/(6*S), 6)
            self.check_restriction(dg, sg_id)

    def test_restrict_o2_dihedral_odd(self):
        dg = O2(10)
        S = 7
        for axis in range(S):
            sg_id = (axis*2*np.pi/(5*S), 5)
            self.check_restriction(dg, sg_id)

    def test_restrict_o2_so2(self):
        dg = O2(10)
        sg_id = (None, -1)
        self.check_restriction(dg, sg_id)
        
    def test_restrict_o2_cyclic_even(self):
        dg = O2(10)
        sg_id = (None, 4)
        self.check_restriction(dg, sg_id)

    def test_restrict_o2_cyclic_odd(self):
        dg = O2(10)
        sg_id = (None, 3)
        self.check_restriction(dg, sg_id)

    def test_restrict_so2_cyclic_even(self):
        dg = SO2(10)
        sg_id = 8
        self.check_restriction(dg, sg_id)

    def test_restrict_so2_cyclic_odd(self):
        dg = SO2(10)
        sg_id = 7
        self.check_restriction(dg, sg_id)

    def test_restrict_o2_o2(self):
        dg = O2(10)
        for a in [0., np.pi/2, np.pi/3, np.pi/4]:
            sg_id = (2*a, -1)
            self.check_restriction(dg, sg_id)

    ####### Octa #######################################################################################################

    def test_restrict_octa_tetra(self):
        dg = octa_group()
        raise NotImplementedError()

    def test_restrict_octa_dn(self):
        dg = octa_group()
        for n in [2, 3, 4]:
            sg_id = (True, n)
            self.check_restriction(dg, sg_id)

            sg, parent_element, child_element = dg.subgroup(sg_id)

            flip_sg = parent_element(sg.element((1, 0))).to('MAT')
            flip_pg = dg.element(
                np.array([1., 0., 0., 0.]),
                'Q'
            ).to('MAT')
            self.assertTrue(np.allclose(flip_sg, flip_pg))

            for _ in range(8):
                adj = dg.sample()
                sg_id = (True, n, adj)
                self.check_restriction(dg, sg_id)

    def test_restrict_octa_cn(self):
        dg = octa_group()
        for n in [1, 2, 3, 4]:
            sg_id = (False, n)
            self.check_restriction(dg, sg_id)

            for _ in range(8):
                adj = dg.sample()
                sg_id = (False, n, adj)
                self.check_restriction(dg, sg_id)

    def test_restrict_octa_flip(self):
        dg = octa_group()

        sg_id = (True, 1)
        self.check_restriction(dg, sg_id)

        sg, parent_element, child_element = dg.subgroup(sg_id)

        flip_sg = parent_element(sg.element(1)).to('MAT')
        flip_pg = dg.element(
            np.array([1., 0., 0., 0.]),
            'Q'
        ).to('MAT')
        self.assertTrue(np.allclose(flip_sg, flip_pg))

        for _ in range(8):
            adj = dg.sample()
            sg_id = (True, 1, adj)
            self.check_restriction(dg, sg_id)

    ####### Ico ########################################################################################################

    def test_restrict_ico_tetra(self):
        dg = ico_group()
        raise NotImplementedError()

    def test_restrict_ico_dn(self):
        dg = ico_group()
        for n in [2, 3, 5]:
            sg_id = (True, n)
            self.check_restriction(dg, sg_id)

            sg, parent_element, child_element = dg.subgroup(sg_id)

            flip_sg = parent_element(sg.element((1, 0))).to('MAT')
            flip_pg = dg.element(
                np.array([1., 0., 0., 0.]),
                'Q'
            ).to('MAT')
            self.assertTrue(np.allclose(flip_sg, flip_pg))

            for _ in range(8):
                adj = dg.sample()
                sg_id = (True, n, adj)
                self.check_restriction(dg, sg_id)

    def test_restrict_ico_cn(self):
        dg = ico_group()
        for n in [1, 2, 3, 5]:
            sg_id = (False, n)
            self.check_restriction(dg, sg_id)

            for _ in range(8):
                adj = dg.sample()
                sg_id = (False, n, adj)
                self.check_restriction(dg, sg_id)

    def test_restrict_ico_flip(self):
        dg = ico_group()

        sg_id = (True, 1)
        self.check_restriction(dg, sg_id)

        sg, parent_element, child_element = dg.subgroup(sg_id)

        flip_sg = parent_element(sg.element(1)).to('MAT')
        flip_pg = dg.element(
            np.array([1., 0., 0., 0.]),
            'Q'
        ).to('MAT')
        self.assertTrue(np.allclose(flip_sg, flip_pg))

        for _ in range(8):
            adj = dg.sample()
            sg_id = (True, 1, adj)
            self.check_restriction(dg, sg_id)

    ####### SO(3) ######################################################################################################

    def test_restrict_so3_o2(self):
        dg = SO3(1)
        for axis in [0., np.pi/2, np.pi/3, np.pi * 1.25, 2*np.pi]:
            # print(axis)
            sg_id = (axis, -1)
            self.check_restriction(dg, sg_id)
            
            o2, parent, child = dg.subgroup(sg_id)

            for _ in range(10):
                h = o2.sample()
                g = parent(h)
    
                flip, rot = h.to('MAT')
                flip = -1. if (flip == 1) else 1.
                hmat = np.eye(3)
                hmat[:2, :2] = rot
                
                if flip < 0.:
                    axis_rot = o2.element((0, axis)).to('MAT')[1]
                    hmat[:2, :2] = hmat[:2, :2] @ axis_rot

                hmat = hmat @ np.asarray([
                    [1., 0., 0.],
                    [0., flip, 0.],
                    [0., 0., flip]
                ])
                
                self.assertTrue(
                    np.allclose(
                        hmat,
                        g.to('MAT')
                    ),
                    f"{hmat}\nvs\n{g.to('MAT')}\n"
                )

            for _ in range(5):
                adj = dg.sample()
                sg_id = (axis, -1, adj)
                self.check_restriction(dg, sg_id)

    def test_restrict_so3_so2(self):
        dg = SO3(1)
        sg_id = (False, -1)
        self.check_restriction(dg, sg_id)
        
        so2, parent, child = dg.subgroup(sg_id)
        
        for _ in range(30):
            h = so2.sample()
            g = parent(h)
            
            hmat = np.eye(3)
            hmat[:2, :2] = h.to('MAT')
            self.assertTrue(
                np.allclose(
                    hmat,
                    g.to('MAT')
                )
            )
        
        for _ in range(5):
            adj = dg.sample()
            sg_id = (False, -1, adj)
            self.check_restriction(dg, sg_id)

    def test_restrict_so3_dn(self):
        dg = SO3(1)
        for n in [3, 4, 6, 9]:
            for axis in [0., np.pi / 2, np.pi / 3, np.pi * 1.25, 2 * np.pi]:
                sg_id = (2*axis, n)
                self.check_restriction(dg, sg_id)
                
                sg, parent_element, child_element = dg.subgroup(sg_id)

                flip_sg = parent_element(sg.element((1, 0))).to('MAT')
                flip_pg = dg.element(
                    np.array([np.cos(axis), np.sin(axis), 0, 0.]),
                    'Q'
                ).to('MAT')
                self.assertTrue(np.allclose(flip_sg, flip_pg))

                for _ in range(5):
                    adj = dg.sample()
                    sg_id = (2*axis, n, adj)
                    self.check_restriction(dg, sg_id)

    def test_restrict_so3_cn(self):
        dg = SO3(1)
        for n in [2, 3, 4, 6, 9]:
            
            sg_id = (False, n)
            self.check_restriction(dg, sg_id)
        
            for _ in range(5):
                adj = dg.sample()
                sg_id = (False, n, adj)
                self.check_restriction(dg, sg_id)

    def test_restrict_so3_flip(self):
        dg = SO3(1)
        for axis in [0., np.pi / 2, np.pi / 3, np.pi * 1.25, 2 * np.pi]:
    
            sg_id = (2*axis, 1)
            self.check_restriction(dg, sg_id)
            
            sg, parent_element, child_element = dg.subgroup(sg_id)
            
            flip_sg = parent_element(sg.element(1)).to('MAT')
            flip_pg = dg.element(
                np.array([np.cos(axis), np.sin(axis), 0, 0.]),
                'Q'
            ).to('MAT')
            self.assertTrue(np.allclose(flip_sg, flip_pg))

            for _ in range(5):
                adj = dg.sample()
                sg_id = (2*axis, 1, adj)
                self.check_restriction(dg, sg_id)

    def test_restrict_so3_ico(self):
        dg = SO3(1)
        sg_id = 'ico'
        self.check_restriction(dg, sg_id)
        
        for _ in range(5):
            adj = dg.sample()
            sg_id = ('ico', adj)
            self.check_restriction(dg, sg_id)

    def test_restrict_so3_octa(self):
        dg = SO3(1)
        sg_id = 'octa'
        self.check_restriction(dg, sg_id)

        for _ in range(5):
            adj = dg.sample()
            sg_id = ('octa', adj)
            self.check_restriction(dg, sg_id)

    ####### O(3) #######################################################################################################
    
    def test_restrict_o3_fullico(self):
        dg = O3(1)
        sg_id = (True, 'ico')
        self.check_restriction(dg, sg_id)
    
        for _ in range(5):
            adj = dg.sample()
            sg_id = (True, 'ico', adj)
            self.check_restriction(dg, sg_id)

    def test_restrict_o3_so3(self):
        dg = O3(1)
        sg_id = 'so3'
        self.check_restriction(dg, sg_id)
    
        for _ in range(5):
            adj = dg.sample()
            sg_id = ('so3', adj)
            self.check_restriction(dg, sg_id)

    def test_restrict_o3_ico(self):
        dg = O3(1)
        sg_id = (False, 'ico')
        self.check_restriction(dg, sg_id)
    
        for _ in range(5):
            adj = dg.sample()
            sg_id = (False, 'ico', adj)
            self.check_restriction(dg, sg_id)

    def test_restrict_o3_octa(self):
        dg = O3(1)
        sg_id = (False, 'octa')
        self.check_restriction(dg, sg_id)

        for _ in range(5):
            adj = dg.sample()
            sg_id = (False, 'octa', adj)
            self.check_restriction(dg, sg_id)

    def test_restrict_o3_dih_o2(self):
        dg = O3(1)
        for axis in [0., np.pi / 2, np.pi / 3, np.pi * 1.25, 2 * np.pi]:
            sg_id = (False, 2*axis, -1)
            self.check_restriction(dg, sg_id)
        
            for _ in range(5):
                adj = dg.sample()
                assert adj.group == dg
                sg_id = (False, 2*axis, -1, adj)
                self.check_restriction(dg, sg_id)
                
        for _ in range(5):
            adj = dg.sample()
            assert adj.group == dg
            sg_id = (False, True, -1, adj)
            self.check_restriction(dg, sg_id)

            sg_id2 = (False, 0., -1, adj)
            
            sg1, p1, c1 = dg.subgroup(sg_id)
            sg2, p2, c2 = dg.subgroup(sg_id2)
            
            self.assertTrue(sg1, sg2)
            for _ in range(20):
                g = sg1.sample()
                self.assertEqual(
                    p1(g), p2(g)
                )
                self.assertEqual(
                    c1(p1(g)), c2(p2(g))
                )

    def test_restrict_o3_so2(self):
        dg = O3(1)
        sg_id = (False, False, -1)
        self.check_restriction(dg, sg_id)
    
        for _ in range(5):
            adj = dg.sample()
            sg_id = (False, False, -1, adj)
            self.check_restriction(dg, sg_id)

    def test_restrict_o3_dih_dn(self):
        dg = O3(1)
        for n in [3, 4, 6, 9]:
            for axis in [0., np.pi / 2, np.pi / 3, np.pi * 1.25, 2 * np.pi]:
                sg_id = (False, 2*axis, n)
                self.check_restriction(dg, sg_id)
            
                sg, parent_element, child_element = dg.subgroup(sg_id)
                
                assert sg.order() == 2 * n, (axis, sg.order(), n, sg.name)
            
                _, flip_sg = parent_element(sg.element((1, 0))).to('MAT')
                _, flip_pg = dg.element(
                    (0, np.array([np.cos(axis), np.sin(axis), 0, 0.])),
                    'Q'
                ).to('MAT')
                self.assertTrue(np.allclose(flip_sg, flip_pg))

                for _ in range(5):
                    adj = dg.sample()
                    sg_id = (False, 2*axis, n, adj)
                    self.check_restriction(dg, sg_id)
                    
            for _ in range(5):
                adj = dg.sample()
                assert adj.group == dg
                sg_id = (False, True, n, adj)
                self.check_restriction(dg, sg_id)
        
                sg_id2 = (False, 0., n, adj)
        
                sg1, p1, c1 = dg.subgroup(sg_id)
                sg2, p2, c2 = dg.subgroup(sg_id2)
        
                self.assertEqual(sg1, sg2)
                for _ in range(20):
                    g = sg1.sample()
                    self.assertEqual(
                        p1(g), p2(g)
                    )
                    self.assertEqual(
                        c1(p1(g)), c2(p2(g))
                    )

    def test_restrict_o3_cn(self):
        dg = O3(1)
        for n in [2, 3, 4, 6, 9]:
        
            sg_id = (False, False, n)
            self.check_restriction(dg, sg_id)
        
            for _ in range(5):
                adj = dg.sample()
                sg_id = (False, False, n, adj)
                self.check_restriction(dg, sg_id)

    def test_restrict_o3_flip(self):
        dg = O3(1)
        for axis in [0., np.pi / 2, np.pi / 3, np.pi * 1.25, 2 * np.pi]:
        
            sg_id = (False, 2*axis, 1)
            self.check_restriction(dg, sg_id)
        
            sg, parent_element, child_element = dg.subgroup(sg_id)
        
            _, flip_sg = parent_element(sg.element(1)).to('MAT')
            _, flip_pg = dg.element(
                (0, np.array([np.cos(axis), np.sin(axis), 0, 0.])),
                'Q'
            ).to('MAT')
            self.assertTrue(np.allclose(flip_sg, flip_pg))
        
            for _ in range(5):
                adj = dg.sample()
                sg_id = (False, 2*axis, 1, adj)
                self.check_restriction(dg, sg_id)

    def test_restrict_o3_cone_o2(self):
        dg = O3(1)
    
        # Cone aligned along Z axis
        # i.e., rotation along Z axis
        # on XY plane, mirror wrt Y axis (i.e. flip along X axis)
        sg_id = ('cone', 0., -1)
        self.check_restriction(dg, sg_id)
    
        sg, parent_element, child_element = dg.subgroup(sg_id)
        flip, flip_sg = parent_element(sg.element((1, 0.))).to('MAT')
        flip = -1 if flip else 1
        flip_sg *= flip
        flip_pg = np.array([
            [1., 0., 0.],
            [0., -1., 0.],
            [0., 0., 1.],
        ])
        self.assertTrue(np.allclose(flip_sg, flip_pg))

        # Cone aligned along Z axis
        # i.e., rotation along Z axis
        # on XY plane, mirror wrt X axis (i.e. flip along Y axis)
        sg_id = ('cone', np.pi, -1)
        self.check_restriction(dg, sg_id)
    
        sg, parent_element, child_element = dg.subgroup(sg_id)
        flip, flip_sg = parent_element(sg.element((1, 0.))).to('MAT')
        flip = -1 if flip else 1
        flip_sg *= flip
        flip_pg = np.array([
            [-1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ])
        self.assertTrue(np.allclose(flip_sg, flip_pg))
    
        for _ in range(5):
            adj = dg.sample()
            sg_id = ('cone', -1, adj)
            self.check_restriction(dg, sg_id)
        
            sg, parent_element, child_element = dg.subgroup(sg_id)
            flip, flip_sg = parent_element(sg.element((1, 0.))).to('MAT')
            flip = -1 if flip else 1
            flip_sg *= flip
            flip_pg = np.array([
                [1., 0., 0.],
                [0., -1., 0.],
                [0., 0., 1.],
            ])
            flip_pg = adj.to('MAT')[1].T @ flip_pg @ adj.to('MAT')[1]
            self.assertTrue(np.allclose(flip_sg, flip_pg))

    def test_restrict_o3_cone_dn(self):
        dg = O3(1)
    
        for N in [2, 3, 5, 8]:
            # Cone aligned along Z axis
            # i.e., rotation along Z axis
            # on XY plane, mirror wrt Y axis (i.e. flip along X axis)
            sg_id = ('cone', 0., N)
            self.check_restriction(dg, sg_id)
        
            sg, parent_element, child_element = dg.subgroup(sg_id)
            flip, flip_sg = parent_element(sg.element((1, 0))).to('MAT')
            flip = -1 if flip else 1
            flip_sg *= flip
            flip_pg = np.array([
                [1., 0., 0.],
                [0., -1., 0.],
                [0., 0., 1.],
            ])
            self.assertTrue(np.allclose(flip_sg, flip_pg))
        
            # Cone aligned along Z axis
            # i.e., rotation along Z axis
            # on XY plane, mirror wrt X axis (i.e. flip along Y axis)
            sg_id = ('cone', np.pi, N)
            self.check_restriction(dg, sg_id)
        
            sg, parent_element, child_element = dg.subgroup(sg_id)
            flip, flip_sg = parent_element(sg.element((1, 0))).to('MAT')
            flip = -1 if flip else 1
            flip_sg *= flip
            flip_pg = np.array([
                [-1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
            ])
            self.assertTrue(np.allclose(flip_sg, flip_pg))
        
            for _ in range(5):
                adj = dg.sample()
                sg_id = ('cone', N, adj)
                self.check_restriction(dg, sg_id)
            
                sg, parent_element, child_element = dg.subgroup(sg_id)
                flip, flip_sg = parent_element(sg.element((1, 0))).to('MAT')
                flip = -1 if flip else 1
                flip_sg *= flip
                flip_pg = np.array([
                    [1., 0., 0.],
                    [0., -1., 0.],
                    [0., 0., 1.],
                ])
                flip_pg = adj.to('MAT')[1].T @ flip_pg @ adj.to('MAT')[1]
                self.assertTrue(np.allclose(flip_sg, flip_pg))

    def test_restrict_o3_mir(self):
        dg = O3(1)
    
        # Mirroring along the Y axis
        sg_id = ('cone', 0., 1)
        self.check_restriction(dg, sg_id)
    
        sg, parent_element, child_element = dg.subgroup(sg_id)
        flip, flip_sg = parent_element(sg.element(1)).to('MAT')
        flip = -1 if flip else 1
        flip_sg *= flip
        flip_pg = np.array([
            [1., 0., 0.],
            [0., -1., 0.],
            [0., 0., 1.],
        ])
        self.assertTrue(np.allclose(flip_sg, flip_pg))

        # Mirroring along the X axis
        sg_id = ('cone', np.pi, 1)
        self.check_restriction(dg, sg_id)

        sg, parent_element, child_element = dg.subgroup(sg_id)
        flip, flip_sg = parent_element(sg.element(1)).to('MAT')
        flip = -1 if flip else 1
        flip_sg *= flip
        flip_pg = np.array([
            [-1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ])
        self.assertTrue(np.allclose(flip_sg, flip_pg))
        
        for _ in range(5):
            adj = dg.sample()
            sg_id = ('cone', 1, adj)
            self.check_restriction(dg, sg_id)

            sg, parent_element, child_element = dg.subgroup(sg_id)
            flip, flip_sg = parent_element(sg.element(1)).to('MAT')
            flip = -1 if flip else 1
            flip_sg *= flip
            flip_pg = np.array([
                [1., 0., 0.],
                [0., -1., 0.],
                [0., 0., 1.],
            ])
            flip_pg = adj.to('MAT')[1].T @ flip_pg @ adj.to('MAT')[1]
            self.assertTrue(np.allclose(flip_sg, flip_pg))

    def test_restrict_o3_inv(self):
        dg = O3(1)
    
        # Inversion wrt the origin
        sg_id = (True, False, 1)
        self.check_restriction(dg, sg_id)
    
        sg, parent_element, child_element = dg.subgroup(sg_id)
        flip, flip_sg = parent_element(sg.element(1)).to('MAT')
        flip = -1 if flip else 1
        flip_sg *= flip
        flip_pg = np.array([
            [-1., 0., 0.],
            [0., -1., 0.],
            [0., 0., -1.],
        ])
        self.assertTrue(np.allclose(flip_sg, flip_pg))

    # TODO test cylinder subgroups and other subgroups

    ####### Some Direct products #######################################################################################

    def test_restrict_cylinder_c2xso2(self):
        dg = cylinder_group(3)

        # the C2 subgrousp
        sg_id = (2, 1)
        self.check_restriction(dg, sg_id)

        for N in [-1, 2, 3, 5, 8]:

            # the CN or SO(2) subgroups
            sg_id = (1, N)
            self.check_restriction(dg, sg_id)

    def test_restrict_cylinder_c2xo2(self):
        dg = full_cylinder_group(3)

        # the C2 subgrousp
        sg_id = (2, (None, 1))
        self.check_restriction(dg, sg_id)

        for N in [-1, 2, 3, 5, 8]:

            # the DN or O(2) subgroups
            sg_id = (1, (0., N))
            self.check_restriction(dg, sg_id)

            # the CN or SO(2) subgroups
            sg_id = (1, (None, N))
            self.check_restriction(dg, sg_id)

    def test_restrict_fullico(self):
        dg = full_ico_group()

        # the C2 subgrousp
        sg_id = (2, (False, 1))
        self.check_restriction(dg, sg_id)

        for n in [1, 2, 3, 5]:
            sg_id = (1, (False, n))
            self.check_restriction(dg, sg_id)

            # sg_id = (1, (True, n))
            # self.check_restriction(dg, sg_id)

    def test_restrict_fullocta(self):
        dg = full_octa_group()

        # the C2 subgrousp
        sg_id = (2, (False, 1))
        self.check_restriction(dg, sg_id)

        for n in [1, 2, 3, 4]:
            sg_id = (1, (False, n))
            self.check_restriction(dg, sg_id)

            # sg_id = (1, (True, n))
            # self.check_restriction(dg, sg_id)

    ####################################################################################################################

    def check_restriction(self, group: Group, subgroup_id):
    
        sgid = group._process_subgroup_id(subgroup_id)
        self.assertTrue(
            sgid == group._process_subgroup_id(sgid)
        )
    
        sg, parent_element, child_element = group.subgroup(subgroup_id)
    
        for e in group.testing_elements():
            c = child_element(e)
            if c is not None:
                assert c.group == sg
                assert parent_element(c) == e, f"{group} to subgroup {sg} with id {subgroup_id} | Element {e} != {parent_element(c)} | {c}"
                
        for _ in range(50):
            e = group.sample()
            c = child_element(e)
            if c is not None:
                assert c.group == sg
                assert parent_element(c) == e, f"{group} to subgroup {sg} with id {subgroup_id} | Element {e} != {parent_element(c)} | {c}"

        for e in sg.testing_elements():
            assert parent_element(e).group == group
            assert child_element(parent_element(e)) == e, f"Element {e} from subgroup {sg.name}: {parent_element(e)}, {child_element(parent_element(e))}"
            
        for _ in range(50):
            e = sg.sample()
            assert parent_element(e).group == group
            assert child_element(parent_element(e)) == e, f"Element {e} from subgroup {sg.name}: {parent_element(e)}, {child_element(parent_element(e))}"

        for a in sg.testing_elements():
            for b in sg.testing_elements():
                if parent_element(a @ b) != parent_element(a) @ parent_element(b):
                    print(f"Elements {a} and {b} from subgroup {sg.name}: product not consistent in group and subgroup")
                assert parent_element(a @ b) == parent_element(a) @ parent_element(b), f"Elements {a} and {b} from subgroup {sg.name}: product not consistent in group and subgroup"


if __name__ == '__main__':
    unittest.main()
