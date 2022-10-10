import unittest
from unittest import TestCase

from escnn.gspaces import *
from escnn.group import *
from escnn.kernels import *
import numpy as np

import torch


def testing_elements(group: Group):
    te = list(group.testing_elements())
    if len(te) < 16:
        return te
    else:
        return [group.sample() for _ in range(15)]


class TestRestrictGSpace(TestCase):
    
    ####################################################################################################################
    ###                         ########################################################################################
    ###     2D   isometries     ########################################################################################
    ###                         ########################################################################################
    ####################################################################################################################

    def test_2d_rotations(self):
        
        space = rot2dOnR2(-1, maximum_frequency=5)
        self.check_equivariance(space, maximum_frequency=3)
        
        for n in [1, 2, 3, 4, 5]:
            space = rot2dOnR2(n)
            self.check_equivariance(space, maximum_frequency=3)

    def test_2d_flipsrotations(self):
    
        for n in [-1, 1, 2, 3, 4, 5]:
            for axis in range(4):
                axis = axis * np.pi / 4
                space = flipRot2dOnR2(n, axis=axis, maximum_frequency=5)
                self.check_equivariance(space, maximum_frequency=3)

    def test_2d_flips(self):
    
        for axis in range(8):
            axis = axis * np.pi / 8
            space = flip2dOnR2(axis)
            self.check_equivariance(space, maximum_frequency=3)

    ####################################################################################################################
    ###                         ########################################################################################
    ###     3D   isometries     ########################################################################################
    ###                         ########################################################################################
    ####################################################################################################################

    ####### SO(3) ######################################################################################################
    
    def test_3d_rotations(self):
        space = rot3dOnR3(maximum_frequency=2)
        self.check_equivariance(space, maximum_frequency=3)

    def test_ico(self):
        space = icoOnR3()
        self.check_equivariance(space, maximum_frequency=3)

    def test_octa(self):
        space = octaOnR3()
        self.check_equivariance(space, maximum_frequency=3)

    def test_3d_2d_rotations(self):
        for n in [1, 2, 4, 7]:
            space = rot2dOnR3(n, maximum_frequency=8)
            self.check_equivariance(space, maximum_frequency=3)

    def test_3d_2d_rotations_adjoint(self):
        o3 = o3_group(1)
        for adj in o3.grid('rand', 5):
            space = rot2dOnR3(-1, adjoint=adj, maximum_frequency=1)
            self.check_equivariance(space, maximum_frequency=3)
            
            # check that the adjoint is changing the rotation axis as expected
            axis = o3.standard_representation()(~adj) @ np.asarray([0, 0, 1.]).T
            for i in range(8):
                g = space.fibergroup.sample()
                self.assertTrue(np.allclose(space.basespace_action(g) @ axis, axis))

    def test_3d_2d_rotations_reflections(self):
    
        for n in [-1, 2, 3, 4]:
            for axis in [0., np.pi/2, np.pi/4]:
                space = dihedralOnR3(n, axis, maximum_frequency=8)
                self.check_equivariance(space, maximum_frequency=3)

    def test_3d_2d_flips(self):
    
        for axis in [0., np.pi / 2, np.pi / 4]:
            space = dihedralOnR3(1, axis, maximum_frequency=8)
            self.check_equivariance(space, maximum_frequency=3)

    ####### O(3) #######################################################################################################

    def test_3d_rotationinversion(self):
        space = flipRot3dOnR3(maximum_frequency=2)
        self.check_equivariance(space, maximum_frequency=3)
    
    def test_3d_full_ico(self):
        space = fullIcoOnR3()
        self.check_equivariance(space, maximum_frequency=3)

    def test_3d_full_octa(self):
        space = fullOctaOnR3()
        self.check_equivariance(space, maximum_frequency=3)

    def test_3d_cone(self):
    
        o3 = o3_group()
        xyz = o3.element(
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
        for n in [-1, 2, 4]:
            # Cone aligned along Z axis
            # i.e., rotation along Z axis
            # on XY plane, mirror wrt Y axis (i.e. flip along X axis)
            space = conicalOnR3(n, 0., maximum_frequency=8)
            self.check_equivariance(space, maximum_frequency=3)

            # Cone aligned along Z axis
            # i.e., rotation along Z axis
            # on XY plane, mirror wrt X axis (i.e. flip along Y axis)
            space = conicalOnR3(n, np.pi/2, maximum_frequency=8)
            self.check_equivariance(space, maximum_frequency=3)

            for adj in [xyz, xyz@xyz, xyz @ o3.inversion, xyz @ xyz @ o3.inversion]:
                space = conicalOnR3(n, 0., maximum_frequency=2, adjoint=adj)
                self.check_equivariance(space, maximum_frequency=3)

    def test_3d_mir(self):
    
        # mirror wrt Y axis (i.e. flip along X axis)
        space = mirOnR3(axis=0.)
        self.check_equivariance(space, maximum_frequency=3)
        
        # mirror wrt X axis (i.e. flip along Y axis)
        space = mirOnR3(axis=np.pi/2)
        self.check_equivariance(space, maximum_frequency=3)

        o3 = o3_group()
        xyz = o3.element(
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

        for adj in [xyz, xyz@xyz, xyz @ o3.inversion, xyz @ xyz @ o3.inversion]:
            space = mirOnR3(0., adjoint=adj)
            self.check_equivariance(space, maximum_frequency=3)

    def test_3d_fullcyl(self):

        o3 = o3_group()
        xyz = o3.element(
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
        for n in [-1, 2, 4]:
            # Cylinder aligned along Z axis
            # i.e., rotation along Z axis
            space = fullCylindricalOnR3(n, maximum_frequency=4)
            self.check_equivariance(space, maximum_frequency=3)

            for adj in [xyz, xyz @ xyz, xyz @ o3.inversion, xyz @ xyz @ o3.inversion]:
                space = fullCylindricalOnR3(n, maximum_frequency=6, adjoint=adj)
                self.check_equivariance(space, maximum_frequency=3)

    def test_3d_cyl(self):

        o3 = o3_group()
        xyz = o3.element(
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
        for n in [-1, 2, 4]:
            # Cylinder aligned along Z axis
            # i.e., rotation along Z axis
            space = cylindricalOnR3(n, maximum_frequency=4)
            self.check_equivariance(space, maximum_frequency=3)

            for adj in [xyz, xyz @ xyz, xyz @ o3.inversion, xyz @ xyz @ o3.inversion]:
                space = cylindricalOnR3(n, maximum_frequency=6, adjoint=adj)
                self.check_equivariance(space, maximum_frequency=3)

    def test_3d_inv(self):
    
        space = invOnR3()
        self.check_equivariance(space, maximum_frequency=3)

    ####################################################################################################################
    def check_equivariance(self, space: GSpace, **kwargs):
        
        base_action = space.basespace_action
        irreps = list(space.fibergroup.irreps())
        
        S = 3
        xs = np.arange(S) - (S // 2)
        if space.dimensionality > 0:
            grid = np.stack(np.meshgrid(*[xs]*space.dimensionality))
            grid = grid.reshape(space.dimensionality, -1)
        else:
            grid = np.zeros((0,))

        grid = torch.tensor(grid, dtype=torch.float32).T
        
        for psi_in in irreps:
            x = torch.randn(S**space.dimensionality, 1, psi_in.size)
            
            for psi_out in irreps:
                
                try:
                    basis = space.build_kernel_basis(psi_in, psi_out,
                                                     rings=np.arange(int(np.ceil(S/2))).tolist(),
                                                     sigma=.5,
                                                     **kwargs)
                except InsufficientIrrepsException as e:
                    print(e)
                    continue
                except EmptyBasisException as e:
                    print(f"{space}: {psi_in} -> {psi_out} : empty basis!")
                    continue

                sampled_basis = basis.sample(grid)
                y = torch.einsum('pkoi,pbi->kbo', sampled_basis, x)

                for _ in range(20):
                    g = space.fibergroup.sample()

                    a = torch.tensor(base_action(g), dtype=torch.float32)
                    g_sampled_basis = basis.sample(grid @ a.T)

                    _psi_out = torch.tensor(psi_out(g), dtype=torch.float32)
                    _psi_in = torch.tensor(psi_in(g), dtype=torch.float32)
                    gy = torch.einsum('oi,kbi->kbo', _psi_out, y)
                    yg = torch.einsum('pkoi,il,pbl->kbo', g_sampled_basis, _psi_in, x)

                    check = torch.allclose(gy, yg, atol=5e-5, rtol=1e-3)

                    if not check:
                        aerr = torch.abs(gy - yg).cpu().detach().numpy()
                        err = aerr.reshape(-1, basis.dim).max(0)
                        print(basis.dim, (err > 0.01).sum(), err.max())
                        print(torch.isclose(gy, yg, atol=5e-5, rtol=1e-3).logical_not().to(int).sum().item())
                        print(err.max(), err.mean(), err.std())
                        for idx in range(basis.dim):
                            if err[idx] > 0.1:
                                print(idx)
                                print(err[idx])
                                print(basis[idx])

                    self.assertTrue(check, msg=f"{space.name}: {g}")
        

if __name__ == '__main__':
    unittest.main()
