import unittest
from unittest import TestCase

import numpy as np

from escnn.group import *
from escnn.kernels import *

from typing import Tuple, List

import torch


class TestWEbasis(TestCase):
    
    ####################################################################################################################
    # 3D
    ####################################################################################################################

    def test_sphere_so3(self):
        o3 = o3_group(6)
        sg_id = 'so3'
        harmonics = [(l,) for l in range(8)]
        X = SphericalShellsBasis(
            L=4,
            radial=GaussianRadialProfile([0., 1., 2.], [0.6, 0.6, 0.6]),
        )

        G, _, _ = o3.subgroup(sg_id)
        irreps = [G.irrep(*irr) for irr in G.bl_irreps(3)]
        for in_rep in irreps:
            for out_rep in irreps:
                self._check_irreps(X, in_rep, out_rep, sg_id=sg_id)

    def test_sphere_ico(self):
        o3 = o3_group(6)
        sg_id = o3._combine_subgroups('so3', 'ico')
        harmonics = [(l,) for l in range(8)]
        X = SphericalShellsBasis(
            L=4,
            radial=GaussianRadialProfile([0., 1., 2.], [0.6, 0.6, 0.6]),
        )

        G, _, _ = o3.subgroup(sg_id)
        irreps = G.irreps()
        for in_rep in irreps:
            for out_rep in irreps:
                self._check_irreps(X, in_rep, out_rep, sg_id=sg_id)

    def test_sphere_octa(self):
        o3 = o3_group(6)
        sg_id = o3._combine_subgroups('so3', 'octa')
        harmonics = [(l,) for l in range(8)]
        X = SphericalShellsBasis(
            L=4,
            radial=GaussianRadialProfile([0., 1., 2.], [0.6, 0.6, 0.6]),
        )

        G, _, _ = o3.subgroup(sg_id)
        irreps = G.irreps()
        for in_rep in irreps:
            for out_rep in irreps:
                self._check_irreps(X, in_rep, out_rep, sg_id=sg_id)

    def test_sphere_so2(self):
        o3 = o3_group(3)
        sg_id = o3._combine_subgroups('so3', (False, -1))
        harmonics = [(l,) for l in range(8)]
        X = SphericalShellsBasis(
            L=4,
            radial=GaussianRadialProfile([0., 1., 2.], [0.6, 0.6, 0.6]),
        )

        G, _, _ = o3.subgroup(sg_id)
        irreps = [G.irrep(*irr) for irr in G.bl_irreps(3)]
        so2_group(10)
        for in_rep in irreps:
            for out_rep in irreps:
                self._check_irreps(X, in_rep, out_rep, sg_id=sg_id)

    def test_sphere_cn(self):
        o3 = o3_group(3)
        harmonics = [(l,) for l in range(8)]
        X = SphericalShellsBasis(
            L=4,
            radial=GaussianRadialProfile([0., 1., 2.], [0.6, 0.6, 0.6]),
        )

        for n in [2, 3, 4, 7, 12]:
            sg_id = o3._combine_subgroups('so3', (False, n))
            G, _, _ = o3.subgroup(sg_id)
            irreps = G.irreps()
            for in_rep in irreps:
                for out_rep in irreps:
                    self._check_irreps(X, in_rep, out_rep, sg_id=sg_id)

    def test_sphere_o2_dihedral(self):
        o3 = o3_group(3)
        harmonics = [(l,) for l in range(8)]
        X = SphericalShellsBasis(
            L=4,
            radial=GaussianRadialProfile([0., 1., 2.], [0.6, 0.6, 0.6]),
        )
        sg_id = o3._combine_subgroups('so3', (True, -1))
        
        G, _, _ = o3.subgroup(sg_id)
        irreps = [G.irrep(*irr) for irr in G.bl_irreps(3)]
        o2_group(10)
        for in_rep in irreps:
            for out_rep in irreps:
                self._check_irreps(X, in_rep, out_rep, sg_id=sg_id)

    def test_sphere_dn_dihedral(self):
        o3 = o3_group(3)
        harmonics = [(l,) for l in range(8)]
        X = SphericalShellsBasis(
            L=4,
            radial=GaussianRadialProfile([0., 1., 2.], [0.6, 0.6, 0.6]),
        )

        for n in [2, 3, 4, 7]:
            sg_id = o3._combine_subgroups('so3', (True, n))
        
            G, _, _ = o3.subgroup(sg_id)
            irreps = G.irreps()
            for in_rep in irreps:
                for out_rep in irreps:
                    self._check_irreps(X, in_rep, out_rep, sg_id=sg_id)

    def test_sphere_o2_conical(self):
        o3 = o3_group(3)
        harmonics = [(k, l) for l in range(3) for k in range(2)]

        X = SphericalShellsBasis(
            L=4,
            radial=GaussianRadialProfile([0., 1., 2.], [0.6, 0.6, 0.6]),
        )

        sg_id = ('cone', -1)
    
        G, _, _ = o3.subgroup(sg_id)
        irreps = [G.irrep(*irr) for irr in G.bl_irreps(3)]
        o2_group(12)
        for in_rep in irreps:
            for out_rep in irreps:
                self._check_irreps(X, in_rep, out_rep, sg_id=sg_id)

    def test_sphere_dn_conical(self):
        o3 = o3_group(3)
        harmonics = [(k, l) for l in range(8) for k in range(2)]

        X = SphericalShellsBasis(
            L=4,
            radial=GaussianRadialProfile([0., 1., 2.], [0.6, 0.6, 0.6]),
        )

    
        for n in [2, 3, 4, 7]:
            sg_id = ('cone', n)
    
            G, _, _ = o3.subgroup(sg_id)
            irreps = G.irreps()
            for in_rep in irreps:
                for out_rep in irreps:
                    self._check_irreps(X, in_rep, out_rep, sg_id=sg_id)

    def test_sphere_c2xso2_cyl(self):
        o3 = o3_group(3)
        harmonics = [(k, l) for l in range(3) for k in range(2)]

        X = SphericalShellsBasis(
            L=4,
            radial=GaussianRadialProfile([0., 1., 2.], [0.6, 0.6, 0.6]),
        )

        sg_id = (True, False, -1)

        G, _, _ = o3.subgroup(sg_id)
        irreps = [G.irrep((k,), (l,)) for k in range(2) for l in range(4)]
        cylinder_group(12)
        for in_rep in irreps:
            for out_rep in irreps:
                self._check_irreps(X, in_rep, out_rep, sg_id=sg_id)

    def test_sphere_c2xo2_fullcyl(self):
        o3 = o3_group(3)
        harmonics = [(k, l) for l in range(8) for k in range(2)]

        X = SphericalShellsBasis(
            L=4,
            radial=GaussianRadialProfile([0., 1., 2.], [0.6, 0.6, 0.6]),
        )

        sg_id = (True, True, -1)

        G, _, _ = o3.subgroup(sg_id)
        irreps = [G.irrep((k,), (1, l)) for k in range(2) for l in range(4)] + [G.irrep((k,), (0, 0)) for k in range(2)]
        full_cylinder_group(10)
        for in_rep in irreps:
            for out_rep in irreps:
                self._check_irreps(X, in_rep, out_rep, sg_id=sg_id)

    def test_sphere_inv(self):
        o3 = o3_group(3)
        harmonics = [(k, l) for l in range(8) for k in range(2)]

        X = SphericalShellsBasis(
            L=4,
            radial=GaussianRadialProfile([0., 1., 2.], [0.6, 0.6, 0.6]),
        )


        sg_id = (True, False, 1)

        G, _, _ = o3.subgroup(sg_id)
        irreps = G.irreps()
        for in_rep in irreps:
            for out_rep in irreps:
                self._check_irreps(X, in_rep, out_rep, sg_id=sg_id)
                
    def test_sphere_trivial(self):
        o3 = o3_group(3)
        harmonics = [(k, l) for l in range(8) for k in range(2)]

        X = SphericalShellsBasis(
            L=4,
            radial=GaussianRadialProfile([0., 1., 2.], [0.6, 0.6, 0.6]),
        )

    
        sg_id = o3.subgroup_trivial_id
    
        G, _, _ = o3.subgroup(sg_id)
        irreps = G.irreps()
        for in_rep in irreps:
            for out_rep in irreps:
                self._check_irreps(X, in_rep, out_rep, sg_id=sg_id)
                
    ####################################################################################################################
    # 2D
    ####################################################################################################################

    def test_circle_so2(self):
        so2 = so2_group(3)
        harmonics = [(l,) for l in range(8)]

        X = CircularShellsBasis(
            L=4,
            radial=GaussianRadialProfile([0., 1., 2.], [0.6, 0.6, 0.6]),
        )

        irreps = [so2.irrep(*irr) for irr in so2.bl_irreps(5)]
        sg_id = (None, -1)
        for in_rep in irreps:
            for out_rep in irreps:
                self._check_irreps(X, in_rep, out_rep, sg_id=sg_id)

    def test_circle_cn(self):
        o2 = o2_group(3)
        harmonics = [(l,) for l in range(8)]

        X = CircularShellsBasis(
            L=4,
            radial=GaussianRadialProfile([0., 1., 2.], [0.6, 0.6, 0.6]),
        )

        for n in [2, 3, 4, 7, 12]:
            sg_id = o2._combine_subgroups((None, -1), n)
            G, _, _ = o2.subgroup(sg_id)
            irreps = G.irreps()
            for in_rep in irreps:
                for out_rep in irreps:
                    self._check_irreps(X, in_rep, out_rep, sg_id=sg_id)

    def test_circle_dn(self):
        o2 = o2_group(3)
        harmonics = [(0,0)] + [(1, l) for l in range(8)]
        X = CircularShellsBasis(
            L=4,
            radial=GaussianRadialProfile([0., 1., 2.], [0.6, 0.6, 0.6]),
        )

        for n in [2, 3, 4, 7, 12]:
            sg_id = (0., n)
            G, _, _ = o2.subgroup(sg_id)
            irreps = G.irreps()
            for in_rep in irreps:
                for out_rep in irreps:
                    self._check_irreps(X, in_rep, out_rep, sg_id=sg_id)

    def _check_irreps(self,
                      X: SteerableFiltersBasis,
                      in_rep: IrreducibleRepresentation,
                      out_rep: IrreducibleRepresentation,
                      sg_id: Tuple,
                      ):

        _G = X.group

        try:
            basis = RestrictedWignerEckartBasis(X, sg_id, in_rep, out_rep)
        except EmptyBasisException:
            print(f"KernelBasis between {in_rep.name} and {out_rep.name} is empty, continuing")
            return

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        P = 20
        points = torch.randn(P, X.dimensionality, device=device)
        assert points.shape == (P, X.dimensionality)
        basis = basis.to(device)

        points.requires_grad_(True)

        from torch.optim import Adam

        optimizer = Adam([points], lr=1e-2)

        for i in range(10):
            optimizer.zero_grad()

            assert not torch.isnan(points).any(), i

            filters = basis.sample(points)

            loss = (filters ** 2 - .3 * filters + filters.abs()).mean()

            loss.backward()

            assert not torch.isnan(points.grad).any(), i
            grad = points.grad.abs()
            assert (grad > 0.).all(), (i, (grad > 0.).sum().item(), grad.mean().item(), grad.std().item(), grad.max().item(), grad.min().item())

            optimizer.step()


if __name__ == '__main__':
    unittest.main()
