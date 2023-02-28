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
        irreps = G.irreps() if G.order() > 0 else [G.irrep(*irr) for irr in G.bl_irreps(5)]
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
        irreps = G.irreps() if G.order() > 0 else [G.irrep(*irr) for irr in G.bl_irreps(5)]
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
        irreps = G.irreps() if G.order() > 0 else [G.irrep(*irr) for irr in G.bl_irreps(5)]
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
        irreps = G.irreps() if G.order() > 0 else [G.irrep(*irr) for irr in G.bl_irreps(5)]
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
            irreps = G.irreps() if G.order() > 0 else [G.irrep(*irr) for irr in G.bl_irreps(5)]
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
        irreps = G.irreps() if G.order() > 0 else [G.irrep(*irr) for irr in G.bl_irreps(5)]
        o2_group(10)
        for in_rep in irreps:
            for out_rep in irreps:
                if in_rep.attributes['frequency'] < 5 and out_rep.attributes['frequency'] < 5:
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
            irreps = G.irreps() if G.order() > 0 else [G.irrep(*irr) for irr in G.bl_irreps(5)]
            for in_rep in irreps:
                for out_rep in irreps:
                    if in_rep.attributes['frequency'] < 5 and out_rep.attributes['frequency'] < 5:
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
        irreps = G.irreps() if G.order() > 0 else [G.irrep(*irr) for irr in G.bl_irreps(5)]
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
            irreps = G.irreps() if G.order() > 0 else [G.irrep(*irr) for irr in G.bl_irreps(5)]
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
        irreps = G.irreps() if G.order() > 0 else [G.irrep(*irr) for irr in G.bl_irreps(5)]
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
        irreps = G.irreps() if G.order() > 0 else [G.irrep(*irr) for irr in G.bl_irreps(5)]
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

        irreps = so2.irreps()
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
            irreps = G.irreps() if G.order() > 0 else [G.irrep(*irr) for irr in G.bl_irreps(5)]
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
            irreps = G.irreps() if G.order() > 0 else [G.irrep(*irr) for irr in G.bl_irreps(5)]
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
        
        G, inclusion, projection = _G.subgroup(sg_id)
        
        try:
            basis = RestrictedWignerEckartBasis(X, sg_id, in_rep, out_rep)
        except EmptyBasisException:
            print(f"KernelBasis between {in_rep.name} and {out_rep.name} is empty, continuing")
            return
        
        P = 10
        points = torch.randn(P, X.dimensionality, dtype=torch.float32)

        assert points.shape == (P, X.dimensionality)
        
        B = 5
        
        features = torch.randn(P, B, in_rep.size, dtype=torch.float32)
        
        filters = torch.zeros((P, basis.dim, out_rep.size, in_rep.size), dtype=torch.float32)
        
        filters = basis.sample(points, out=filters)
        
        self.assertFalse(torch.isnan(filters).any())
        self.assertFalse(torch.allclose(filters, torch.zeros_like(filters)))
        
        a = basis.sample(points)
        b = basis.sample(points)
        assert torch.allclose(a, b)
        del a, b

        output = torch.einsum("pfoi,pbi->fbo", filters, features)
        
        for _ in range(20):
            g = G.sample()
            
            output1 = torch.einsum("oi,fbi->fbo",
                                   torch.tensor(out_rep(g), dtype=output.dtype),
                                   output)

            a = X.action(inclusion(g))
            transformed_points = points @ torch.tensor(a,dtype=points.dtype).T

            transformed_filters = basis.sample(transformed_points)
            
            transformed_features = torch.einsum("oi,pbi->pbo",
                                                torch.tensor(in_rep(g), dtype=features.dtype),
                                                features)
            output2 = torch.einsum("pfoi,pbi->fbo", transformed_filters, transformed_features)

            if not torch.allclose(output1, output2, atol=1e-5, rtol=1e-4):
                print(f"{in_rep.name}, {out_rep.name}: Error at {g}")
                print(a)
                
                aerr = torch.abs(output1 - output2).detach().numpy()
                err = aerr.reshape(-1, basis.dim).max(0)
                print(basis.dim, (err > 0.01).sum())
                for idx in range(basis.dim):
                    if err[idx] > 0.1:
                        print(idx)
                        print(err[idx])
                        print(basis[idx])

            self.assertTrue(torch.allclose(output1, output2, atol=1e-5, rtol=1e-4),
                            f"Group {G.name}, {in_rep.name} - {out_rep.name},\n"
                            f"element {g},\n"
                            f"action:\n"
                            f"{a}")
                            # f"element {g}, action {a}, {basis.b1.bases[0][0].axis}")


if __name__ == '__main__':
    unittest.main()
