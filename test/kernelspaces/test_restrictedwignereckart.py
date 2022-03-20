import unittest
from unittest import TestCase

import numpy as np

from escnn.group import *
from escnn.kernels import *

from typing import Tuple, List


class TestWEbasis(TestCase):
    
    ####################################################################################################################
    # 3D
    ####################################################################################################################

    def test_sphere_ico(self):
        so3 = so3_group(6)
        sg_id = 'ico'
        harmonics = [(l,) for l in range(8)]
        X = SphereSO3()

        G, _, _ = so3.subgroup(sg_id)
        irreps = G.irreps()
        for in_rep in irreps:
            for out_rep in irreps:
                self._check_irreps(X, in_rep, out_rep, sg_id=sg_id, harmonics=harmonics)

    def test_sphere_octa(self):
        so3 = so3_group(6)
        sg_id = 'octa'
        harmonics = [(l,) for l in range(8)]
        X = SphereSO3()

        G, _, _ = so3.subgroup(sg_id)
        irreps = G.irreps()
        for in_rep in irreps:
            for out_rep in irreps:
                self._check_irreps(X, in_rep, out_rep, sg_id=sg_id, harmonics=harmonics)

    def test_sphere_so2(self):
        so3 = so3_group(3)
        sg_id = (False, -1)
        harmonics = [(l,) for l in range(8)]
        X = SphereSO3()
    
        G, _, _ = so3.subgroup(sg_id)
        irreps = G.irreps()
        so2_group(10)
        for in_rep in irreps:
            for out_rep in irreps:
                self._check_irreps(X, in_rep, out_rep, sg_id=sg_id, harmonics=harmonics)

    def test_sphere_cn(self):
        so3 = so3_group(3)
        harmonics = [(l,) for l in range(8)]
        X = SphereSO3()

        for n in [2, 3, 4, 7, 12]:
            sg_id = (False, n)
            G, _, _ = so3.subgroup(sg_id)
            irreps = G.irreps()
            for in_rep in irreps:
                for out_rep in irreps:
                    self._check_irreps(X, in_rep, out_rep, sg_id=sg_id, harmonics=harmonics)

    def test_sphere_o2_dihedral(self):
        so3 = so3_group(3)
        harmonics = [(l,) for l in range(8)]
        X = SphereSO3()
        sg_id = (True, -1)
        
        G, _, _ = so3.subgroup(sg_id)
        irreps = G.irreps()
        o2_group(10)
        for in_rep in irreps:
            for out_rep in irreps:
                self._check_irreps(X, in_rep, out_rep, sg_id=sg_id, harmonics=harmonics)

    def test_sphere_dn_dihedral(self):
        so3 = so3_group(3)
        harmonics = [(l,) for l in range(8)]
        X = SphereSO3()
        
        for n in [2, 3, 4, 7]:
            sg_id = (True, n)
        
            G, _, _ = so3.subgroup(sg_id)
            irreps = G.irreps()
            for in_rep in irreps:
                for out_rep in irreps:
                    self._check_irreps(X, in_rep, out_rep, sg_id=sg_id, harmonics=harmonics)

    def test_sphere_o2_conical(self):
        o3 = o3_group(3)
        harmonics = [(k, l) for l in range(8) for k in range(2)]
        X = SphereO3()
        sg_id = ('cone', -1)
    
        G, _, _ = o3.subgroup(sg_id)
        irreps = G.irreps()
        o2_group(10)
        for in_rep in irreps:
            for out_rep in irreps:
                self._check_irreps(X, in_rep, out_rep, sg_id=sg_id, harmonics=harmonics)

    def test_sphere_dn_conical(self):
        o3 = o3_group(3)
        harmonics = [(k, l) for l in range(8) for k in range(2)]
        X = SphereO3()
    
        for n in [2, 3, 4, 7]:
            sg_id = ('cone', n)
    
            G, _, _ = o3.subgroup(sg_id)
            irreps = G.irreps()
            for in_rep in irreps:
                for out_rep in irreps:
                    self._check_irreps(X, in_rep, out_rep, sg_id=sg_id, harmonics=harmonics)

    def test_sphere_c2xso2_cyl(self):
        o3 = o3_group(3)
        harmonics = [(k, l) for l in range(8) for k in range(2)]
        X = SphereO3()
        sg_id = (True, False, -1)

        G, _, _ = o3.subgroup(sg_id)
        irreps = G.irreps()
        cylinder_group(7)
        for in_rep in irreps:
            for out_rep in irreps:
                self._check_irreps(X, in_rep, out_rep, sg_id=sg_id, harmonics=harmonics)

    def test_sphere_c2xo2_fullcyl(self):
        o3 = o3_group(3)
        harmonics = [(k, l) for l in range(8) for k in range(2)]
        X = SphereO3()
        sg_id = (True, True, -1)

        G, _, _ = o3.subgroup(sg_id)
        irreps = G.irreps()
        full_cylinder_group(10)
        for in_rep in irreps:
            for out_rep in irreps:
                self._check_irreps(X, in_rep, out_rep, sg_id=sg_id, harmonics=harmonics)

    def test_sphere_inv(self):
        o3 = o3_group(3)
        harmonics = [(k, l) for l in range(8) for k in range(2)]
        X = SphereO3()

        sg_id = (True, False, 1)

        G, _, _ = o3.subgroup(sg_id)
        irreps = G.irreps()
        for in_rep in irreps:
            for out_rep in irreps:
                self._check_irreps(X, in_rep, out_rep, sg_id=sg_id, harmonics=harmonics)
                
    def test_sphere_trivial(self):
        o3 = o3_group(3)
        harmonics = [(k, l) for l in range(8) for k in range(2)]
        X = SphereO3()
    
        sg_id = o3.subgroup_trivial_id
    
        G, _, _ = o3.subgroup(sg_id)
        irreps = G.irreps()
        for in_rep in irreps:
            for out_rep in irreps:
                self._check_irreps(X, in_rep, out_rep, sg_id=sg_id, harmonics=harmonics)
                
        so3 = so3_group(3)
        harmonics = [(l,) for l in range(3)]
        X = SphereSO3()

        sg_id = so3.subgroup_trivial_id

        G, _, _ = so3.subgroup(sg_id)
        irreps = G.irreps()
        for in_rep in irreps:
            for out_rep in irreps:
                self._check_irreps(X, in_rep, out_rep, sg_id=sg_id, harmonics=harmonics)

    ####################################################################################################################
    # 2D
    ####################################################################################################################

    def test_circle_cn(self):
        so2 = so2_group(3)
        harmonics = [(l,) for l in range(8)]
        X = CircleSO2()
    
        for n in [2, 3, 4, 7, 12]:
            sg_id = n
            G, _, _ = so2.subgroup(sg_id)
            irreps = G.irreps()
            for in_rep in irreps:
                for out_rep in irreps:
                    self._check_irreps(X, in_rep, out_rep, sg_id=sg_id, harmonics=harmonics)

    def test_circle_dn(self):
        o2 = o2_group(3)
        harmonics = [(0,0)] + [(1, l) for l in range(8)]
        X = CircleO2()
    
        for n in [2, 3, 4, 7, 12]:
            sg_id = (0., n)
            G, _, _ = o2.subgroup(sg_id)
            irreps = G.irreps()
            for in_rep in irreps:
                for out_rep in irreps:
                    self._check_irreps(X, in_rep, out_rep, sg_id=sg_id, harmonics=harmonics)

    def _check_irreps(self,
                      X: SpaceIsomorphism,
                      in_rep: IrreducibleRepresentation,
                      out_rep: IrreducibleRepresentation,
                      sg_id: Tuple,
                      harmonics : List[Tuple]
                      ):
        
        _G = X.G
        
        G, inclusion, projection = _G.subgroup(sg_id)
        
        try:
            basis = RestrictedWignerEckartBasis(X, sg_id, in_rep, out_rep, harmonics)
        except EmptyBasisException:
            print(f"KernelBasis between {in_rep.name} and {out_rep.name} is empty, continuing")
            return
        
        P = 10
        points = np.concatenate(
            [X.projection(_G.sample()) for _ in range(P)],
            axis=1
        )
        
        assert points.shape == (X.dim, P)
        
        B = 5
        
        features = np.random.randn(B, in_rep.size, P)
        
        filters = np.zeros((out_rep.size, in_rep.size, basis.dim, P), dtype=np.float)
        
        filters = basis.sample(points, out=filters)
        
        self.assertFalse(np.isnan(filters).any())
        self.assertFalse(np.allclose(filters, np.zeros_like(filters)))
        
        a = basis.sample(points)
        b = basis.sample(points)
        assert np.allclose(a, b)

        output = np.einsum("oifp,bip->bof", filters, features)
        
        for g in G.testing_elements():
            
            output1 = np.einsum("oi,bif->bof", out_rep(g), output)

            transformed_points = X.action(inclusion(g)) @ points

            transformed_filters = basis.sample(transformed_points)
            
            transformed_features = np.einsum("oi,bip->bop", in_rep(g), features)
            output2 = np.einsum("oifp,bip->bof", transformed_filters, transformed_features)

            if not np.allclose(output1, output2):
                print(f"{in_rep.name}, {out_rep.name}: Error at {g}")
                print(a)
                
                aerr = np.abs(output1 - output2)
                err = aerr.reshape(-1, basis.dim).max(0)
                print(basis.dim, (err > 0.01).sum())
                for idx in range(basis.dim):
                    if err[idx] > 0.1:
                        print(idx)
                        print(err[idx])
                        print(basis[idx])

            self.assertTrue(np.allclose(output1, output2), f"Group {G.name}, {in_rep.name} - {out_rep.name},\n"
                                                           f"element {g},\n"
                                                           f"action:\n"
                                                           f"{a}")
                                                           # f"element {g}, action {a}, {basis.b1.bases[0][0].axis}")


if __name__ == '__main__':
    unittest.main()
