import unittest
from unittest import TestCase

import numpy as np

from escnn.group import *
from escnn.kernels import *


class TestSparse(TestCase):
    
    def test_ico_edges(self):
        X = Icosidodecahedron()

        G = X.G
        irreps = [G.irrep(l) for l in range(5)]
        for in_rep in irreps:
            for out_rep in irreps:
                self._check_irreps(X, in_rep, out_rep)

    def test_ico_vertices(self):
        X = Icosahedron()

        G = X.G
        irreps = [G.irrep(l) for l in range(5)]
        for in_rep in irreps:
            for out_rep in irreps:
                self._check_irreps(X, in_rep, out_rep)

    def test_ico_faces(self):
        X = Dodecahedron()

        G = X.G
        irreps = [G.irrep(l) for l in range(5)]
        for in_rep in irreps:
            for out_rep in irreps:
                self._check_irreps(X, in_rep, out_rep)

    def _check_irreps(self, X: SpaceIsomorphism, in_rep: IrreducibleRepresentation, out_rep: IrreducibleRepresentation):
        
        G = X.G
        
        try:
            basis = SparseSteerableBasis(X, in_rep, out_rep, sigma=.4)
        except EmptyBasisException:
            print(f"KernelBasis between {in_rep.name} and {out_rep.name} is empty, continuing")
            return
        
        P = 10
        points = np.concatenate(
            [X.projection(G.sample()) for _ in range(P)],
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

            transformed_points = X.action(g) @ points

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
