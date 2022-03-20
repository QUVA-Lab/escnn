import unittest
from unittest import TestCase

from escnn.group import *
import numpy as np


def testing_elements(group: Group):
    te = list(group.testing_elements())
    if len(te) < 16:
        return te
    else:
        return [group.sample() for _ in range(15)]
    

class TestDirectproduct(TestCase):
    
    def test_trivial(self):
        H = trivial_group()
        G = double_group(H)
        self.check_group(G, H, H)
        self.check_irreps(G)
        self.check_regular_repr(G)
        
    def test_cn(self):
        for n in [1, 2, 3, 5]:
            G1 = cyclic_group(n)
            G = double_group(G1)
            self.check_group(G, G1, G1)
            self.check_irreps(G)
            self.check_regular_repr(G)

    def test_dn(self):
        for n in [1, 2, 3, 4]:
            G1 = dihedral_group(n)
            G = double_group(G1)
            self.check_group(G, G1, G1)
            self.check_irreps(G)
            self.check_regular_repr(G)

    def test_so2(self):
        G1 = so2_group(2)
        G = double_group(G1)
        self.check_group(G, G1, G1)
        self.check_irreps(G)
        self.check_regular_repr(G)

    def test_o2(self):
        G1 = o2_group(2)
        G = double_group(G1)
        self.check_group(G, G1, G1)
        self.check_irreps(G)
        self.check_regular_repr(G)

    def test_so3(self):
        G1 = so3_group(2)
        G = double_group(G1)
        self.check_group(G, G1, G1)
        self.check_irreps(G)
        self.check_regular_repr(G)

    # Tests

    def check_group(self, group: DirectProductGroup, G1: Group, G2: Group):
        
        assert group.G1 == G1
        assert group.G2 == G2
        
        # Check basic group properties
        
        self.check_generators(group)
    
        e = group.identity

        for _ in range(150):
            a = group.sample()
            b = group.sample()
            c = group.sample()

            self.assertTrue(a @ e == a)
            self.assertTrue(e @ a == a)
        
            i = ~a
            self.assertTrue(a @ i, e)
            self.assertTrue(i @ a, e)

            ab = a @ b
            bc = b @ c
            a_bc = a @ bc
            ab_c = ab @ c
            self.assertTrue(a_bc == ab_c, f"{a_bc} != {ab_c}")

        # Check direct product properties
        
        G1 = group.G1
        G2 = group.G2

        for _ in range(100):
            a = G1.sample()
            b = G2.sample()
            a = group.inclusion1(a)
            b = group.inclusion2(b)

            ab = a @ b
            ba = b @ a

            self.assertTrue(ab == ba)

        for _ in range(100):
            a = G1.sample()
            C = group.sample()

            AC = group.inclusion1(a) @ C
            c1, c2 = group.split_element(C)

            ac = group.pair_elements(a @ c1, c2)

            self.assertTrue(ac == AC)

        for _ in range(100):
            b = G2.sample()
            C = group.sample()
            BC = group.inclusion2(b) @ C
            c1, c2 = group.split_element(C)

            bc = group.pair_elements(c1, b@c2)

            self.assertTrue(bc == BC)

    def check_generators(self, group: Group):
        if group.order() > 0:
            generators = group.generators
            if group.order() > 1:
                self.assertTrue(len(generators) > 0)
        else:
            with self.assertRaises(ValueError):
                generators = group.generators
            return
    
        identity = group.identity
    
        added = set()
        elements = set()
    
        added.add(identity)
        elements.add(identity)
    
        while len(added) > 0:
            new = set()
            for g in generators:
                for e in added:
                    new |= {g @ e, ~g @ e}
            added = new - elements
            elements |= added
    
        self.assertTrue(
            len(elements) == group.order(),
            'Error! The set of generators does not generate the whole group'
        )
    
        for a in elements:
            self.assertIn(~a, elements)
            for b in elements:
                self.assertIn(a @ b, elements)
    
    def check_regular_repr(self, group: Group):
        if group.order() > 0:
            reg = group.regular_representation
            self.check_representation(reg)
    
    def check_irreps(self, group: Group):
        for irrep in group.irreps():
            self.check_irrep_endom(irrep)
            self.check_representation(irrep)
            self.check_character(irrep)
            
    def check_irrep_endom(self, irrep: IrreducibleRepresentation):
        group = irrep.group
    
        np.set_printoptions(precision=2, threshold=2 * irrep.size ** 2, suppress=True,
                            linewidth=10 * irrep.size + 3)

        self.assertTrue(irrep.sum_of_squares_constituents == irrep.endomorphism_basis().shape[0])

        for k in range(irrep.sum_of_squares_constituents):
            E_k = irrep.endomorphism_basis()[k, ...]
    
            # check orthogonality
            self.assertTrue(np.allclose(E_k @ E_k.T, np.eye(irrep.size)))
            self.assertTrue(np.allclose(E_k.T @ E_k, np.eye(irrep.size)))
    
            if k == 0:
                # if k=0, the matrix need to be the identity
                self.assertTrue(np.allclose(E_k, np.eye(irrep.size)))
            else:
                # all other ones need to be skew symmetric
                self.assertTrue(np.allclose(E_k, -E_k.T))
    
            for l in range(irrep.sum_of_squares_constituents):
                E_l = irrep.endomorphism_basis()[l, ...]
        
                d_kl = (E_l * E_k).sum() / irrep.size
        
                if k == l:
                    self.assertTrue(np.allclose(d_kl, 1.))
                else:
                    self.assertTrue(np.allclose(d_kl, 0.))
            
                    for i in range(irrep.size):
                        self.assertTrue(np.allclose(
                            E_k[:, i].T @ E_l[:, i], 0.
                        ))

            w = (irrep.endomorphism_basis() ** 2).sum(axis=0) > 1e-9
            self.assertTrue(np.allclose(w.sum(axis=1), irrep.sum_of_squares_constituents))

        end_basis = irrep.endomorphism_basis()

        for a in testing_elements(group):
            r = irrep(a)
    
            self.assertTrue(np.allclose(r @ end_basis, end_basis @ r),
                            msg=f"{group.name} | {irrep.name} | {a}: endomorphism basis not equivariant\n\n")

    def check_representation(self, repr: Representation):
    
        group = repr.group
        
        np.set_printoptions(precision=2, threshold=2 * repr.size ** 2, suppress=True,
                            linewidth=10 * repr.size + 3)
        
        P = directsum([group.irrep(*irr) for irr in repr.irreps], name="irreps")
        
        self.assertTrue(np.allclose(repr.change_of_basis @ repr.change_of_basis.T, np.eye(repr.size)))
        self.assertTrue(np.allclose(repr.change_of_basis.T @ repr.change_of_basis, np.eye(repr.size)))
        
        for _ in range(20):
            a = group.sample()

            repr_1 = repr(a)
            repr_2 = repr.change_of_basis @ P(a) @ repr.change_of_basis_inv

            self.assertTrue(np.allclose(repr_1, repr_2),
                            msg=f"{a}:\n{repr_1}\ndifferent from\n {repr_2}\n")
            
            repr_a_inv = repr(~a)
            repr_inv_a = np.linalg.inv(repr_1)
            self.assertTrue(np.allclose(repr_a_inv, repr_inv_a),
                            msg=f"{a}:\n{repr_a_inv}\ndifferent from\n {repr_inv_a}\n")

            for _ in range(20):
                b = group.sample()
                repr_ab = repr(a) @ repr(b)
                c = a @ b
                repr_c = repr(c)
                
                self.assertTrue(np.allclose(repr_ab, repr_c), msg=f"{a} x {b} = {c}:\n{repr_ab}\ndifferent from\n {repr_c}\n")

    def check_character(self, repr: Representation):
    
        group = repr.group
    
        np.set_printoptions(precision=2, threshold=2 * repr.size ** 2, suppress=True,
                            linewidth=10 * repr.size + 3)
    
        # for a in group.testing_elements():
        for a in testing_elements(group):
            char_a_1 = repr.character(a)
            char_a_2 = np.trace(repr(a))
        
            self.assertAlmostEqual(char_a_1, char_a_2,
                                   msg=f"""{a}: Character of {repr} different from its trace \n {char_a_1} != {char_a_2} \n""")
        

if __name__ == '__main__':
    unittest.main()
