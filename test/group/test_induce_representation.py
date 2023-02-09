import unittest
from unittest import TestCase

from escnn.group import *

from collections import defaultdict
from typing import Tuple, List

import math
import numpy as np


class TestInducedRepresentations(TestCase):
    
    def test_quotient_cyclic_even(self):
        N = 20
        dg = cyclic_group(N)
        for n in range(1, int(round(np.sqrt(N)))+1):
            if N % n == 0:
                sg_id = n
                sg, _, _ = dg.subgroup(sg_id)
                self.check_induction(dg, sg_id, sg.trivial_representation)

    def test_quotient_cyclic_odd(self):
        N = 21
        dg = cyclic_group(N)
        for n in range(1, int(round(np.sqrt(N)))+1):
            if N % n == 0:
                sg_id = n
                sg, _, _ = dg.subgroup(sg_id)
                self.check_induction(dg, sg_id, sg.trivial_representation)

    def test_quotient_dihedral_even(self):
        N = 20
        dg = dihedral_group(N)
        for n in range(1, int(round(np.sqrt(N)))+1):
            if N % n == 0:
                for f in range(N//n):
                    sg_id = (f, n)
                    sg, _, _ = dg.subgroup(sg_id)
                    self.check_induction(dg, sg_id, sg.trivial_representation)
                sg_id = (None, n)
                sg, _, _ = dg.subgroup(sg_id)
                self.check_induction(dg, sg_id, sg.trivial_representation)

    def test_quotient_dihedral_odd(self):
        N = 21
        dg = dihedral_group(N)
        for n in range(1, int(round(np.sqrt(N)))+1):
            if N % n == 0:
                for f in range(N//n):
                    sg_id = (f, n)
                    sg, _, _ = dg.subgroup(sg_id)
                    self.check_induction(dg, sg_id, sg.trivial_representation)
                sg_id = (None, n)
                sg, _, _ = dg.subgroup(sg_id)
                self.check_induction(dg, sg_id, sg.trivial_representation)
                
    def test_induce_irreps_dihedral_odd_dihedral_odd(self):
        dg = dihedral_group(9)
        
        for axis in range(3):
            sg_id = (axis, 3)
            
            sg, _, _ = dg.subgroup(sg_id)
            for irrep in sg.irreps():
                self.check_induction(dg, sg_id, irrep)
    
    def test_induce_irreps_dihedral_odd_cyclic_odd(self):
        dg = dihedral_group(9)
        sg_id = (None, 3)
        sg, _, _ = dg.subgroup(sg_id)
        for irrep in sg.irreps():
            self.check_induction(dg, sg_id, irrep)
    
    def test_induce_irreps_dihedral_odd_flips(self):
        dg = dihedral_group(11)
        for axis in range(11):
            sg_id = (axis, 1)
            sg, _, _ = dg.subgroup(sg_id)
            for irrep in sg.irreps():
                self.check_induction(dg, sg_id, irrep)
    
    def test_induce_irreps_cyclic_odd_cyclic_odd(self):
        dg = cyclic_group(9)
        sg_id = 3
        sg, _, _ = dg.subgroup(sg_id)
        for irrep in sg.irreps():
            self.check_induction(dg, sg_id, irrep)
            
    def test_induce_irreps_dihedral_even_dihedral_even(self):
        dg = dihedral_group(12)
        for axis in range(2):
            sg_id = (axis, 6)

            sg, _, _ = dg.subgroup(sg_id)
            for irrep in sg.irreps():
                self.check_induction(dg, sg_id, irrep)

    def test_induce_irreps_dihedral_even_cyclic_even(self):
        dg = dihedral_group(12)
        sg_id = (None, 4)
        sg, _, _ = dg.subgroup(sg_id)
        for irrep in sg.irreps():
            self.check_induction(dg, sg_id, irrep)

    def test_induce_irreps_dihedral_even_dihedral_odd(self):
        dg = dihedral_group(12)
        for axis in range(4):
            sg_id = (axis, 3)
            sg, _, _ = dg.subgroup(sg_id)
            for irrep in sg.irreps():
                self.check_induction(dg, sg_id, irrep)

    def test_induce_irreps_dihedral_even_cyclic_odd(self):
        dg = dihedral_group(12)
        sg_id = (None, 3)
        sg, _, _ = dg.subgroup(sg_id)
        for irrep in sg.irreps():
            self.check_induction(dg, sg_id, irrep)
            
    def test_induce_irreps_dihedral_even_flips(self):
        dg = dihedral_group(12)
        for axis in range(12):
            sg_id = (0, 1)
            sg, _, _ = dg.subgroup(sg_id)
            for irrep in sg.irreps():
                self.check_induction(dg, sg_id, irrep)

    def test_induce_irreps_cyclic_even_cyclic_even(self):
        dg = cyclic_group(8)
        sg_id = 2
        sg, _, _ = dg.subgroup(sg_id)
        for irrep in sg.irreps():
            self.check_induction(dg, sg_id, irrep)
            
    def test_induce_irreps_cyclic_even_cyclic_odd(self):
        dg = cyclic_group(10)
        sg_id = 5
        sg, _, _ = dg.subgroup(sg_id)
        for irrep in sg.irreps():
            self.check_induction(dg, sg_id, irrep)

    def test_induce_irreps_dihedral_even_cyclic(self):
        dg = dihedral_group(12)
        sg_id = (None, 12)
        sg, _, _ = dg.subgroup(sg_id)
        for irrep in sg.irreps():
            self.check_induction(dg, sg_id, irrep)
            
    def test_induce_irreps_dihedral_odd_cyclic(self):
        dg = dihedral_group(13)
        sg_id = (None, 13)
        sg, _, _ = dg.subgroup(sg_id)
        for irrep in sg.irreps():
            self.check_induction(dg, sg_id, irrep)
            
    def test_induce_irreps_so2_o2(self):
        g = o2_group(10)
        sg_id = (None, -1)
        sg, _, _ = g.subgroup(sg_id)
        for irrep in sg.irreps():
            self.check_induction_so2_o2(g, sg_id, irrep)

    def check_induction(self, group, subgroup_id, repr):
        
        # print("#######################################################################################################")
        
        subgroup, parent, child = group.subgroup(subgroup_id)
        
        assert repr.group == subgroup
    
        induced_repr = group.induced_representation(subgroup_id, repr)
        
        assert induced_repr.group == group
        
        # check the change of basis is orthonormal
        self.assertTrue(
            np.allclose(induced_repr.change_of_basis.T @ induced_repr.change_of_basis, np.eye(induced_repr.size)),
            "Change of Basis not orthonormal"
        )
        self.assertTrue(
            np.allclose(induced_repr.change_of_basis @ induced_repr.change_of_basis.T, np.eye(induced_repr.size)),
            "Change of Basis not orthonormal"
        )
        self.assertTrue(
            np.allclose(induced_repr.change_of_basis, induced_repr.change_of_basis_inv.T),
            "Change of Basis not orthonormal"
        )
        
        restricted_repr = group.restrict_representation(subgroup_id, induced_repr)
        for e in subgroup.testing_elements():
            
            repr_a = repr(e)
            repr_b = induced_repr(parent(e))[:repr.size, :repr.size]
            repr_c = restricted_repr(e)[:repr.size, :repr.size]

            np.set_printoptions(precision=2, threshold=2 * repr_a.size**2, suppress=True, linewidth=10*repr_a.size + 3)
            self.assertTrue(np.allclose(repr_a, repr_b), msg=f"{group.name}\{subgroup.name}: {repr.name} - {e}:\n{repr_a}\ndifferent from\n {repr_b}\n")

            if not np.allclose(repr_c, repr_b):
                print(e, parent(e))
                print(induced_repr.change_of_basis_inv @ induced_repr(parent(e)) @ induced_repr.change_of_basis)
                print(restricted_repr.change_of_basis_inv @ restricted_repr(e) @ restricted_repr.change_of_basis)
                print(induced_repr.irreps)
                print(restricted_repr.irreps)
                
                # print(induced_repr.change_of_basis)
                # print(restricted_repr.change_of_basis)
                print(np.allclose(induced_repr.change_of_basis, restricted_repr.change_of_basis))
                
            self.assertTrue(np.allclose(repr_c, repr_b), msg=f"{group.name}\{subgroup.name}: {repr.name} - {e}:\n{repr_c}\ndifferent from\n {repr_b}\n")
            
        quotient_size = int(group.order() / subgroup.order())
        size = repr.size * quotient_size

        # the coset each element belongs to
        cosets = {}

        # map from a representative to the elements of its coset
        representatives = defaultdict(lambda: [])

        for e in group.elements:
            if e not in cosets:
                representatives[e] = []
                for g in subgroup.elements:
                    eg = e @ parent(g)
                
                    cosets[eg] = e
                
                    representatives[e].append(eg)

        index = {e: i for i, e in enumerate(representatives)}
        
        P = directsum([group.irrep(*irr) for irr in induced_repr.irreps], name="irreps")
        
        for g in group.testing_elements():
            repr_g = np.zeros((size, size), dtype=float)
            for r in representatives:
            
                gr = g @ r
            
                g_r = cosets[gr]
            
                i = index[r]
                j = index[g_r]
            
                hp = ~g_r @ gr
            
                h = child(hp)
                assert h is not None, (g, r, gr, g_r, group.inverse(g_r), hp)
            
                repr_g[j*repr.size:(j+1)*repr.size, i*repr.size:(i+1)*repr.size] = repr(h)
            
            ind_g = induced_repr(g)
            self.assertTrue(np.allclose(repr_g, ind_g), msg=f"{group.name}\{subgroup.name}: {repr.name} - {g}:\n{repr_g}\ndifferent from\n {ind_g}\n")
            
            ind_g2 = induced_repr.change_of_basis @ P(g) @ induced_repr.change_of_basis_inv
            self.assertTrue(np.allclose(ind_g2, ind_g),
                            msg=f"{group.name}\{subgroup.name}: {repr.name} - {g}:\n{ind_g2}\ndifferent from\n {ind_g}\n")

    def check_induction_so2_o2(self, group, subgroup_id, repr):
    
        # print("#######################################################################################################")
    
        subgroup, parent, child = group.subgroup(subgroup_id)
    
        assert repr.group == subgroup
    
        # induced_repr = build_induced_representation(group, subgroup_id, repr)
        induced_repr = group.induced_representation(subgroup_id, repr)
    
        assert induced_repr.group == group
        
        assert np.allclose(induced_repr.change_of_basis@induced_repr.change_of_basis_inv, np.eye(induced_repr.size))
        assert np.allclose(induced_repr.change_of_basis_inv@induced_repr.change_of_basis, np.eye(induced_repr.size))
    
        restricted_repr = group.restrict_representation(subgroup_id, induced_repr)
        for e in subgroup.testing_elements():
        
            repr_a = repr(e)
            repr_b = induced_repr(parent(e))[:repr.size, :repr.size]
            repr_c = restricted_repr(e)[:repr.size, :repr.size]
        
            np.set_printoptions(precision=2, threshold=2 * repr_a.size ** 2, suppress=True,
                                linewidth=10 * repr_a.size + 3)
            self.assertTrue(np.allclose(repr_a, repr_b),
                            msg=f"{group.name}\{subgroup.name}: {repr.name} - {e}:\n{repr_a}\ndifferent from\n {repr_b}\n")
        
            if not np.allclose(repr_c, repr_b):
                print(e, parent(e))
                print(induced_repr.change_of_basis_inv @ induced_repr(parent(e)) @ induced_repr.change_of_basis)
                print(restricted_repr.change_of_basis_inv @ restricted_repr(e) @ restricted_repr.change_of_basis)
                print(induced_repr.irreps)
                print(restricted_repr.irreps)
            
                # print(induced_repr.change_of_basis)
                # print(restricted_repr.change_of_basis)
                print(np.allclose(induced_repr.change_of_basis, restricted_repr.change_of_basis))
        
            self.assertTrue(np.allclose(repr_c, repr_b),
                            msg=f"{group.name}\{subgroup.name}: {repr.name} - {e}:\n{repr_c}\ndifferent from\n {repr_b}\n")
    
        quotient_size = 2
        size = repr.size * quotient_size
    
        # the coset each element belongs to
        cosets = {}
    
        # map from a representative to the elements of its coset
        representatives = defaultdict(lambda: [])
    
        for e in group.testing_elements():
            flip, rot = e.to('radians')
            cosets[e] = group.element((flip, 0.))
            representatives[cosets[e]].append(e)
            
        index = {e: i for i, e in enumerate(representatives)}
    
        P = directsum([group.irrep(*irr) for irr in induced_repr.irreps], name="irreps")
    
        for g in group.testing_elements():
            repr_g = np.zeros((size, size), dtype=float)
            for r in representatives:
                gr = g @ r
            
                g_r = cosets[gr]
            
                i = index[r]
                j = index[g_r]
            
                hp = ~g_r @ gr
            
                h = child(hp)
                assert h is not None, (g, r, gr, g_r, group.inverse(g_r), hp)
            
                repr_g[j * repr.size:(j + 1) * repr.size, i * repr.size:(i + 1) * repr.size] = repr(h)
        
            ind_g = induced_repr(g)
            self.assertTrue(np.allclose(repr_g, ind_g),
                            msg=f"{group.name}\{subgroup.name}: {repr.name} - {g}:\n{repr_g}\ndifferent from\n {ind_g}\n")
        
            ind_g2 = induced_repr.change_of_basis @ P(g) @ induced_repr.change_of_basis_inv
            self.assertTrue(np.allclose(ind_g2, ind_g),
                            msg=f"{group.name}\{subgroup.name}: {repr.name} - {g}:\n{ind_g2}\ndifferent from\n {ind_g}\n")

    def test_compatibility_old_induction(self):
        
        def compare_representations(G: Group, sgid, alternatie_implementation):
            H, _, _ = G.subgroup(sgid)
            for psi in H.irreps():
                r1 = G.induced_representation(sgid, psi)
                irreps, cob, cob_inv = alternatie_implementation(G, sgid, psi)
                r2 = directsum(irreps, cob)
                for _ in range(50):
                    g = G.sample()
                    ind_psi1_g = r1(g)
                    ind_psi2_g = r2(g)
                    self.assertTrue(np.allclose(
                        ind_psi1_g,
                        ind_psi2_g
                    ))

        compare_representations(o2_group(3), (None, -1), old_so2_o2_induction)
        
        for n in [2, 3, 4, 5, 8, 12, 15, 20]:
            for j in range(3, n):
                if n % j == 0:
                    compare_representations(cyclic_group(n), j, old_induced_representation)
                    compare_representations(dihedral_group(n), (None, j), old_induced_representation)
                    compare_representations(dihedral_group(n), (0, j), old_induced_representation)


def old_induced_representation(group: Group, subgroup_id, repr: IrreducibleRepresentation) -> Tuple[List[IrreducibleRepresentation], np.ndarray, np.ndarray]:
    r"""

    Build the induced representation of the input ``group`` from the representation ``repr`` of the subgroup
    identified by ``subgroup_id``.

    .. seealso::
        See the :class:`~escnn.group.Group` instance's implementation of the method :meth:`~escnn.group.Group.subgroup`
        for more details on ``subgroup_id``.

    .. warning ::
        Only irreducible representations are supported as the subgroup representation.

    .. warning ::
        Only finite groups are supported.

    Args:
        group (Group): the group whose representation has to be built
        subgroup_id: identifier of the subgroup
        repr (IrreducibleRepresentation): the representation of the subgroup

    Returns:
        a tuple containing the list of irreps, the change of basis and the inverse change of basis of
        the induced representation

    """

    assert repr.irreducible, "Induction from general representations is not supported yet"
    assert group.order() > 0, "Induction from non-discrete groups is not supported yet"
    assert group.elements is not None and len(group.elements) > 0

    subgroup, parent, child = group.subgroup(subgroup_id)

    assert repr.group == subgroup

    quotient_size = int(group.order() / subgroup.order())
    size = repr.size * quotient_size

    # the coset each element belongs to
    cosets = {}

    # map from a representative to the elements of its coset
    representatives = defaultdict(lambda: [])
    index = {}

    for e in group.elements:
        if e not in cosets:
            index[e] = len(representatives)
            representatives[e] = []
        
            for g in subgroup.elements:
                eg = e @ parent(g)
            
                cosets[eg] = e
            
                representatives[e].append(eg)

    # for r, coset in representatives.items():
    #     print(r, coset)

    # index = {e: i for i, e in enumerate(representatives)}

    representation = {}
    character = {}

    for g in group.elements:
        repr_g = np.zeros((size, size), dtype=float)
        for r in representatives:
            gr = g @ r
        
            g_r = cosets[gr]
        
            i = index[r]
            j = index[g_r]
        
            hp = ~g_r @ gr
        
            h = child(hp)
            assert h is not None, (g, r, gr, g_r, ~g_r, hp)
        
            repr_g[j * repr.size:(j + 1) * repr.size, i * repr.size:(i + 1) * repr.size] = repr(h)
    
        representation[g] = repr_g
    
        # the character maps an element to the trace of its representation
        character[g] = np.trace(repr_g)

    # compute the multiplicities of the irreps from the dot product between
    # their characters and the character of the representation
    irreps = []
    multiplicities = []
    for irrep in group.irreps():
        # for each irrep
        multiplicity = 0.0
    
        # compute the inner product with the representation's character
        for element, char in character.items():
            multiplicity += char * irrep.character(~element)
    
        multiplicity /= len(character) * irrep.sum_of_squares_constituents
    
        # the result has to be an integer
        assert math.isclose(multiplicity, round(multiplicity), abs_tol=1e-9), \
            "Multiplicity of irrep %s is not an integer: %f" % (irrep.name, multiplicity)
    
        multiplicity = int(round(multiplicity))
        irreps += [irrep] * multiplicity
        multiplicities += [(irrep, multiplicity)]
        # if multiplicity > 0:
        #     print(irrep_name, irrep.size, multiplicity)

    P = directsum(irreps, name="irreps")

    v = np.zeros((repr.size, size), dtype=float)

    def build_commuting_matrix(rho, t):
        X = np.zeros((rho.size, rho.size))
        if rho.size == 1:
            E = np.eye(1)
        else:
            E = np.array([[1, -1], [1, 1]])
            if t % 2 == 0:
                E = E.T
    
        gr = rho.group
        for h in gr.elements:
            r = rho(h)
            X += r.T @ E @ r
        X /= gr.order()
    
        X /= np.sqrt(np.sum(X @ X.T) / rho.size)
        return X

    p = 0
    for irr, m in multiplicities:
        assert irr.size >= m
    
        if m > 0:
            restricted_irr = group.restrict_representation(subgroup_id, irr)
        
            n_repetitions = len([id for id in restricted_irr.irreps if id == repr.id])
            assert repr.size * n_repetitions >= m, (
                f"{group.name}\{subgroup.name}:{repr.name}", irr.name, m, n_repetitions)
        
            for shift in range(m):
                commuting_matrix = build_commuting_matrix(repr, shift // n_repetitions)
                x = p
                i = 0
                for r_irrep in restricted_irr.irreps:
                    if r_irrep == repr.id:
                        if i == shift % n_repetitions:
                            v[:, x:x + repr.size] = commuting_matrix
                        i += 1
                    x += subgroup.irrep(*r_irrep).size
            
                v[:, p:p + irr.size] = v[:, p:p + irr.size] @ restricted_irr.change_of_basis_inv
                v[:, p:p + irr.size] *= np.sqrt(irr.size)
            
                p += irr.size

    # np.set_printoptions(precision=4, threshold=10 * size ** 2, suppress=True, linewidth=25 * size + 5)
    # Pr = group.restrict_representation(subgroup_id, P)
    # v = v @ Pr.change_of_basis_inv

    # print(v)

    v /= np.sqrt(size)

    change_of_basis = np.zeros((size, size))

    for r in representatives:
        i = index[r]
        change_of_basis[i * repr.size:(i + 1) * repr.size, :] = v @ P(~r)

    change_of_basis_inv = change_of_basis.T

    # for g, r in representation.items():
    #     print(g, np.allclose(change_of_basis @ P(g) @ change_of_basis_inv, representation[g]))
    #     ir = change_of_basis @ P(g) @ change_of_basis_inv
    #     assert np.allclose(ir, representation[g]), f"{group.name}\{subgroup.name}:{repr.name} - {g}:\n{ir}\n{representation[g]}\n"

    return irreps, change_of_basis, change_of_basis_inv


def old_so2_o2_induction(o2: O2, subgroup_id, repr: IrreducibleRepresentation) -> Tuple[List[IrreducibleRepresentation], np.ndarray, np.ndarray]:
    
    # Induced representation from SO(2)
    # As the quotient set is finite, a finite dimensional representation of SO(2)
    # defines a finite dimensional induced representation of O(2)
    
    assert isinstance(o2, O2)
    assert subgroup_id == (None, -1)
    
    subgroup, parent, child = o2.subgroup(subgroup_id)
    assert repr.group == subgroup
    
    name = f"induced[{subgroup_id}][{repr.name}]"
    
    frequency = repr.attributes["frequency"]
    
    if frequency > 0:
        multiplicities = [(o2.irrep(1, frequency), 2)]
    else:
        multiplicities = [(o2.irrep(0, 0), 1), (o2.irrep(1, 0), 1)]
    
    irreps = []
    for irr, multiplicity in multiplicities:
        irreps += [irr] * multiplicity
    
    P = directsum(irreps, name=f"{name}_irreps")
    
    size = P.size
    
    v = np.zeros((repr.size, size), dtype=float)
    
    def build_commuting_matrix(rho, t):
        k = rho.attributes["frequency"]
        
        if rho.size == 1:
            E = np.eye(1)
            M = 2 * np.pi * np.eye(1)
        else:
            E = np.array([[1, -1], [1, 1]])
            if t % 2 == 0:
                E = E.T
            I = np.eye(4)
            A = np.fliplr(np.eye(4)) * np.array([1, -1, -1, 1])
            M = np.pi * (A + I)
        
        # compute the averaging of rho(g).T @ E @ rho(g)
        # i.e. X = 1/2pi Integral_{0, 2pi} rho(theta).T @ E @ rho(theta) d theta
        # as vec(X) = 1/2pi Integral_{0, 2pi} (rho *tensor* rho)(theta) @ vec(E) d theta
        # where M = Integral_{0, 2pi} (rho *tensor* rho)(theta) d theta
        X = M @ E.reshape(-1, 1)
        X /= 2 * np.pi
        
        # normalization
        X /= np.sqrt(np.sum(X @ X.T) / rho.size)
        
        X = X.reshape(rho.size, rho.size)
        
        return X
    
    p = 0
    for irr, m in multiplicities:
        assert irr.size >= m
        
        if m > 0:
            restricted_irr = o2.restrict_representation(subgroup_id, irr)
            
            n_repetitions = len([id for id in restricted_irr.irreps if id == repr.id])
            assert repr.size * n_repetitions >= m, (
                f"{o2.name}\{subgroup.name}:{repr.name}", irr.name, m, n_repetitions)
            
            for shift in range(m):
                commuting_matrix = build_commuting_matrix(repr, shift // n_repetitions)
                x = p
                i = 0
                for r_irrep in restricted_irr.irreps:
                    if r_irrep == repr.id:
                        if i == shift % n_repetitions:
                            v[:, x:x + repr.size] = commuting_matrix
                        i += 1
                    x += subgroup.irrep(*r_irrep).size
                
                v[:, p:p + irr.size] = v[:, p:p + irr.size] @ restricted_irr.change_of_basis_inv
                v[:, p:p + irr.size] *= np.sqrt(irr.size)
                
                p += irr.size
    
    v /= np.sqrt(size)
    
    change_of_basis = np.zeros((size, size))
    
    change_of_basis[:repr.size, :] = v @ P(o2.identity)
    change_of_basis[repr.size:, :] = v @ P(o2.reflection)
    
    change_of_basis_inv = change_of_basis.T
    
    return irreps, change_of_basis, change_of_basis_inv


if __name__ == '__main__':
    unittest.main()
