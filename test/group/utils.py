import numpy as np
import pickle
import os

from unittest import TestCase
from escnn.group import *
from collections.abc import Callable
from typing import Callable, Any


def check_singleton(
        tester: TestCase,
        factory: Callable[..., Group],
        *args: Any,
        **kwargs: Any,
):
    g1 = factory(*args, **kwargs)
    g2 = factory(*args, **kwargs)
    g3 = pickle.loads(pickle.dumps(g1))

    tester.assertIs(g1, g2)
    tester.assertIs(g1, g3)
    tester.assertIs(g2, g3)

    # Make sure group elements are also picklable.
    for e1 in testing_elements(g1):
        e2 = pickle.loads(pickle.dumps(e1))

        assert e1.group is e2.group
        assert e1 == e2

    return g1

def check_generators(tester: TestCase, group: Group):
    if group.order() > 0:
        generators = group.generators
        if group.order() > 1:
            tester.assertTrue(len(generators) > 0)
    else:
        with tester.assertRaises(ValueError):
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

    tester.assertTrue(
        len(elements) == group.order(),
        'Error! The set of generators does not generate the whole group'
    )

    for a in elements:
        tester.assertIn(~a, elements)
        for b in elements:
            tester.assertIn(a @ b, elements)

def check_operations(tester: TestCase, group: Group):
    e = group.identity
    
    for a in testing_elements(group):
        
        tester.assertTrue(a @ e == a)
        tester.assertTrue(e @ a == a)
        
        i = ~a
        tester.assertTrue(a @ i, e)
        tester.assertTrue(i @ a, e)
        
        for b in testing_elements(group):
            for c in testing_elements(group):

                ab = a @ b
                bc = b @ c
                a_bc = a @ bc
                ab_c = ab @ c
                tester.assertTrue(a_bc == ab_c, f"{a_bc} != {ab_c}")

def check_direct_product(tester: TestCase, group: Group):
    G1 = group.G1
    G2 = group.G2

    for _ in range(100):
        a = G1.sample()
        b = G2.sample()
        a = group.inclusion1(a)
        b = group.inclusion2(b)

        ab = a @ b
        ba = b @ a

        tester.assertTrue(ab == ba)

    for _ in range(100):
        a = G1.sample()
        C = group.sample()

        AC = group.inclusion1(a) @ C
        c1, c2 = group.split_element(C)

        ac = group.pair_elements(a @ c1, c2)

        tester.assertTrue(ac == AC)

    for _ in range(100):
        b = G2.sample()
        C = group.sample()
        BC = group.inclusion2(b) @ C
        c1, c2 = group.split_element(C)

        bc = group.pair_elements(c1, b@c2)

        tester.assertTrue(bc == BC)

def check_regular_repr(tester: TestCase, group: Group):
    if group.order() > 0:
        reg = group.regular_representation
        check_representation(tester, reg)

def check_irreps(tester: TestCase, group: Group):
    for irrep in group.irreps():
        check_irrep_endom(tester, irrep)
        check_representation(tester, irrep)
        check_character(tester, irrep)
        
def check_irrep_endom(tester: TestCase, irrep: IrreducibleRepresentation):
    group = irrep.group

    np.set_printoptions(precision=2, threshold=2 * irrep.size ** 2, suppress=True,
                        linewidth=10 * irrep.size + 3)

    tester.assertTrue(irrep.sum_of_squares_constituents == irrep.endomorphism_basis().shape[0])

    for k in range(irrep.sum_of_squares_constituents):
        E_k = irrep.endomorphism_basis()[k, ...]

        # check orthogonality
        tester.assertTrue(np.allclose(E_k @ E_k.T, np.eye(irrep.size)))
        tester.assertTrue(np.allclose(E_k.T @ E_k, np.eye(irrep.size)))

        if k == 0:
            # if k=0, the matrix need to be the identity
            tester.assertTrue(np.allclose(E_k, np.eye(irrep.size)))
        else:
            # all other ones need to be skew symmetric
            tester.assertTrue(np.allclose(E_k, -E_k.T))

        for l in range(irrep.sum_of_squares_constituents):
            E_l = irrep.endomorphism_basis()[l, ...]
    
            d_kl = (E_l * E_k).sum() / irrep.size
    
            if k == l:
                tester.assertTrue(np.allclose(d_kl, 1.))
            else:
                tester.assertTrue(np.allclose(d_kl, 0.))
        
                for i in range(irrep.size):
                    tester.assertTrue(np.allclose(
                        E_k[:, i].T @ E_l[:, i], 0.
                    ))

        w = (irrep.endomorphism_basis() ** 2).sum(axis=0) > 1e-9
        tester.assertTrue(np.allclose(w.sum(axis=1), irrep.sum_of_squares_constituents))

    end_basis = irrep.endomorphism_basis()

    for a in testing_elements(group):
        r = irrep(a)

        tester.assertTrue(np.allclose(r @ end_basis, end_basis @ r),
                        msg=f"{group.name} | {irrep.name} | {a}: endomorphism basis not equivariant\n\n")

def check_representation(tester: TestCase, repr: Representation):

    group = repr.group
    
    np.set_printoptions(precision=2, threshold=2 * repr.size ** 2, suppress=True,
                        linewidth=10 * repr.size + 3)
    
    P = directsum([group.irrep(*irr) for irr in repr.irreps], name="irreps")
    
    tester.assertTrue(np.allclose(repr.change_of_basis @ repr.change_of_basis.T, np.eye(repr.size)))
    tester.assertTrue(np.allclose(repr.change_of_basis.T @ repr.change_of_basis, np.eye(repr.size)))
    
    for a in testing_elements(group):
        repr_1 = repr(a)
        repr_2 = repr.change_of_basis @ P(a) @ repr.change_of_basis_inv

        tester.assertTrue(np.allclose(repr_1, repr_2),
                        msg=f"{a}:\n{repr_1}\ndifferent from\n {repr_2}\n")
        
        repr_a_inv = repr(~a)
        repr_inv_a = np.linalg.inv(repr_1)
        tester.assertTrue(np.allclose(repr_a_inv, repr_inv_a),
                        msg=f"{a}:\n{repr_a_inv}\ndifferent from\n {repr_inv_a}\n")
        
        for b in testing_elements(group):
            repr_ab = repr(a) @ repr(b)
            c = a @ b
            repr_c = repr(c)
            
            tester.assertTrue(np.allclose(repr_ab, repr_c), msg=f"{a} x {b} = {c}:\n{repr_ab}\ndifferent from\n {repr_c}\n")

def check_character(tester: TestCase, repr: Representation):

    group = repr.group

    np.set_printoptions(precision=2, threshold=2 * repr.size ** 2, suppress=True,
                        linewidth=10 * repr.size + 3)

    for a in testing_elements(group):
        char_a_1 = repr.character(a)
        char_a_2 = np.trace(repr(a))
    
        tester.assertAlmostEqual(char_a_1, char_a_2,
                               msg=f"""{a}: Character of {repr} different from its trace \n {char_a_1} != {char_a_2} \n""")

def testing_elements(group: Group):
    max_n = int(os.getenv('ESCNN_MAX_TESTING_ELEMENTS', 15))

    elements = list(group.testing_elements())
    if len(elements) <= max_n:
        return elements
    else:
        return [group.sample() for _ in range(max_n)]
    
