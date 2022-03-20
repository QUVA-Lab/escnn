import unittest
from unittest import TestCase

from escnn.group import *

import numpy as np


class TestSubgroups(TestCase):
    
    ####################################################################################################################
    
    def test_cn(self):
        for n in [1, 2, 4, 6]:
            self.check_all_combinations(cyclic_group(n))
            
    def test_dn(self):
        for n in [1, 2, 4, 6]:
            self.check_all_combinations(dihedral_group(n))

    def test_so2(self):
        self.check_all_combinations(so2_group(1))
        
    def test_o2(self):
        self.check_all_combinations(o2_group(1))

    def test_ico(self):
        self.check_all_combinations(ico_group())

    def test_so3(self):
        self.check_all_combinations(so3_group(1))

    def test_o3(self):
        self.check_all_combinations(o3_group(1))

    ####################################################################################################################
    
    def subgroup_to_test(self, G: Group):
        
        if isinstance(G, CyclicGroup):
            for i in range(1, G.order()):
                if G.order() % i == 0:
                    yield i
                    
        elif isinstance(G, DihedralGroup):
            for i in range(1, G.rotation_order):
                if G.rotation_order % i == 0:
                    yield (None, i)
                    for k in range(1, G.rotation_order // i):
                        yield (k, i)
                        
        elif isinstance(G, SO2):
            for i in [1, 2, 3, 5]:
                yield i
        elif isinstance(G, O2):
            for i in [1, 2, 3]:
                yield (None, i)
                K = 4
                for k in range(K):
                    # axis = np.random.rand() * 2* np.pi / i
                    axis = k * 2 * np.pi / i / K
                    yield (axis, i)

        elif isinstance(G, Icosahedral):

            for a in range(1):

                adjoint = G.sample()

                yield ('ico', adjoint)
                yield ('tetra', adjoint)

                for i in [1, 2, 3, 5]:
                    for k in [True, False]:
                        yield (k, i, adjoint)

        elif isinstance(G, SO3):

            for a in range(3):
                
                # r = np.zeros(4)
                # r[a] = 1.
                # adjoint = G.element(r, 'Q')
                # adjoint = G.sample()
                adjoint = G.identity

                yield ('so3', adjoint)
                yield ('ico', adjoint)

                for i in [-1, 1, 2, 3]:
                    for k in [True, False]:
                        yield (k, i, adjoint)
                        
        elif isinstance(G, O3):
            so3 = so3_group()

            for inv in [True, False]:
                for so3_sg in self.subgroup_to_test(so3):
                    adj = so3_sg[-1]
                    adj = G.element((0, adj.to('Q')), 'Q')
                    yield (inv,) + so3_sg[:-1] + (adj,)
                    
            for _ in range(3):
                adjoint = G.sample()
                
                for i in [-1, 1, 2, 3]:
                    yield ('cone', i, adjoint)
        else:
            raise ValueError
    
    def check_all_combinations(self, group: Group):
        for sg_id1 in self.subgroup_to_test(group):
            try:
                sg, _, _ = group.subgroup(sg_id1)
            except NotImplementedError:
                print(f"{group}: subgroup {sg_id1} not implemented")
                return

            try:
                sg_ids = self.subgroup_to_test(sg)
                
                for sg_id2 in sg_ids:
                    self.check_restriction(group, sg_id1, sg_id2)
                    
            except ValueError as e:
                print(f'No testing subgroups for {sg}')
                continue

    def check_restriction(self, group: Group, sg_id1, sg_id2):
        
        try:
            sg1, inclusion1, restriction1 = group.subgroup(sg_id1)
        except NotImplementedError:
            print(f"{group}: subgroup {sg_id1} not implemented")
            return
        
        try:
            sg2, inclusion2, restriction2 = sg1.subgroup(sg_id2)
        except NotImplementedError:
            print(f"{sg1}: subgroup {sg_id2} not implemented")
            return
    
        sg_id12 = group._combine_subgroups(sg_id1, sg_id2)
        sg12, inclusion12, restriction12 = group.subgroup(sg_id12)
    
        self.assertTrue(sg2 == sg12, f"{group} | {sg_id1} - {sg_id2}: {sg2} != {sg12}")
    
        for e in sg12.testing_elements():
            p = inclusion1(inclusion2(e))
            p12 = inclusion12(e)
            self.assertTrue(p == p12,
                             f"{group} to subgroup {sg2} with id {sg_id1} combined with {sg_id2} | Combined id {sg_id12} | Element {e}")
                

if __name__ == '__main__':
    unittest.main()
