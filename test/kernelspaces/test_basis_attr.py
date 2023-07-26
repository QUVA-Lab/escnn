import unittest
from unittest import TestCase

import numpy as np

from escnn.group import *
from escnn.kernels import *


class TestBasisAttributes(TestCase):
    
    def test_trivial(self):
        N = 1
        group = cyclic_group(N)
        in_rep = group.regular_representation
        out_rep = group.regular_representation
        
        basis = kernels_Trivial_act_R2(in_rep, out_rep,
                                       radii=[0., 1., 2., 5, 10],
                                       sigma=[0.6, 1., 1.3, 2.5, 3.],
                                       maximum_frequency=5)
        self._check(basis)

    def test_flips(self):
        group = cyclic_group(2)
        # in_rep = group.regular_representation
        # out_rep = group.regular_representation
        in_rep = directsum(group.irreps(), name="irreps_sum")
        out_rep = directsum(group.irreps(), name="irreps_sum")

        # axis = 0.
        # for axis in [0., np.pi / 2, np.pi/3]:
        # for axis in [np.pi/2]:
        A = 10
        for a in range(A):
            axis = a*np.pi/A
    
            basis = kernels_Flip_act_R2(in_rep, out_rep, axis=axis,
                                        radii=[0., 1., 2., 5, 10],
                                        sigma=[0.6, 1., 1.3, 2.5, 3.],
                                        maximum_frequency=5)
    
            self._check(basis)

    def test_cyclic_odd_regular(self):
        N = 3
        group = cyclic_group(N)
        in_rep = group.regular_representation
        out_rep = group.regular_representation
        
        for _ in range(5):
            basis = kernels_CN_act_R2(in_rep, out_rep,
                                      radii=[0., 1., 2., 5, 10],
                                      sigma=[0.6, 1., 1.3, 2.5, 3.],
                                      maximum_frequency=5)
            self._check(basis)
    
    def test_cyclic_even_regular(self):
        N = 6
        group = cyclic_group(N)
        in_rep = group.regular_representation
        out_rep = group.regular_representation
        
        basis = kernels_CN_act_R2(in_rep, out_rep,
                                  radii=[0., 1., 2., 5, 10],
                                  sigma=[0.6, 1., 1.3, 2.5, 3.],
                                  maximum_frequency=5)
        self._check(basis)

    def test_cyclic_mix(self):
        N = 3
        group = cyclic_group(N)
        in_rep = directsum(group.irreps(), name="irreps_sum")
        out_rep = directsum(group.irreps(), name="irreps_sum")
    
        for _ in range(5):
            basis = kernels_CN_act_R2(in_rep, out_rep,
                                      radii=[0., 1., 2., 5, 10],
                                      sigma=[0.1, 1., 1.3, 2.5, 3.],
                                      maximum_frequency=5)
            self._check(basis)

    def test_cyclic_output_changeofbasis(self):
        N = 3
        group = cyclic_group(N)
        in_rep = directsum(group.irreps(), name="irreps_sum")
        out_rep = group.regular_representation

        for _ in range(5):
            basis = kernels_CN_act_R2(in_rep, out_rep,
                                      radii=[0., 1., 2., 5, 10],
                                      sigma=[0.1, 1., 1.3, 2.5, 3.],
                                      maximum_frequency=5)
            self._check(basis)

    def test_cyclic_input_changeofbasis(self):
        N = 3
        group = cyclic_group(N)
        in_rep = group.regular_representation
        out_rep = directsum(group.irreps(), name="irreps_sum")
    
        for _ in range(5):
            basis = kernels_CN_act_R2(in_rep, out_rep,
                                      radii=[0., 1., 2., 5, 10],
                                      sigma=[0.1, 1., 1.3, 2.5, 3.],
                                      maximum_frequency=5)
            self._check(basis)

    def test_dihedral_odd_regular(self):
        N = 5
        group = dihedral_group(N)
        in_rep = group.regular_representation
        out_rep = group.regular_representation
        axis = np.pi / 2
    
        basis = kernels_DN_act_R2(in_rep, out_rep,
                                  axis=axis,
                                  radii=[0., 1., 2., 5, 10],
                                  sigma=[0.6, 1., 1.3, 2.5, 3.],
                                  maximum_frequency=5)

        self._check(basis)

    def test_dihedral_even_regular(self):
        N = 2
        group = dihedral_group(N)
        in_rep = group.regular_representation
        out_rep = group.regular_representation
        axis = np.pi/2

        basis = kernels_DN_act_R2(in_rep, out_rep,
                                  axis=axis,
                                  radii=[0., 1., 2., 5, 10],
                                  sigma=[0.6, 1., 1.3, 2.5, 3.],
                                  maximum_frequency=5)

        self._check(basis)

    def test_cyclic_irreps(self):
        N = 8
        group = cyclic_group(N)
        
        irreps = group.irreps()
        for in_rep in irreps:
            for out_rep in irreps:
    
                basis = kernels_CN_act_R2(in_rep, out_rep,
                                          radii=[0., 1., 2., 5, 10],
                                          sigma=[0.6, 1., 1.3, 2.5, 3.],
                                          maximum_frequency=5)
                self._check(basis)

    def test_dihedral_irreps(self):
        N = 4
        group = dihedral_group(N)

        for axis in [0., np.pi/2, np.pi/3]:
            for in_rep in group.irreps():
                for out_rep in group.irreps():
                    basis = kernels_DN_act_R2(in_rep, out_rep,
                                              axis=axis,
                                              radii=[0., 1., 2., 5, 10],
                                              sigma=[0.6, 1., 1.3, 2.5, 3.],
                                              maximum_frequency=13)

                    self._check(basis)

    def test_dihedral_2_irreps(self):
        N = 2
        group = dihedral_group(N)
        axis = np.pi / 2
        
        reprs = group.irreps() + [directsum(group.irreps(), name="irreps_sum"), group.regular_representation]

        for in_rep in reprs:
            for out_rep in reprs:
                try:
                    basis = kernels_DN_act_R2(in_rep, out_rep,
                                              axis=axis,
                                              radii=[0., 1., 2., 5, 10],
                                              sigma=[0.6, 1., 1.3, 2.5, 3.],
                                              maximum_frequency=2)
                except EmptyBasisException:
                    print(f"KernelBasis between {in_rep.name} and {out_rep.name} is empty, continuing")
                    continue

                self._check(basis)

    def test_flips_irreps(self):
        group = cyclic_group(2)

        # for axis in [0., np.pi/3, np.pi/2,]:
        A = 10
        for axis in range(A):
            axis = axis * np.pi/A
            for in_rep in group.irreps() + [group.regular_representation]:
                for out_rep in group.irreps() + [group.regular_representation]:
                    basis = kernels_Flip_act_R2(in_rep, out_rep,
                                                axis=axis,
                                                radii=[0., 1., 2., 5, 10],
                                                sigma=[0.6, 1., 1.3, 2.5, 3.],
                                                maximum_frequency=7)

                    self._check(basis)

    def test_so2_irreps(self):
        
        group = so2_group(10)
    
        irreps = [group.irrep(l) for l in range(5)]
        for in_rep in irreps:
            for out_rep in irreps:
                basis = kernels_SO2_act_R2(in_rep, out_rep,
                                           radii=[0., 1., 2., 5, 10],
                                           sigma=[0.6, 1., 1.3, 2.5, 3.]
                                           )
                self._check(basis)

    def test_o2_irreps(self):
    
        group = o2_group(10)

        irreps = [group.irrep(0, 0)] + [group.irrep(1, l) for l in range(5)]
        for in_rep in irreps:
            for out_rep in irreps:
                for axis in [0., np.pi/2, np.pi/3, np.pi / 4]:
                    try:
                        basis = kernels_O2_act_R2(in_rep, out_rep,
                                                  axis=axis,
                                                  radii=[0., 1., 2., 5, 10],
                                                  sigma=[0.6, 1., 1.3, 2.5, 3.]
                                                  )
                    except EmptyBasisException:
                        print(f"KernelBasis between {in_rep.name} and {out_rep.name} is empty, continuing")
                        continue

                    self._check(basis)

    def test_ico_irreps(self):

        group = ico_group()

        irreps = [group.irrep(l) for l in range(5)]
        for in_rep in irreps:
            for out_rep in irreps:
                basis = kernels_Ico_act_R3(in_rep, out_rep,
                                           radii=[0., 1., 2., 5, 10],
                                           sigma=[0.3, 1., 1.3, 2.5, 3.]
                                           )
                self._check(basis)

    def test_octa_irreps(self):

        group = octa_group()

        irreps = [group.irrep(l) for l in range(-1, 4)]
        for in_rep in irreps:
            for out_rep in irreps:
                basis = kernels_Octa_act_R3(in_rep, out_rep,
                                           radii=[0., 1., 2., 5, 10],
                                           sigma=[0.3, 1., 1.3, 2.5, 3.]
                                           )
                self._check(basis)

    def test_so3_irreps(self):
    
        group = so3_group(9)

        irreps = [group.irrep(l) for l in range(5)]
        for in_rep in irreps:
            for out_rep in irreps:
                basis = kernels_SO3_act_R3(in_rep, out_rep,
                                           radii=[0., 1., 2., 5, 10],
                                           sigma=[0.3, 1., 1.3, 2.5, 3.]
                                           )
                self._check(basis)

    def test_so3_others(self):
    
        group = so3_group(5)
        
        reprs = [
        #     group.irrep(l) for l in range(3)
        # ] + [
            group.standard_representation(),
        ] + [
            group.bl_regular_representation(l) for l in range(1, 4)
        ]
        
        for in_rep in reprs:
            for out_rep in reprs:
                print(in_rep, out_rep)
                basis = kernels_SO3_act_R3(in_rep, out_rep,
                                           radii=[0., 1., 2., 5, 10],
                                           sigma=[0.3, 1., 1.3, 2.5, 3.]
                                           )
                self._check(basis)

    def test_o3_irreps(self):
    
        group = o3_group(9)
    
        irreps = [group.irrep(j, l) for l in range(5) for j in range(2)]
        for in_rep in irreps:
            for out_rep in irreps:
                try:
                    basis = kernels_O3_act_R3(in_rep, out_rep,
                                               radii=[0., 1., 2., 5, 10],
                                               sigma=[0.3, 1., 1.3, 2.5, 3.]
                                               )
                except EmptyBasisException:
                    print(f"KernelBasis between {in_rep.name} and {out_rep.name} is empty, continuing")
                    continue

                self._check(basis)

    def test_o3_others(self):
    
        group = o3_group(5)
    
        reprs = [
                #     group.irrep(j, l) for l in range(3) for j in range(2)
                # ] + [
                    group.standard_representation(),
                ] + [
                    group.bl_regular_representation(l) for l in range(1, 5, 2)
                ]
    
        for in_rep in reprs:
            for out_rep in reprs:
                try:
                    basis = kernels_O3_act_R3(in_rep, out_rep,
                                               radii=[0., 1., 2., 5, 10],
                                               sigma=[0.3, 1., 1.3, 2.5, 3.]
                                               )
                except EmptyBasisException:
                    print(f"KernelBasis between {in_rep.name} and {out_rep.name} is empty, continuing")
                    continue
                    
                self._check(basis)

    def test_so2_irreps_filtered(self):

        group = so2_group(10)

        irreps = [group.irrep(l) for l in range(5)]
        for in_rep in irreps:
            for out_rep in irreps:
                try:
                    basis = kernels_SO2_act_R2(in_rep, out_rep,
                                               radii=[0., 1., 2., 5, 10],
                                               sigma=[0.6, 1., 1.3, 2.5, 3.],
                                               filter = lambda attr : (attr['j'][1] == 0)
                                               )
                    self._check(basis)
                except EmptyBasisException:
                    pass

    def test_ico_irreps_sparse_dodec(self):
        group = ico_group()
        irreps = [group.irrep(l) for l in range(5)]
        for in_rep in irreps:
            for out_rep in irreps:
                try:
                    basis = kernels_aliased_Ico_act_R3_dodecahedron(in_rep, out_rep,
                                               radii=[0., 1., 2., 5, 10],
                                               sigma=[0.3, 1., 1.3, 2.5, 3.]
                                               )
                except EmptyBasisException:
                    print(f"KernelBasis between {in_rep.name} and {out_rep.name} is empty, continuing")
                    continue
                self._check(basis)

    def test_ico_irreps_sparse_icosidodec(self):
        group = ico_group()
        irreps = [group.irrep(l) for l in range(5)]
        for in_rep in irreps:
            for out_rep in irreps:
                try:
                    basis = kernels_aliased_Ico_act_R3_icosidodecahedron(in_rep, out_rep,
                                                                    radii=[0., 1., 2., 5, 10],
                                                                    sigma=[0.3, 1., 1.3, 2.5, 3.]
                                                                    )
                except EmptyBasisException:
                    print(f"KernelBasis between {in_rep.name} and {out_rep.name} is empty, continuing")
                    continue
                self._check(basis)

    def test_ico_irreps_sparse_ico(self):
        group = ico_group()
        irreps = [group.irrep(l) for l in range(5)]
        for in_rep in irreps:
            for out_rep in irreps:
                try:
                    basis = kernels_aliased_Ico_act_R3_icosahedron(in_rep, out_rep,
                                                                    radii=[0., 1., 2., 5, 10],
                                                                    sigma=[0.3, 1., 1.3, 2.5, 3.]
                                                                    )
                except EmptyBasisException:
                    print(f"KernelBasis between {in_rep.name} and {out_rep.name} is empty, continuing")
                    continue
                self._check(basis)

    def _check(self, basis: KernelBasis):
        if basis is None:
            print("Empty KernelBasis!")
            return

        if isinstance(basis, SteerableKernelBasis):
            for ii in range(len(basis.in_sizes)):
                for oo in range(len(basis.out_sizes)):
                    b = basis.bases[ii][oo]
                    if b is not None:

                        for j in b.js:
                            l = len(list(b.attrs_j_iter(j)))
                            assert l == b.dim_harmonic(j), (l, b.dim_harmonic(j), j)

                        l = len(list(b))
                        assert l == b.dim, (l, b.dim, len(b.js), ii, oo, basis.in_repr.irreps[ii], basis.out_repr.irreps[oo])

        l = len(list(basis))
        assert l == basis.dim, (l, basis.dim)

        for i, attr1 in enumerate(basis):
            attr2 = basis[i]
            self.assertEqual(attr1['idx'], i)
            self.assertEqual(attr1, attr2)


if __name__ == '__main__':
    unittest.main()
