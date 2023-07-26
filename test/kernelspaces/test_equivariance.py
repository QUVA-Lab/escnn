import unittest
from unittest import TestCase

import numpy as np
import torch

from escnn.group import *
from escnn.kernels import *

from escnn.gspaces import *
import scipy


class TestSolutionsEquivariance(TestCase):
    
    def test_trivial(self):
        N = 1
        group = cyclic_group(N)
        in_rep = group.regular_representation
        out_rep = group.regular_representation
        
        basis = kernels_Trivial_act_R2(in_rep, out_rep,
                                       radii=[0., 1., 2., 5, 10],
                                       sigma=[0.6, 1., 1.3, 2.5, 3.],
                                       maximum_frequency=5)
        action = group.irrep(0) + group.irrep(0)
        self._check(basis, group, in_rep, out_rep, action)

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
            print(axis)
    
            basis = kernels_Flip_act_R2(in_rep, out_rep, axis=axis,
                                        radii=[0., 1., 2., 5, 10],
                                        sigma=[0.6, 1., 1.3, 2.5, 3.],
                                        maximum_frequency=5)
    
            cob = so2_group(1).irrep(1)(so2_group().element(axis, 'radians'))
            action = directsum([group.irrep(0), group.irrep(1)], cob, f"horizontal_flip_{axis}")
    
            self._check(basis, group, in_rep, out_rep, action)

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
            action = group.irrep(1)
            self._check(basis, group, in_rep, out_rep, action)
    
    def test_cyclic_even_regular(self):
        N = 6
        group = cyclic_group(N)
        in_rep = group.regular_representation
        out_rep = group.regular_representation
        
        basis = kernels_CN_act_R2(in_rep, out_rep,
                                  radii=[0., 1., 2., 5, 10],
                                  sigma=[0.6, 1., 1.3, 2.5, 3.],
                                  maximum_frequency=5)
        action = group.irrep(1)
        self._check(basis, group, in_rep, out_rep, action)

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
            action = group.irrep(1)
            self._check(basis, group, in_rep, out_rep, action)

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
            action = group.irrep(1)
            self._check(basis, group, in_rep, out_rep, action)

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
            action = group.irrep(1)
            self._check(basis, group, in_rep, out_rep, action)

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

        cob = so2_group(1).irrep(1)(so2_group().element(axis, 'radians'))
        action = change_basis(group.irrep(1, 1), cob, "horizontal_flip")
    
        self._check(basis, group, in_rep, out_rep, action)

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

        cob = so2_group(1).irrep(1)(so2_group().element(axis, 'radians'))
        action = change_basis(group.irrep(0, 1) + group.irrep(1, 1), cob, "horizontal_flip")

        self._check(basis, group, in_rep, out_rep, action)

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
                action = group.irrep(1)
                self._check(basis, group, in_rep, out_rep, action)

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

                    cob = so2_group(1).irrep(1)(so2_group().element(axis, 'radians'))
                    action = change_basis(group.irrep(1, 1), cob, "horizontal_flip")

                    self._check(basis, group, in_rep, out_rep, action)

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

                cob = so2_group().element(axis).to('MAT')
                action = change_basis(group.irrep(0, 1) + group.irrep(1, 1), cob, "horizontal_flip")
            
                self._check(basis, group, in_rep, out_rep, action)

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

                    cob = so2_group().element(axis).to('MAT')
                    action = directsum([group.irrep(0), group.irrep(1)], cob, f"horizontal_flip_{axis}")

                    self._check(basis, group, in_rep, out_rep, action)

    def test_so2_irreps(self):
        
        group = so2_group(10)
    
        irreps = [group.irrep(l) for l in range(5)]
        for in_rep in irreps:
            for out_rep in irreps:
                basis = kernels_SO2_act_R2(in_rep, out_rep,
                                           radii=[0., 1., 2., 5, 10],
                                           sigma=[0.6, 1., 1.3, 2.5, 3.]
                                           )
                action = group.irrep(1)
                self._check(basis, group, in_rep, out_rep, action)

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

                    action = group.irrep(1, 1)
                    action = change_basis(action, action(group.element((0, axis), 'radians')),
                                          name=f'StandardAction|axis=[{axis}]')
                    self._check(basis, group, in_rep, out_rep, action)

    def test_so2_irreps_onR3(self):

        group = so2_group(10)

        irreps = [group.irrep(l) for l in range(5)]
        for in_rep in irreps:
            for out_rep in irreps:
                basis = kernels_SO2_act_R3(in_rep, out_rep,
                                           radii=[0., 1., 2., 5, 10],
                                           sigma=[0.6, 1., 1.3, 2.5, 3.]
                                           )
                action = so3_group().standard_representation().restrict((False, -1))
                self._check(basis, group, in_rep, out_rep, action, D=3)

    def test_o2_irreps_onR3(self):

        group = o2_group(10)

        irreps = [group.irrep(0, 0)] + [group.irrep(1, l) for l in range(5)]
        for in_rep in irreps:
            for out_rep in irreps:
                try:
                    basis = kernels_O2_conical_act_R3(in_rep, out_rep,
                                                      radii=[0., 1., 2., 5, 10],
                                                      sigma=[0.6, 1., 1.3, 2.5, 3.]
                                                      )
                except EmptyBasisException:
                    print(f"KernelBasis between {in_rep.name} and {out_rep.name} is empty, continuing")
                    continue

                action = o3_group().standard_representation().restrict(('cone', -1))
                self._check(basis, group, in_rep, out_rep, action, D=3)

    def test_so3_irreps(self):

        group = so3_group(9)

        irreps = [group.irrep(l) for l in range(5)]
        for in_rep in irreps:
            for out_rep in irreps:
                basis = kernels_SO3_act_R3(in_rep, out_rep,
                                           radii=[0., 1., 2., 5, 10],
                                           sigma=[0.3, 1., 1.3, 2.5, 3.]
                                           )
                action = group.standard_representation()
                self._check(basis, group, in_rep, out_rep, action, D=3)

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
                action = group.standard_representation()
                self._check(basis, group, in_rep, out_rep, action, D=3)

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

                action = group.standard_representation()
                self._check(basis, group, in_rep, out_rep, action, D=3)

    def test_o3_others(self):

        group = o3_group(5)

        reprs = [
                #     group.irrep(j, l) for l in range(3) for j in range(2)
                # ] + [
                    group.standard_representation(),
                ] + [
                    group.bl_regular_representation(l) for l in range(1, 3)
                ]

        for in_rep in reprs:
            for out_rep in reprs:
                print(in_rep, out_rep)
                try:
                    basis = kernels_O3_act_R3(in_rep, out_rep,
                                               radii=[0., 1., 2., 5, 10],
                                               sigma=[0.3, 1., 1.3, 2.5, 3.]
                                               )
                except EmptyBasisException:
                    print(f"KernelBasis between {in_rep.name} and {out_rep.name} is empty, continuing")
                    continue

                action = group.standard_representation()
                self._check(basis, group, in_rep, out_rep, action, D=3)

    def test_sparse_dodec_verteces(self):

        group = ico_group()

        reprs = [
                    #     group.irrep(j, l) for l in range(3) for j in range(2)
                    # ] + [
                    group.standard_representation,
                ] + [
                    group.bl_regular_representation(l) for l in range(1, 5, 2)
                ]

        for in_rep in reprs:
            for out_rep in reprs:
                try:
                    basis = kernels_aliased_Ico_act_R3_dodecahedron(in_rep, out_rep,
                                                                    radii=[0., 1., 2., 5, 10],
                                                                    sigma=[0.3, 1., 1.3, 2.5, 3.]
                                                                    )
                except EmptyBasisException:
                    print(f"KernelBasis between {in_rep.name} and {out_rep.name} is empty, continuing")
                    continue

                action = group.standard_representation
                self._check(basis, group, in_rep, out_rep, action, D=3)

    def test_sparse_ico_verteces(self):

        group = ico_group()

        reprs = [
                    #     group.irrep(j, l) for l in range(3) for j in range(2)
                    # ] + [
                    group.standard_representation,
                ] + [
                    group.bl_regular_representation(l) for l in range(1, 5, 2)
                ]

        for in_rep in reprs:
            for out_rep in reprs:
                try:
                    basis = kernels_aliased_Ico_act_R3_icosahedron(in_rep, out_rep,
                                                                   radii=[0., 1., 2., 5, 10],
                                                                   sigma=[0.3, 1., 1.3, 2.5, 3.]
                                                                   )
                except EmptyBasisException:
                    print(f"KernelBasis between {in_rep.name} and {out_rep.name} is empty, continuing")
                    continue

                action = group.standard_representation
                self._check(basis, group, in_rep, out_rep, action, D=3)

    def test_sparse_ico_edges(self):

        group = ico_group()

        reprs = [
                    #     group.irrep(j, l) for l in range(3) for j in range(2)
                    # ] + [
                    group.standard_representation,
                ] + [
                    group.bl_regular_representation(l) for l in range(1, 5, 2)
                ]

        for in_rep in reprs:
            for out_rep in reprs:
                try:
                    basis = kernels_aliased_Ico_act_R3_icosidodecahedron(in_rep, out_rep,
                                                                         radii=[0., 1., 2., 5, 10],
                                                                         sigma=[0.3, 1., 1.3, 2.5, 3.]
                                                                         )
                except EmptyBasisException:
                    print(f"KernelBasis between {in_rep.name} and {out_rep.name} is empty, continuing")
                    continue

                action = group.standard_representation
                self._check(basis, group, in_rep, out_rep, action, D=3)

    def _check(self, basis: KernelBasis, group, in_rep, out_rep, action, D=2):
        if basis is None:
            print("Empty KernelBasis!")
            return
        
        P = 9
        B = 100

        if D == 2:
            square_points = torch.tensor([
                [0., 0.],
                [1., 0.],
                [1., 1.],
                [0., 1.],
                [-1., 1.],
                [-1., 0.],
                [-1., -1.],
                [0., -1.],
                [1., -1.],
            ], dtype=torch.float32).T
        else:
            vals = [-1, 0, 1]
            square_points = torch.tensor([
                [x, y, z] for x in vals for y in vals for z in vals
            ], dtype=torch.float32).T

        random_points = 3 * torch.randn(D, P - 1)
        
        points = torch.cat([random_points, square_points], dim=1).T

        origin = -1
        for i, p in enumerate(points):
            if np.allclose(p.numpy(), 0.):
                origin = i

        assert origin >= 0

        P = points.shape[0]
        
        features = torch.randn(P, B, in_rep.size)
        
        filters = torch.zeros((P, basis.dim, out_rep.size, in_rep.size))
        
        filters = basis.sample(points, out=filters)
        self.assertFalse(torch.isnan(filters).any())
        self.assertFalse(torch.allclose(filters, torch.zeros_like(filters)))
        
        a = basis.sample(points)
        b = basis.sample(points)
        assert torch.allclose(a, b)
        del a, b

        # for idx in range(basis.dim):
        #     b = basis[idx]
        #     if b['j'] != basis:
        #         assert np.allclose(filters[..., idx, origin], 0.), basis[idx]

        output = torch.einsum("pfoi,pbi->fbo", filters, features)
        
        for _ in range(20):
            g = group.sample()
            
            output1 = torch.einsum("oi,fbi->fbo",
                                   torch.tensor(out_rep(g), dtype=output.dtype, device=output.device),
                                   output)

            a = torch.tensor(action(g), dtype=output.dtype, device=output.device)
            transformed_points = points @ a.T
            
            transformed_filters = basis.sample(transformed_points)

            in_rep_g = torch.tensor(in_rep(g), dtype=output.dtype, device=output.device)
            transformed_features = torch.einsum("oi,pbi->pbo", in_rep_g, features)
            output2 = torch.einsum("pfoi,pbi->fbo", transformed_filters, transformed_features)

            if not torch.allclose(output1, output2, atol=5e-5, rtol=1e-4):
                print(f"{in_rep.name}, {out_rep.name}: Error at {g}")
                print(a)
                
                aerr = torch.abs(output1 - output2).cpu().detach().numpy()
                err = aerr.reshape(-1, basis.dim).max(0)
                print(basis.dim, (err > 0.01).sum(), err.max())
                print(torch.isclose(output1, output2, atol=1e-5, rtol=5e-4).logical_not().to(int).sum().item())
                for idx in range(basis.dim):
                    if err[idx] > 0.1:
                        print(idx)
                        print(err[idx])
                        print(basis[idx])

            self.assertTrue(torch.allclose(output1, output2, atol=5e-5, rtol=1e-4),
                            f"Group {group.name}, {in_rep.name} - {out_rep.name},\n"
                            f"element {g},\n"
                            f"action:\n"
                            f"{a}")
                            # f"element {g}, action {a}, {basis.b1.bases[0][0].axis}")


if __name__ == '__main__':
    unittest.main()
