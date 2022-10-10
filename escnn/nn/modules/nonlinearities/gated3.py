

from typing import List, Tuple, Any, Callable

import numpy as np

from escnn.group import *
from escnn.gspaces import *
from escnn.nn import FieldType
from escnn.nn import GeometricTensor

from ..equivariant_module import EquivariantModule

import torch

from torch.nn import Parameter


__all__ = ["GatedNonLinearityUniform"]


class GatedNonLinearityUniform(EquivariantModule):
    
    def __init__(self,
                 in_type: FieldType,
                 gate: Callable = torch.sigmoid,
                 ):
        r"""
        
        Gated non-linearities.
        This module applies a bias and a sigmoid function of the gates fields and, then, multiplies each gated
        field by one of the gates.
        
        The input representation of the gated features is preserved by this operation while the gate fields are
        discarded.

        This module is a less general-purpose version of the other Gated-Non-Linearity modules, optimized for
        *uniform* field types.
        This module assumes that the input type contains only copies of the same field (same representation) and that
        such field internally contains a trivial representation for each other irrep in it.

        This means that the number of irreps in the representation must be even and that the first half of them need to
        be trivial representations.

        The input representation is also assumed to have no change of basis, i.e. its change-of-basis must be equal to
        the identity matrix.

        .. note::
            The documentation of this method is still work in progress.

        Args:
            in_type (FieldType): the input field type
            gate (optional, Callable): the gate fucntion to apply. By default, it is the sigmoid function.

        """

        assert isinstance(in_type.gspace, GSpace)

        assert in_type.uniform

        rho = in_type.representations[0]

        assert len(rho.irreps) % 2 == 0
        assert np.allclose(rho.change_of_basis, np.eye(rho.size))

        N = len(rho.irreps)

        gates = rho.irreps[:N//2]
        gated = rho.irreps[N//2:]

        G = rho.group

        for gate_irr in gates:
            assert G.irrep(*gate_irr).is_trivial()

        out_rho = directsum([G.irrep(*irr) for irr in gated])

        super(GatedNonLinearityUniform, self).__init__()

        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = self.space.type(*[out_rho]*len(in_type))

        self.rho_in = self.in_type.representations[0]
        self.rho_out = self.out_type.representations[0]
        self.n_gates = N//2

        self.gate_function = gate

        expansion = torch.zeros(out_rho.size, N//2, dtype=torch.float)

        p = 0
        for i, gated_irr in enumerate(gated):
            gated_irr = G.irrep(*gated_irr)
            expansion[p:p+gated_irr.size, i] = 1.
            p += gated_irr.size
        self.register_buffer('expansion', expansion)

        # the bias for the gates
        self.bias = Parameter(torch.randn(1, len(self.in_type), self.n_gates, dtype=torch.float), requires_grad=True)

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""
        
        Apply the gated non-linearity to the input feature map.
        
        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map
            
        """
        
        assert isinstance(input, GeometricTensor)
        assert input.type == self.in_type

        b, c = input.shape[:2]
        spatial_shape = input.shape[2:]

        coords = input.coords

        input = input.tensor.view(b, len(self.in_type), self.rho_in.size, *spatial_shape)

        gates = input[:, :, :self.n_gates, ...]
        features = input[:, :, self.n_gates:, ...]

        # transform the gates
        gates = self.gate_function(gates - self.bias.view(1, len(self.in_type), self.n_gates, *[1]*len(spatial_shape)))

        expanded_gates = torch.einsum('oi,bci...->bco...', self.expansion, gates)

        assert expanded_gates.shape == features.shape, (expanded_gates.shape, features.shape)

        output = expanded_gates * features

        output = output.reshape(b, self.out_type.size, *spatial_shape)

        # wrap the result in a GeometricTensor
        return GeometricTensor(output, self.out_type, coords)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        
        assert len(input_shape) >= 2
        assert input_shape[1] == self.in_type.size
    
        b, c = input_shape[:2]
        spatial_shape = input_shape[2:]

        return (b, self.out_type.size, *spatial_shape)
        
    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
    
        c = self.in_type.size

        x = torch.randn(3, c, *[10]*self.in_type.gspace.dimensionality)

        x = GeometricTensor(x, self.in_type)

        errors = []

        for el in self.space.testing_elements:
            out1 = self(x).transform_fibers(el)
            out2 = self(x.transform_fibers(el))
    
            errs = (out1.tensor - out2.tensor).detach().numpy()
            errs = np.abs(errs).reshape(-1)
            print(el, errs.max(), errs.mean(), errs.var())
    
            assert torch.allclose(out1.tensor, out2.tensor, atol=atol, rtol=rtol), \
                'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}' \
                    .format(el, errs.max(), errs.mean(), errs.var())
    
            errors.append((el, errs.mean()))

        return errors

