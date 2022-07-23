
from escnn.nn import GeometricTensor
from .equivariant_module import EquivariantModule

import torch

from typing import List, Tuple, Union, Any, Iterator

from collections import OrderedDict

__all__ = ["SequentialModule"]


class SequentialModule(EquivariantModule):
    
    def __init__(self,
                 *args: EquivariantModule,
                 ):
        r"""
        
        A sequential container similar to :class:`torch.nn.Sequential`.
        
        The constructor accepts both a list or an ordered dict of :class:`~escnn.nn.EquivariantModule` instances.

        The module also supports indexing, slicing and iteration.
        If slicing with a step different from 1 is used, one should ensure that adjacent modules in the new sequence
        are compatible.
        
        Example::
        
            # Example of SequentialModule
            s = escnn.gspaces.rot2dOnR2(8)
            c_in = escnn.nn.FieldType(s, [s.trivial_repr]*3)
            c_out = escnn.nn.FieldType(s, [s.regular_repr]*16)
            model = escnn.nn.SequentialModule(
                      escnn.nn.R2Conv(c_in, c_out, 5),
                      escnn.nn.InnerBatchNorm(c_out),
                      escnn.nn.ReLU(c_out),
            )

            len(module) # returns 3

            module[:2] # returns another SequentialModule containing the first two modules

            # Example with OrderedDict
            s = escnn.gspaces.rot2dOnR2(8)
            c_in = escnn.nn.FieldType(s, [s.trivial_repr]*3)
            c_out = escnn.nn.FieldType(s, [s.regular_repr]*16)
            model = escnn.nn.SequentialModule(OrderedDict([
                      ('conv', escnn.nn.R2Conv(c_in, c_out, 5)),
                      ('bn', escnn.nn.InnerBatchNorm(c_out)),
                      ('relu', escnn.nn.ReLU(c_out)),
            ]))
        
        """
        
        super(SequentialModule, self).__init__()

        self.in_type = None
        self.out_type = None
        
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                assert isinstance(module, EquivariantModule)
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                assert isinstance(module, EquivariantModule)
                self.add_module(str(idx), module)
        
        # for i in range(1, len(self._modules.values())):
        #     assert self._modules.values()[i-1].out_type == self._modules.values()[i].in_type
        
    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""
        
        Args:
            input (GeometricTensor): the input GeometricTensor

        Returns:
            the output tensor
            
        """
        
        assert input.type == self.in_type
        x = input
        for m in self._modules.values():
            x = m(x)

        assert x.type == self.out_type
        
        return x
    
    def add_module(self, name: str, module: EquivariantModule):
        r"""
        Append ``module`` to the sequence of modules applied in the forward pass.
        
        """
        
        if len(self._modules) == 0:
            assert self.in_type is None
            assert self.out_type is None
            self.in_type = module.in_type
        else:
            assert module.in_type == self.out_type, f"{module.in_type} != {self.out_type}"
            
        self.out_type = module.out_type
        super(SequentialModule, self).add_module(name, module)

    def append(self, module: EquivariantModule) -> 'SequentialModule':
        r"""Appends a new EquivariantModule at the end.
        """
        self.add_module(str(len(self)), module)
        return self

    def __getitem__(self, idx) -> Union['SequentialModule', EquivariantModule]:
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        elif isinstance(idx, int):
            assert -len(self) <= idx < len(self), (idx, len(self))
            idx = idx % len(self)
            for i, module in enumerate(self._modules.values()):
                if i == idx:
                    return module
            raise ValueError(f'Index {idx} not found!')
        else:
            raise ValueError(f'Index {idx} not valid!')

    def __iter__(self) -> Iterator[EquivariantModule]:
        return iter(self._modules.values())

    def __len__(self) -> int:
        return len(self._modules)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        
        assert len(input_shape) > 1
        assert input_shape[1] == self.in_type.size
        
        out_shape = input_shape

        for m in self._modules.values():
            out_shape = m.evaluate_output_shape(out_shape)
        
        return out_shape

    def check_equivariance(self, atol: float = 2e-6, rtol: float = 1e-5) -> List[Tuple[Any, float]]:
        return super(SequentialModule, self).check_equivariance(atol=atol, rtol=rtol)

    def export(self):
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.Sequential` module and set to "eval" mode.

        """
    
        self.eval()
    
        submodules = []
        
        # convert all the submodules if necessary
        for name, module in self._modules.items():
            if isinstance(module, EquivariantModule):
                module = module.export()
                
            submodules.append(
                (name, module)
            )

        return torch.nn.Sequential(OrderedDict(submodules))
