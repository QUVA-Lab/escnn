
from escnn.group import *
from escnn.gspaces import *
from escnn.nn import FourierFieldType, GeometricTensor

from ..equivariant_module import EquivariantModule
from ..fourier import InverseFourierTransform, FourierTransform

import torch
import torch.nn.functional as F

from typing import List, Tuple, Any, Optional

import numpy as np

__all__ = ["FourierPointwise", "FourierELU"]


class FourierPointwise(EquivariantModule):
    
    def __init__(self, in_type, *args, **kwargs):
        r"""
        
        Perform an Inverse Fourier Transform to sample the input features, apply the pointwise non-linearity in the
        group domain (Dirac-delta basis) and, finally, compute the Fourier Transform to obtain irreps coefficients.
        
        .. warning::
            This operation is only *approximately* equivariant and its equivariance depends on the sampling grid and the
            non-linear activation used, as well as the original band-limitation of the input features.
            
        The same function is applied to every channel independently.
        By default, the input representation is preserved by this operation and, therefore, it equals the output
        representation.
        Optionally, the output can have a different band-limit by using the argument ``out_type``.
        
        The class first constructs a band-limited regular representation of ```gspace.fibergroup``` using
        :meth:`escnn.group.Group.spectral_regular_representation`.
        The band-limitation of this representation is specified by ```irreps``` which should be a list containing
        a list of ids identifying irreps of ```gspace.fibergroup``` (see :attr:`escnn.group.IrreducibleRepresentation.id`).
        This representation is used to define the input and output field types, each containing ```channels``` copies
        of a feature field transforming according to this representation.
        A feature vector transforming according to such representation is interpreted as a vector of coefficients
        parameterizing a function over the group using a band-limited Fourier basis.

        .. note::
            Instead of building the list ``irreps`` manually, most groups implement a method ``bl_irreps()`` which can be
            used to generate this list with through a simpler interface. Check each group's documentation.
        
        To approximate the Fourier transform, this module uses a finite number of samples from the group.
        The set of samples used is specified by the ``grid`` argument.  This can be any list of group elements, but most
        often the :meth:`~escnn.group.Group.grid` is used to generate this list.
        
        Args:
            gspace (GSpace):  the gspace describing the symmetries of the data. The Fourier transform is
                              performed over the group ```gspace.fibergroup```
            channels (int): number of independent fields in the input `FieldType`
            irreps (list): list of irreps' ids to construct the band-limited representation
            function (str): the identifier of the non-linearity.
                    It is used to specify which function to apply.
                    By default (``'p_relu'``), ReLU is used.
            *grid_args: parameters used to construct the discretization grid
            inplace (bool): applies the non-linear activation in-place. Default: `True`
            out_irreps (list, optional): optionally, one can specify a different band-limiting in output
            normalize (bool, optional): if ``True``, the rows of the IFT matrix (and the columns of the FT matrix) are normalized. Default: ``True``
            **grid_kwargs: keyword parameters used to construct the discretization grid
            
        """
        super().__init__()

        if isinstance(in_type, FourierFieldType):
            self._init(in_type, *args, **kwargs)
        else:
            self._init_deprecated(in_type, *args, **kwargs)

    def _init(
            self,
            in_type: FourierFieldType,
            grid: List[GroupElement],
            function: str = 'p_relu',
            *,
            out_type: Optional[FourierFieldType] = None,
            inplace: bool = True,
            normalize: bool = True,
    ):
        self.in_type = in_type
        self.out_type = out_type or in_type

        self.ift = InverseFourierTransform(in_type, grid, normalize=normalize)
        self.ft = FourierTransform(grid, self.out_type, extra_irreps=in_type.bl_irreps, normalize=normalize)

        if callable(function):
            self._function = function
        elif function == 'p_relu':
            self._function = F.relu_ if inplace else F.relu
        elif function == 'p_elu':
            self._function = F.elu_ if inplace else F.elu
        elif function == 'p_sigmoid':
            self._function = torch.sigmoid_ if inplace else F.sigmoid
        elif function == 'p_tanh':
            self._function = torch.tanh_ if inplace else F.tanh
        else:
            raise ValueError('Function "{}" not recognized!'.format(function))
        
    def _init_deprecated(
            self,
            gspace: GSpace,
            channels: int,
            irreps: List,
            *grid_args,
            grid: List[GroupElement] = None,
            function: str = 'p_relu',
            inplace: bool = True,
            out_irreps: List = None,
            normalize: bool = True,
            **grid_kwargs
    ):
        from warnings import warn
        warn("The `gspace`, `channels`, and `irreps` arguments are deprecated.  Use the `in_type` argument instead.", DeprecationWarning)

        in_type = FourierFieldType(gspace, channels, irreps)
        out_type = FourierFieldType(gspace, channels, out_irreps or irreps)

        if grid is None:
            grid = gspace.fibergroup.grid(*grid_args, **grid_kwargs)

        self._init(
                in_type, grid, function,
                out_type=out_type,
                inplace=inplace,
                normalize=normalize,
        )
        
    def forward(self, x_hat: GeometricTensor) -> GeometricTensor:
        r"""

        Applies the pointwise activation function on the input fields

        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map after the non-linearities have been applied

        """

        assert x_hat.type == self.in_type

        x = self.ift(x_hat)

        x.tensor = self._function(x.tensor)

        return self.ft(x)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:

        assert len(input_shape) >= 2
        assert input_shape[1] == self.in_type.size

        b, c = input_shape[:2]
        spatial_shape = input_shape[2:]

        return (b, self.out_type.size, *spatial_shape)

    def check_equivariance(self, atol: float = 1e-5, rtol: float = 2e-2, assert_raise: bool = True) -> List[Tuple[Any, float]]:
    
        gspace = self.in_type.gspace
        rho = self.in_type.rho
        rho_out = self.out_type.rho
        c = self.in_type.size
        B = 128
        x = torch.randn(B, c, *[3]*gspace.dimensionality)

        # since we mostly use non-linearities like relu or elu, we make sure 
        # the average value of the features is
        # positive, such that, when we test inputs with only frequency 0 (or only low frequencies), the output is not
        # zero everywhere
        x = x.view(B, len(self.in_type), rho.size, *[3]*gspace.dimensionality)
        p = 0
        for irr in rho.irreps:
            irr = gspace.irrep(*irr)
            if irr.is_trivial():
                x[:, :, p] = x[:, :, p].abs()
            p+=irr.size

        x = x.view(B, self.in_type.size, *[3]*gspace.dimensionality)

        errors = []

        # for el in gspace.testing_elements:
        for _ in range(100):
            
            el = gspace.fibergroup.sample()
    
            x1 = GeometricTensor(x.clone(), self.in_type)
            x2 = GeometricTensor(x.clone(), self.in_type).transform_fibers(el)

            out1 = self(x1).transform_fibers(el)
            out2 = self(x2)

            out1 = out1.tensor.view(B, len(self.out_type), rho_out.size, *out1.shape[2:]).detach().numpy()
            out2 = out2.tensor.view(B, len(self.out_type), rho_out.size, *out2.shape[2:]).detach().numpy()

            errs = np.linalg.norm(out1 - out2, axis=2).reshape(-1)
            errs[errs < atol] = 0.
            norm = np.sqrt(np.linalg.norm(out1, axis=2).reshape(-1) * np.linalg.norm(out2, axis=2).reshape(-1))
            
            relerr = errs / norm

            # print(el, errs.max(), errs.mean(), relerr.max(), relerr.min())

            if assert_raise:
                assert relerr.mean()+ relerr.std() < rtol, \
                    'The error found during equivariance check with element "{}" is too high: max = {}, mean = {}, std ={}' \
                        .format(el, relerr.max(), relerr.mean(), relerr.std())

            # errors.append((el, errs.mean()))
            errors.append(relerr)

        # return errors
        return np.concatenate(errors).reshape(-1)

    # The following properties are just for backwards compatibility
    @property
    def space(self):
        return self.in_type.gspace

    @property
    def rho(self):
        return self.in_type.rho
    
    @property
    def rho_out(self):
        return self.out_type.rho
    


class FourierELU(FourierPointwise):
    
    def __init__(self, *args, **kwargs):
        r"""

        Applies a Inverse Fourier Transform to sample the input features, apply ELU point-wise in the
        group domain (Dirac-delta basis) and, finally, computes the Fourier Transform to obtain irreps coefficients.
        See :class:`~escnn.nn.FourierPointwise` for more details.

        Args:
            gspace (GSpace):  the gspace describing the symmetries of the data. The Fourier transform is
                              performed over the group ```gspace.fibergroup```
            channels (int): number of independent fields in the input `FieldType`
            irreps (list): list of irreps' ids to construct the band-limited representation
            *grid_args: parameters used to construct the discretization grid
            inplace (bool): applies the non-linear activation in-place. Default: `True`
            out_irreps (list, optional): optionally, one can specify a different band-limiting in output
            normalize (bool, optional): if ``True``, the rows of the IFT matrix (and the columns of the FT matrix) are normalized. Default: ``True``
            **grid_kwargs: keyword parameters used to construct the discretization grid

        """
        
        kwargs['function'] = 'p_elu'
        super().__init__(*args, **kwargs)

