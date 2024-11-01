
from escnn.group import *
from escnn.gspaces import *
from escnn.nn import FourierFieldType
from escnn.nn import GeometricTensor

from .fourier import FourierPointwise

from typing import List, Tuple, Any

__all__ = ["QuotientFourierPointwise", "QuotientFourierELU"]


class QuotientFourierPointwise(FourierPointwise):
    
    def __init__(
            self,
            gspace: GSpace,
            subgroup_id: Tuple,
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
        r"""

        Perform an Inverse Fourier Transform to sample the input features on a *quotient space* :math:`X`, apply the
        pointwise non-linearity in the spatial domain (Dirac-delta basis) and, finally, compute the Fourier Transform
        to obtain irreps coefficients.

        The quotient space used is isomorphic to :math:`X \cong G / H` where :math:`G` is ```gspace.fibergroup``` while
        :math:`H` is the subgroup of :math:`G` idenitified by ```subgroup_id```; see
        :meth:`~escnn.group.Group.subgroup` and :meth:`~escnn.group.Group.homspace`
        
        .. warning::
            This operation is only *approximately* equivariant and its equivariance depends on the sampling grid and the
            non-linear activation used, as well as the original band-limitation of the input features.
            
        The same function is applied to every channel independently.
        By default, the input representation is preserved by this operation and, therefore, it equals the output
        representation.
        Optionally, the output can have a different band-limit by using the argument ``out_irreps``.

        The class first constructs a band-limited quotient representation of ```gspace.fibergroup``` using
        :meth:`escnn.group.Group.spectral_quotient_representation`.
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
        The set of samples to be used can be specified through the parameter ```grid``` or by the ```grid_args``` and
        ```grid_kwargs``` which will then be passed to the method :meth:`~escnn.group.Group.grid`.
        
        .. warning ::
            By definition, an homogeneous space is invariant under a right action of the subgroup :math:`H`.
            That means that a feature representing a function over a homogeneous space :math:`X \cong G/H`, when
            interpreted as a function over :math:`G` (as we do here when sampling), the function will be constant along
            each coset, i.e. :math:`f(gh) = f(g)` if :math:`g \in G, h\in H`.
            An approximately uniform sampling grid over :math:`G` creates an approximately uniform grid over :math:`G/H`
            through projection but might contain redundant elements (if the grid contains :math:`g \in G`, any element
            :math:`gh` in the grid will be redundant).
            It is therefore advised to create a grid directly in the quotient space, e.g. using
            :meth:`escnn.group.SO3.sphere_grid`, :meth:`escnn.group.O3.sphere_grid`.
            We do not support yet a general method and interface to generate grids over any homogeneous space for any
            group, so you should check each group's methods.

        Args:
            gspace (GSpace):  the gspace describing the symmetries of the data. The Fourier transform is
                              performed over the group ```gspace.fibergroup```
            subgroup_id (tuple): identifier of the subgroup :math:`H` to construct the quotient space
            channels (int): number of independent fields in the input `FieldType`
            irreps (list): list of irreps' ids to construct the band-limited representation
            *grid_args: parameters used to construct the discretization grid
            grid (list, optional): list containing the elements of the group to use for sampling. Optional (default ``None``).
            function (str): the identifier of the non-linearity.
                    It is used to specify which function to apply.
                    By default (``'p_relu'``), ReLU is used.
            inplace (bool): applies the non-linear activation in-place. Default: `True`
            out_irreps (list, optional): optionally, one can specify a different band-limiting in output
            normalize (bool, optional): if ``True``, the rows of the IFT matrix (and the columns of the FT matrix) are normalized. Default: ``True``
            **grid_kwargs: keyword parameters used to construct the discretization grid
            
        """
        from warnings import warn
        warn("The `QuotientFourierPointwise` module is deprecated.  Instead, create a \"quotient\" field type by passing the `subgroup_id` argument to `FourierFieldType`, then use `FourierPointwise` on that field.", DeprecationWarning)

        in_type = FourierFieldType(
                gspace, channels, irreps,
                subgroup_id=subgroup_id,
        )
        out_type = FourierFieldType(
                gspace, channels, out_irreps or irreps,
                subgroup_id=subgroup_id,
        )

        if grid is None:
            grid = gspace.fibergroup.grid(*grid_args, **grid_kwargs)

        super().__init__(
                in_type, grid, function,
                out_type=out_type,
                inplace=inplace,
                normalize=normalize,
        )


class QuotientFourierELU(QuotientFourierPointwise):
    
    def __init__(self,
                 gspace: GSpace,
                 subgroup_id: Tuple,
                 channels: int,
                 irreps: List,
                 *grid_args,
                 grid: List[GroupElement] = None,
                 inplace: bool = True,
                 out_irreps: List = None,
                 normalize: bool = True,
                 **grid_kwargs
                 ):
        r"""

        Applies a Inverse Fourier Transform to sample the input features on a quotient space, apply ELU point-wise
        in the spatial domain (Dirac-delta basis) and, finally, computes the Fourier Transform to obtain irreps
        coefficients. See :class:`~escnn.nn.QuotientFourierPointwise` for more details.

        Args:
            gspace (GSpace):  the gspace describing the symmetries of the data. The Fourier transform is
                              performed over the group ```gspace.fibergroup```
            subgroup_id (tuple): identifier of the subgroup :math:`H` to construct the quotient space
            channels (int): number of independent fields in the input `FieldType`
            irreps (list): list of irreps' ids to construct the band-limited representation
            *grid_args: parameters used to construct the discretization grid
            grid (list, optional): list containing the elements of the group to use for sampling. Optional (default ``None``).
            inplace (bool): applies the non-linear activation in-place. Default: ``True``
            out_irreps (list, optional): optionally, one can specify a different band-limiting in output
            normalize (bool, optional): if ``True``, the rows of the IFT matrix (and the columns of the FT matrix) are normalized. Default: ``True``
            **grid_kwargs: keyword parameters used to construct the discretization grid

        """
        
        super(QuotientFourierELU, self).__init__(
            gspace, subgroup_id, channels, irreps, *grid_args,
            function='p_elu',
            inplace=inplace,
            grid=grid,
            out_irreps=out_irreps,
            normalize=normalize,
            **grid_kwargs
        )

