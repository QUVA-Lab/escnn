

import escnn.nn
from escnn.nn import FieldType, GeometricTensor
from escnn.group import Representation
from escnn.kernels import KernelBasis

from .rd_point_convolution import _RdPointConv

from typing import Callable, Tuple, Dict, Union, List

import torch
import numpy as np


import math


__all__ = ["R2PointConv"]


class R2PointConv(_RdPointConv):
    
    def __init__(self,
                 in_type: FieldType,
                 out_type: FieldType,
                 groups: int = 1,
                 bias: bool = True,
                 sigma: Union[List[float], float] = None,
                 width: float = None,
                 n_rings: int = None,
                 frequencies_cutoff: Union[float, Callable[[float], int]] = None,
                 rings: List[float] = None,
                 basis_filter: Callable[[dict], bool] = None,
                 recompute: bool = False,
                 initialize: bool = True,
                 ):
        r"""

        G-steerable planar convolution mapping between the input and output :class:`~escnn.nn.FieldType` s specified by
        the parameters ``in_type`` and ``out_type``.
        This operation is equivariant under the action of :math:`\R^2\rtimes G` where :math:`G` is the
        :attr:`escnn.nn.FieldType.fibergroup` of ``in_type`` and ``out_type``.

        This class implements convolution with steerable filters over sparse planar geometric graphs.
        Instead, :class:`~escnn.nn.R2Conv` implements an equivalent convolution layer over a pixel grid.
        See the documentation of :class:`~escnn.nn.R2Conv` for more details about equivariance and steerable convolution.

        The input of this module is a planar geometric graph, i.e. a graph whose nodes are associated with 2D
        coordinates in :math:`\R^2`.
        The nodes' coordinates should be stored in the ``coords`` attribute of the input
        :class:`~escnn.nn.GeometricTensor`.
        The adjacency of the graph should be passed as a second input tensor ``edge_index``, like commonly done in
        :class:`~torch_geometric.nn.conv.message_passing.MessagePassing`.
        See :meth:`~escnn.nn.modules.pointconv._RdPointConv.forward`.

        In each forward pass, the module computes the relative coordinates of the points on the edges and samples each
        filter in the basis of G-steerable kernels at these relative locations.
        The basis filters are expanded using the learnable weights and used to perform convolution over the graph in the
        message passing framework.
        Optionally, the relative coordinates can be pre-computed and passed in the input ``edge_delta`` tensor.

        .. note ::
            In practice, we first apply the basis filters on the input features and then combine the responses via the
            learnable weights. See also :meth:`~escnn.nn.modules.basismanager.BlocksBasisSampler.compute_messages`.

        .. warning ::

            When :meth:`~torch.nn.Module.eval()` is called, the bias is built with the current trained weights and stored
            for future reuse such that no overhead of expanding the bias remains.

            When :meth:`~torch.nn.Module.train()` is called, the attribute :attr:`~escnn.nn.R2PointConv.expanded_bias`
            is discarded to avoid situations of mismatch with the learnable expansion coefficients.
            See also :meth:`escnn.nn.R2PointConv.train`.

            This behaviour can cause problems when storing the :meth:`~torch.nn.Module.state_dict` of a model while in
            a mode and lately loading it in a model with a different mode, as the attributes of the class change.
            To avoid this issue, we recommend converting the model to eval mode before storing or loading the state
            dictionary.

        The learnable expansion coefficients of this module can be initialized with the methods in
        :mod:`escnn.nn.init`.
        By default, the weights are initialized in the constructors using :func:`~escnn.nn.init.generalized_he_init`.

        .. warning ::

            This initialization procedure can be *extremely* slow for wide layers.
            In case initializing the model is not required (e.g. before loading the state dict of a pre-trained model)
            or another initialization method is preferred (e.g. :func:`~escnn.nn.init.deltaorthonormal_init`), the
            parameter ``initialize`` can be set to ``False`` to avoid unnecessary overhead.

        The parameters ``width``, ``n_rings``, ``sigma``, ``frequencies_cutoff``, ``rings`` are
        optional parameters used to control how the basis for the filters is built, how it is sampled and how it is
        expanded to build the filter.
        In practice, steerable filters are parameterized independently on a number of concentric rings by using
        *circular harmonics*.
        These rings can be specified by *i)* either using the list ``rings``, which defines the radii of each ring, or
        by *ii)* indicating the maximum radius ``width`` and the number ``n_rings`` of rings to include.
        ``sigma`` defines the "thickness" of each ring as the standard deviation of a Gaussian bell along the radial
        direction.
        ``frequencies_cutoff`` is a function defining the maximum frequency of the circular harmonics allowed at each
        radius. If a `float` value is passed, the maximum frequency is equal to the radius times this factor.

        .. note ::
            These parameters should be carefully tuned depending on the typical connectivity and the scale of the
            geometric graphs of interest.

        .. warning ::
            We don't support ``groups > 1`` yet.
            We include this parameter for future compatibility.

        Args:
            in_type (FieldType): the type of the input field, specifying its transformation law
            out_type (FieldType): the type of the output field, specifying its transformation law
            groups (int, optional): number of blocked connections from input channels to output channels.
                                    It allows depthwise convolution. When used, the input and output types need to be
                                    divisible in ``groups`` groups, all equal to each other.
                                    Default: ``1``.
            bias (bool, optional): Whether to add a bias to the output (only to fields which contain a
                    trivial irrep) or not. Default ``True``
            sigma (list or float, optional): width of each ring where the bases are sampled. If only one scalar
                    is passed, it is used for all rings.
            width (float, optional): radius of the support of the learnable filters.
                    Setting ``n_rings`` and ``width`` is an alternative to use ``rings``.
            n_rings (int, optional): number of (equally spaced) rings the support of the filters is split into.
                    Setting ``n_rings`` and ``width`` is an alternative to use ``rings``.
            frequencies_cutoff (callable or float, optional): function mapping the radii of the basis elements to the
                    maximum frequency accepted. If a float values is passed, the maximum frequency is equal to the
                    radius times this factor. Default: ``3.``.
            rings (list, optional): radii of the rings where to sample the bases
            basis_filter (callable, optional): function which takes as input a descriptor of a basis element
                    (as a dictionary) and returns a boolean value: whether to preserve (``True``) or discard (``False``)
                    the basis element. By default (``None``), no filtering is applied.
            recompute (bool, optional): if ``True``, recomputes a new basis for the equivariant kernels.
                    By Default (``False``), it  caches the basis built or reuse a cached one, if it is found.
            initialize (bool, optional): initialize the weights of the model. Default: ``True``

        Attributes:

            ~.weights (torch.Tensor): the learnable parameters which are used to expand the kernel
            ~.bias (torch.Tensor): the learnable parameters which are used to expand the bias, if ``bias=True``
            ~.expanded_bias (torch.Tensor): the equivariant bias which is summed to the output, obtained by expanding
                                    the parameters in :attr:`~escnn.nn.R2PointConv.bias`

        """

        basis_filter, self._rings, self._sigma, self._maximum_frequency = compute_basis_params(
            frequencies_cutoff, rings, sigma, width, n_rings, basis_filter
        )

        super(R2PointConv, self).__init__(
            in_type, out_type,
            d=2,
            groups=groups,
            bias=bias,
            basis_filter=basis_filter,
            recompute=recompute
        )

        if initialize:
            # by default, the weights are initialized with a generalized form of He's weight initialization
            escnn.nn.init.generalized_he_init(self.weights.data, self.basissampler)

    def _build_kernel_basis(self, in_repr: Representation, out_repr: Representation) -> KernelBasis:
        return self.space.build_kernel_basis(in_repr, out_repr,
                                             self._sigma, self._rings,
                                             maximum_frequency=self._maximum_frequency
                                             )


def bandlimiting_filter(frequency_cutoff: Union[float, Callable[[float], float]]) -> Callable[[dict], bool]:
    r"""

    Returns a method which takes as input the attributes (as a dictionary) of a basis element and returns a boolean
    value: whether to preserve that element (true) or not (false)

    If the parameter ``frequency_cutoff`` is a scalar value, the maximum frequency allowed at a certain radius is
    proportional to the radius itself. in thi case, the parameter ``frequency_cutoff`` is the factor controlling this
    proportionality relation.

    If the parameter ``frequency_cutoff`` is a callable, it needs to take as input a radius (a scalar value) and return
    the maximum frequency which can be sampled at that radius.

    args:
        frequency_cutoff (float): factor controlling the bandlimiting

    returns:
        a function which checks the attributes of individual basis elements and chooses whether to discard them or not

    """

    if isinstance(frequency_cutoff, float):
        frequency_cutoff = lambda r, fco=frequency_cutoff: r * frequency_cutoff

    def bl_filter(attributes: dict) -> bool:
        return math.fabs(attributes["irrep:frequency"]) <= frequency_cutoff(attributes["radius"])

    return bl_filter


def compute_basis_params(
        frequencies_cutoff: Union[float, Callable[[float], float]] = None,
        rings: List[float] = None,
        sigma: List[float] = None,
        width: float = None,
        n_rings: int = None,
        custom_basis_filter: Callable[[dict], bool] = None,
):

    assert (width is not None and n_rings is not None) != (rings is not None)

    # by default, the number of rings equals half of the filter size
    if rings is None:
        assert width > 0.
        assert n_rings > 0
        rings = torch.linspace(0, width, n_rings)
        rings = rings.tolist()

    if sigma is None:
        sigma = [0.6] * (len(rings) - 1) + [0.4]
        for i, r in enumerate(rings):
            if r == 0.:
                sigma[i] = 0.005
    elif isinstance(sigma, float):
        sigma = [sigma] * len(rings)

    if frequencies_cutoff is None:
        frequencies_cutoff = 3.

    if isinstance(frequencies_cutoff, float):
        frequencies_cutoff = lambda r, fco=frequencies_cutoff: fco * r

    # check if the object is a callable function
    assert callable(frequencies_cutoff)

    maximum_frequency = int(max(frequencies_cutoff(r) for r in rings))

    fco_filter = bandlimiting_filter(frequencies_cutoff)

    if custom_basis_filter is not None:
        basis_filter = lambda d, custom_basis_filter=custom_basis_filter, fco_filter=fco_filter: (
                custom_basis_filter(d) and fco_filter(d)
        )
    else:
        basis_filter = fco_filter

    return basis_filter, rings, sigma, maximum_frequency

