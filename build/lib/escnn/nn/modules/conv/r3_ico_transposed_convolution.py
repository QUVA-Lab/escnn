
from escnn.nn import FieldType

from escnn.gspaces import GSpace3D
from escnn.group import Representation, Group, Icosahedral
from escnn.kernels import KernelBasis
from escnn.kernels import kernels_aliased_Ico_act_R3_icosahedron, kernels_aliased_Ico_act_R3_icosidodecahedron, kernels_aliased_Ico_act_R3_dodecahedron

from .r3_transposed_convolution import R3ConvTransposed

from typing import Callable, Union, List


__all__ = ["R3IcoConvTransposed"]


class R3IcoConvTransposed(R3ConvTransposed):

    def __init__(
            self,
            in_type: FieldType,
            out_type: FieldType,
            kernel_size: int,
            padding: int = 0,
            output_padding: int = 0,
            stride: int = 1,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            samples: str = 'ico',
            sigma: Union[List[float], float] = None,
            rings: List[float] = None,
            recompute: bool = False,
            basis_filter: Callable[[dict], bool] = None,
            initialize: bool = True,
    ):
        r"""


        Icosahedral-steerable volumetric convolution mapping between the input and output :class:`~escnn.nn.FieldType` s specified by
        the parameters ``in_type`` and ``out_type``.
        This operation is equivariant under the action of :math:`\R^3\rtimes I` where :math:`I`
        (the :class:`~escnn.group.Icosahedral` group) is the :attr:`escnn.nn.FieldType.fibergroup` of
        ``in_type`` and ``out_type``.

        This class is mostly similar to :class:`~escnn.nn.R3ConvTransposed`, with the only difference that it only
        supports the group :class:`~escnn.group.Icosahedral` since it uses a kernel basis which is specific for
        this group.

        The argument ``frequencies_cutoff`` of :class:`~escnn.nn.R3ConvTransposed` is not supported here since the
        steerable kernels are not generated from a band-limited set of harmonic functions.

        Instead, the argument ``samples`` specifies the polyhedron (symmetric with respect to the
        :class:`~escnn.group.Icosahedral`  group) whose vertices are used to define the kernel on :math:`\R^3`.
        The supported polyhedrons are ``"ico"`` (the 12 vertices of the icosahedron), ``"dodeca"`` (the 20 vertices
        of the dodecahedron) or ``"icosidodeca"`` (the 30 vertices of the icosidodecahedron, which correspond to the
        centers of the 30 edges of either the icosahedron or the dodecahedron).

        For each ring ``r`` in ``rings``, the polyhedron specified in embedded in the sphere of radius ``r``.
        The analytical kernel, which is only defined on the vertices of this polyhedron, is then "diffused" in the
        ambient space :math:`\R^3` by means of a small Gaussian kernel with std ``sigma``.

        .. warning ::
            Even if the input tensor has a `coords` attribute, the output of this module will not have one.

        Args:
            in_type (FieldType): the type of the input field, specifying its transformation law
            out_type (FieldType): the type of the output field, specifying its transformation law
            kernel_size (int): the size of the (square) filter
            padding(int, optional): implicit zero paddings on both sides of the input. Default: ``0``
            output_padding(int, optional): implicit zero paddings on both sides of the input. Default: ``0``
            stride(int, optional): the stride of the kernel. Default: ``1``
            dilation(int, optional): the spacing between kernel elements. Default: ``1``
            groups (int, optional): number of blocked connections from input channels to output channels.
                                    It allows depthwise convolution. When used, the input and output types need to be
                                    divisible in ``groups`` groups, all equal to each other.
                                    Default: ``1``.
            bias (bool, optional): Whether to add a bias to the output (only to fields which contain a
                    trivial irrep) or not. Default ``True``
            sigma (list or float, optional): width of each ring where the bases are sampled. If only one scalar
                    is passed, it is used for all rings.
            rings (list, optional): radii of the rings where to sample the bases
            recompute (bool, optional): if ``True``, recomputes a new basis for the equivariant kernels.
                    By Default (``False``), it  caches the basis built or reuse a cached one, if it is found.
            basis_filter (callable, optional): function which takes as input a descriptor of a basis element
                    (as a dictionary) and returns a boolean value: whether to preserve (``True``) or discard (``False``)
                    the basis element. By default (``None``), no filtering is applied.
            initialize (bool, optional): initialize the weights of the model. Default: ``True``

        """

        assert isinstance(in_type.gspace, GSpace3D)
        assert isinstance(out_type.gspace, GSpace3D)
        assert in_type.gspace == out_type.gspace
        assert isinstance(in_type.gspace.fibergroup, Icosahedral)

        assert samples in ['ico', 'dodeca', 'icosidodeca']
        self._samples = samples

        super(R3IcoConvTransposed, self).__init__(
            in_type,
            out_type,
            kernel_size,
            padding,
            output_padding,
            stride,
            dilation,
            groups,
            bias,
            sigma,
            frequencies_cutoff=5., # to avoid performing any frequency cut-off, set a large upperbound
            rings=rings,
            recompute=recompute,
            basis_filter=basis_filter,
            initialize=initialize,
        )

    def _build_kernel_basis(self, in_repr: Representation, out_repr: Representation) -> KernelBasis:
        if self._samples == 'ico':
            return kernels_aliased_Ico_act_R3_icosahedron(in_repr, out_repr, sigma=self._sigma, radii=self._rings)
        if self._samples == 'dodeca':
            return kernels_aliased_Ico_act_R3_dodecahedron(in_repr, out_repr, sigma=self._sigma, radii=self._rings)
        if self._samples == 'icosidodeca':
            return kernels_aliased_Ico_act_R3_icosidodecahedron(in_repr, out_repr, sigma=self._sigma, radii=self._rings)
        else:
            raise ValueError


