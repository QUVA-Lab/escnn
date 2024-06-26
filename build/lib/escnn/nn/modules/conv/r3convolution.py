from torch.nn.functional import conv3d, pad

import escnn.nn
from escnn.nn import FieldType
from escnn.nn import GeometricTensor

from escnn.gspaces import GSpace3D
from escnn.group import Representation, Group
from escnn.kernels import KernelBasis

from .rd_convolution import _RdConv

from typing import Callable, Union, List

import torch
import numpy as np
import math


__all__ = ["R3Conv"]


class R3Conv(_RdConv):
    
    def __init__(self,
                 in_type: FieldType,
                 out_type: FieldType,
                 kernel_size: int,
                 padding: int = 0,
                 stride: int = 1,
                 dilation: int = 1,
                 padding_mode: str = 'zeros',
                 groups: int = 1,
                 bias: bool = True,
                 sigma: Union[List[float], float] = None,
                 frequencies_cutoff: Union[float, Callable[[float], int]] = None,
                 rings: List[float] = None,
                 recompute: bool = False,
                 basis_filter: Callable[[dict], bool] = None,
                 initialize: bool = True,
                 ):
        r"""


        G-steerable planar convolution mapping between the input and output :class:`~escnn.nn.FieldType` s specified by
        the parameters ``in_type`` and ``out_type``.
        This operation is equivariant under the action of :math:`\R^3\rtimes G` where :math:`G` is the
        :attr:`escnn.nn.FieldType.fibergroup` of ``in_type`` and ``out_type``.

        Specifically, let :math:`\rho_\text{in}: G \to \GL{\R^{c_\text{in}}}` and
        :math:`\rho_\text{out}: G \to \GL{\R^{c_\text{out}}}` be the representations specified by the input and output
        field types.
        Then :class:`~escnn.nn.R3Conv` guarantees an equivariant mapping

        .. math::
            \kappa \star [\mathcal{T}^\text{in}_{g,u} . f] = \mathcal{T}^\text{out}_{g,u} . [\kappa \star f] \qquad\qquad \forall g \in G, u \in \R^3

        where the transformation of the input and output fields are given by

        .. math::
            [\mathcal{T}^\text{in}_{g,u} . f](x) &= \rho_\text{in}(g)f(g^{-1} (x - u)) \\
            [\mathcal{T}^\text{out}_{g,u} . f](x) &= \rho_\text{out}(g)f(g^{-1} (x - u)) \\

        The equivariance of G-steerable convolutions is guaranteed by restricting the space of convolution kernels to an
        equivariant subspace.
        As proven in `3D Steerable CNNs <https://arxiv.org/abs/1807.02547>`_, this parametrizes the *most general
        equivariant convolutional map* between the input and output fields.

        .. warning ::

            This class implements a *discretized* convolution operator over a discrete grid.
            This means that equivariance to continuous symmetries is *not* perfect.
            In practice, by using sufficiently band-limited filters, the equivariance error introduced by the
            discretization of the filters and the features is contained, but some design choices may have a negative
            effect on the overall equivariance of the architecture.

            We provide some :doc:`practical notes <conv_notes>` on using this discretized
            convolution module.

        During training, in each forward pass the module expands the basis of G-steerable kernels with learned weights
        before calling :func:`torch.nn.functional.conv3d`.
        When :meth:`~torch.nn.Module.eval()` is called, the filter is built with the current trained weights and stored
        for future reuse such that no overhead of expanding the kernel remains.

        .. warning ::

            When :meth:`~torch.nn.Module.train()` is called, the attributes :attr:`~escnn.nn.R3Conv.filter` and
            :attr:`~escnn.nn.R3Conv.expanded_bias` are discarded to avoid situations of mismatch with the
            learnable expansion coefficients.
            See also :meth:`escnn.nn.R3Conv.train`.

            This behaviour can cause problems when storing the :meth:`~torch.nn.Module.state_dict` of a model while in
            a mode and lately loading it in a model with a different mode, as the attributes of the class change.
            To avoid this issue, we recommend converting the model to eval mode before storing or loading the state
            dictionary.


        The learnable expansion coefficients of the this module can be initialized with the methods in
        :mod:`escnn.nn.init`.
        By default, the weights are initialized in the constructors using :func:`~escnn.nn.init.generalized_he_init`.

        .. warning ::

            This initialization procedure can be extremely slow for wide layers.
            In case initializing the model is not required (e.g. before loading the state dict of a pre-trained model)
            or another initialization method is preferred (e.g. :func:`~escnn.nn.init.deltaorthonormal_init`), the
            parameter ``initialize`` can be set to ``False`` to avoid unnecessary overhead.
            See also `this issue <https://github.com/QUVA-Lab/escnn/issues/54>`_


        The parameters ``sigma``, ``frequencies_cutoff`` and ``rings`` are
        optional parameters used to control how the basis for the filters is built, how it is sampled on the filter
        grid and how it is expanded to build the filter. We suggest to keep these default values.

        .. warning ::
            Even if the input tensor has a `coords` attribute, the output of this module will not have one.

        Args:
            in_type (FieldType): the type of the input field, specifying its transformation law
            out_type (FieldType): the type of the output field, specifying its transformation law
            kernel_size (int): the size of the (square) filter
            padding(int, optional): implicit zero paddings on both sides of the input. Default: ``0``
            stride(int, optional): the stride of the kernel. Default: ``1``
            dilation(int, optional): the spacing between kernel elements. Default: ``1``
            padding_mode(str, optional): ``zeros``, ``reflect``, ``replicate`` or ``circular``. Default: ``zeros``
            groups (int, optional): number of blocked connections from input channels to output channels.
                                    It allows depthwise convolution. When used, the input and output types need to be
                                    divisible in ``groups`` groups, all equal to each other.
                                    Default: ``1``.
            bias (bool, optional): Whether to add a bias to the output (only to fields which contain a
                    trivial irrep) or not. Default ``True``
            sigma (list or float, optional): width of each ring where the bases are sampled. If only one scalar
                    is passed, it is used for all rings.
            frequencies_cutoff (callable or float, optional): function mapping the radii of the basis elements to the
                    maximum frequency accepted. If a float values is passed, the maximum frequency is equal to the
                    radius times this factor. By default (``None``), a more complex policy is used.
            rings (list, optional): radii of the rings where to sample the bases
            recompute (bool, optional): if ``True``, recomputes a new basis for the equivariant kernels.
                    By Default (``False``), it  caches the basis built or reuse a cached one, if it is found.
            basis_filter (callable, optional): function which takes as input a descriptor of a basis element
                    (as a dictionary) and returns a boolean value: whether to preserve (``True``) or discard (``False``)
                    the basis element. By default (``None``), no filtering is applied.
            initialize (bool, optional): initialize the weights of the model. Default: ``True``

        Attributes:

            ~.weights (torch.Tensor): the learnable parameters which are used to expand the kernel
            ~.filter (torch.Tensor): the convolutional kernel obtained by expanding the parameters
                                    in :attr:`~escnn.nn.R3Conv.weights`
            ~.bias (torch.Tensor): the learnable parameters which are used to expand the bias, if ``bias=True``
            ~.expanded_bias (torch.Tensor): the equivariant bias which is summed to the output, obtained by expanding
                                    the parameters in :attr:`~escnn.nn.R3Conv.bias`

        """

        assert isinstance(in_type.gspace, GSpace3D)
        assert isinstance(out_type.gspace, GSpace3D)

        basis_filter, self._rings, self._sigma, self._maximum_frequency = compute_basis_params(
            kernel_size, frequencies_cutoff, rings, sigma, dilation, basis_filter
        )

        super(R3Conv, self).__init__(
            in_type,
            out_type,
            3,
            kernel_size,
            padding,
            stride,
            dilation,
            padding_mode,
            groups,
            bias,
            basis_filter,
            recompute,
        )
        
        if initialize:
            # by default, the weights are initialized with a generalized form of He's weight initialization
            escnn.nn.init.generalized_he_init(self.weights.data, self.basisexpansion)

    def _build_kernel_basis(self, in_repr: Representation, out_repr: Representation) -> KernelBasis:
        return self.space.build_kernel_basis(in_repr, out_repr, self._sigma, self._rings,
                                             maximum_frequency=self._maximum_frequency
                                             )
    
    def forward(self, input: GeometricTensor):
        r"""
        Convolve the input with the expanded filter and bias.

        Args:
            input (GeometricTensor): input feature field transforming according to ``in_type``

        Returns:
            output feature field transforming according to ``out_type``

        """
        
        assert input.type == self.in_type
        
        if not self.training:
            _filter = self.filter
            _bias = self.expanded_bias
        else:
            # retrieve the filter and the bias
            _filter, _bias = self.expand_parameters()
        
        # use it for convolution and return the result
        if self.padding_mode == 'zeros':
            output = conv3d(input.tensor, _filter,
                            stride=self.stride,
                            padding=self.padding,
                            dilation=self.dilation,
                            groups=self.groups,
                            bias=_bias)
        else:
            output = conv3d(pad(input.tensor, self._reversed_padding_repeated_twice, self.padding_mode),
                            _filter,
                            stride=self.stride,
                            dilation=self.dilation,
                            groups=self.groups,
                            bias=_bias)

        return GeometricTensor(output, self.out_type, coords=None)
    
    def check_equivariance(self, atol: float = 0.1, rtol: float = 0.1, assertion: bool = True, verbose: bool = True, device: str = 'cpu'):

        # np.set_printoptions(precision=5, threshold=30 *self.in_type.size**2, suppress=False, linewidth=30 *self.in_type.size**2)

        feature_map_size = 9
        last_downsampling = 3
        first_downsampling = 3

        initial_size = (feature_map_size * last_downsampling - 1 + self.kernel_size - 2*self.padding) * first_downsampling

        c = self.in_type.size

        from tqdm import tqdm
        from skimage.transform import resize

        import scipy
        x = scipy.datasets.face().transpose((2, 0, 1))[np.newaxis, 0:c, :, :]

        x = resize(
            x,
            (x.shape[0], x.shape[1], initial_size, initial_size, initial_size),
            anti_aliasing=True
        )

        assert x.shape[0] == 1, x.shape

        x = x / 255.0 - 0.5

        if x.shape[1] < c:
            to_stack = [x for i in range(c // x.shape[1])]
            if c % x.shape[1] > 0:
                to_stack += [x[:, :(c % x.shape[1]), ...]]

            x = np.concatenate(to_stack, axis=1)

        x = torch.FloatTensor(x)
        x = self.in_type(x)

        def shrink(t: GeometricTensor, s) -> GeometricTensor:
            # return GeometricTensor(torch.FloatTensor(block_reduce(t.tensor.detach().numpy(), s, func=np.mean)), t.type)
            return t.type(torch.nn.functional.avg_pool3d(t.tensor, kernel_size=(s, s, s), stride=s, padding=0))

        with torch.no_grad():
            self.to(device)

            gx = self.in_type(torch.cat([x.transform(el).tensor for el in self.space.testing_elements], dim=0))

            gx = gx.to(device)
            gx = shrink(gx, first_downsampling)
            assert gx.shape[-3:] == (initial_size // first_downsampling,) * 3, (gx.shape, initial_size // first_downsampling)
            outs_2 = self(gx)
            outs_2 = shrink(outs_2, last_downsampling)
            outs_2 = outs_2.tensor.detach().cpu().numpy()
            assert outs_2.shape[-3:] == (feature_map_size, ) * 3, (outs_2.shape, feature_map_size)

            out_1 = shrink(x.to(device), first_downsampling)
            assert out_1.shape[-3:] == (initial_size // first_downsampling,) * 3, (out_1.shape, initial_size // first_downsampling)
            out_1 = self(out_1).to('cpu')
            outs_1 = torch.cat([out_1.transform(el).tensor for el in self.space.testing_elements], dim=0)
            del out_1
            outs_1 = shrink(self.out_type(outs_1.to(device)), last_downsampling).tensor.detach().cpu().numpy()
            assert outs_1.shape[-3:] == (feature_map_size, ) * 3, (outs_1.shape, feature_map_size)

            errors = []

            for i, el in tqdm(enumerate(self.space.testing_elements)):

                # out1 = shrink(out_1.transform(el), last_downsampling).tensor.detach().numpy()

                out1 = outs_1[i:i+1]
                out2 = outs_2[i:i+1]

                b, c, h, w, d = out2.shape

                center_mask = np.stack(np.meshgrid(*[np.arange(0, _w) - _w // 2 for _w in [h, w, d]]), axis=0)
                assert center_mask.shape == (3, h, w, d), (center_mask.shape, h, w, d)
                center_mask = center_mask[0, :, :] ** 2 + center_mask[1, :, :] ** 2 + center_mask[2, :, :] ** 2 < (h / 4) ** 2

                out1 = out1[..., center_mask]
                out2 = out2[..., center_mask]

                out1 = out1.reshape(-1)
                out2 = out2.reshape(-1)

                errs = np.abs(out1 - out2)

                esum = np.maximum(np.abs(out1), np.abs(out2))
                esum[esum == 0.0] = 1

                relerr = errs / esum

                if verbose:
                    print(el, relerr.max(), relerr.mean(), relerr.var(), errs.max(), errs.mean(), errs.var())

                tol = rtol * esum + atol

                if np.any(errs > tol) and verbose:
                    print(out1[errs > tol])
                    print(out2[errs > tol])
                    print(tol[errs > tol])

                if assertion:
                    assert np.all(
                        errs < tol), 'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}'.format(
                        el, errs.max(), errs.mean(), errs.var())

                errors.append((el, errs.mean()))

        return errors

        # init.deltaorthonormal_init(self.weights.data, self.basisexpansion)
        # filter = self.basisexpansion()
        # center = self.s // 2
        # filter = filter[..., center, center]
        # assert torch.allclose(torch.eye(filter.shape[1]), filter.t() @ filter, atol=3e-7)

    def export(self):
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.Conv3d` module and set to "eval" mode.

        """
    
        # set to eval mode so the filter and the bias are updated with the current
        # values of the weights
        self.eval()
        _filter = self.filter
        _bias = self.expanded_bias
    
        # build the PyTorch Conv3d module
        has_bias = self.bias is not None
        conv = torch.nn.Conv3d(self.in_type.size,
                               self.out_type.size,
                               self.kernel_size,
                               padding=self.padding,
                               stride=self.stride,
                               dilation=self.dilation,
                               groups=self.groups,
                               bias=has_bias)
    
        # set the filter and the bias
        conv.weight.data = _filter.data
        if has_bias:
            conv.bias.data = _bias.data
    
        return conv


def bandlimiting_filter(frequency_cutoff: Union[float, Callable[[float], float]]) -> Callable[[dict], bool]:
    r"""

    returns a method which takes as input the attributes (as a dictionary) of a basis element and returns a boolean
    value: whether to preserve that element (true) or not (false)

    if the parameter ``frequency_cutoff`` is a scalar value, the maximum frequency allowed at a certain radius is
    proportional to the radius itself. in thi case, the parameter ``frequency_cutoff`` is the factor controlling this
    proportionality relation.

    if the parameter ``frequency_cutoff`` is a callable, it needs to take as input a radius (a scalar value) and return
    the maximum frequency which can be sampled at that radius.

    args:
        frequency_cutoff (float): factor controlling the bandlimiting

    returns:
        a function which checks the attributes of individual basis elements and chooses whether to discard them or not

    """
    
    if isinstance(frequency_cutoff, float):
        frequency_cutoff = lambda r, fco=frequency_cutoff: r * frequency_cutoff
    
    def filter(attributes: dict) -> bool:
        return math.fabs(attributes["irrep:frequency"]) <= frequency_cutoff(attributes["radius"])
    
    return filter


def compute_basis_params(
        kernel_size: int,
        frequencies_cutoff: Union[float, Callable[[float], float]] = None,
        rings: List[float] = None,
        sigma: List[float] = None,
        dilation: int = 1,
        custom_basis_filter: Callable[[dict], bool] = None,
):
    width = dilation * (kernel_size - 1) / 2
    max_radius = width * np.sqrt(3)
    
    # by default, the number of rings equals half of the filter size
    if rings is None:
        n_rings = math.ceil(kernel_size / 2)
        rings = torch.linspace(0, (kernel_size - 1) // 2, n_rings) * dilation
        rings = rings.tolist()
    
    assert all([max_radius >= r >= 0 for r in rings])
    
    if sigma is None:
        sigma = [0.6] * (len(rings) - 1) + [0.4]
        for i, r in enumerate(rings):
            if r == 0.:
                sigma[i] = 0.005
    
    elif isinstance(sigma, float):
        sigma = [sigma] * len(rings)
    
    if frequencies_cutoff is None:
        frequencies_cutoff = 'default1'
    
    if isinstance(frequencies_cutoff, float) and frequencies_cutoff >= 0.:
        frequencies_cutoff = lambda r, fco=frequencies_cutoff: fco * r
    elif frequencies_cutoff == 'default1':
        frequencies_cutoff = _manual_fco1(kernel_size // 2)
    elif frequencies_cutoff == 'default2':
        frequencies_cutoff = _manual_fco2(kernel_size // 2)
    elif frequencies_cutoff == 'default3':
        frequencies_cutoff = _manual_fco3(kernel_size // 2)
    elif not callable(frequencies_cutoff):
        raise ValueError(f"Frequency cut-off policy '{frequencies_cutoff}' not recognized.")

    # check if the object is a callable function
    assert callable(frequencies_cutoff)
    
    maximum_frequency = int(max(frequencies_cutoff(r) for r in rings))
    
    fco_filter = bandlimiting_filter(frequencies_cutoff)
    
    if custom_basis_filter is not None:
        basis_filter = lambda d, custom_basis_filter=custom_basis_filter, fco_filter=fco_filter: (
                custom_basis_filter(d) and fco_filter(d))
    else:
        basis_filter = fco_filter
    
    return basis_filter, rings, sigma, maximum_frequency


def _manual_fco3(max_radius: float) -> Callable[[float], float]:
    r"""

    Returns a method which takes as input the radius of a ring and returns the maximum frequency which can be sampled
    on that ring.

    Args:
        max_radius (float): radius of the last ring touching the border of the grid

    Returns:
        a function which checks the attributes of individual basis elements and chooses whether to discard them or not

    """
    
    def filter(r: float) -> float:
        max_freq = 0 if r == 0. else 1 if r == max_radius else 2
        return max_freq
    
    return filter


def _manual_fco2(max_radius: float) -> Callable[[float], float]:
    r"""

    Returns a method which takes as input the radius of a ring and returns the maximum frequency which can be sampled
    on that ring.

    Args:
        max_radius (float): radius of the last ring touching the border of the grid

    Returns:
        a function which checks the attributes of individual basis elements and chooses whether to discard them or not

    """
    
    def filter(r: float) -> float:
        max_freq = 0 if r == 0. else min(2 * r, 1 if r == max_radius else 2 * r - (r + 1) % 2)
        return max_freq
    
    return filter


def _manual_fco1(max_radius: float) -> Callable[[float], float]:
    r"""

    Returns a method which takes as input the radius of a ring and returns the maximum frequency which can be sampled
    on that ring.

    Args:
        max_radius (float): radius of the last ring touching the border of the grid

    Returns:
        a function which checks the attributes of individual basis elements and chooses whether to discard them or not

    """
    
    def filter(r: float, max_radius=max_radius) -> float:
        max_freq = min(2*r, max_radius - r + 2)
        return max_freq
    
    return filter



