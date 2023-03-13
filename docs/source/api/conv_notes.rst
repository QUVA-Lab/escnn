
Practical Notes on Steerable Dense Convolution
----------------------------------------------

In this section, we include a few notes and practical tips for using
:ref:`discretized convolution layers <steerable-dense-conv>`.

These modules implement **discretized convolution** operators over discrete grids.
This means that **equivariance** to continuous symmetries is **not perfect**.
In practice, by using sufficiently band-limited filters, the equivariance error introduced by the
discretization of the filters and the features is contained, but some design choices may have a negative
effect on the overall equivariance of the architecture.

.. contents::
    :local:
    :backlinks: top

Data augmentation to prevent propagation of equivariance error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unfortunately, the small equivariance error can be leveraged during optimization to break the equivariance in case the
training data does not present the chosen symmetry.
The use of padding can have a similar effect, allowing the neural network to detect the vertical and horizontal axis.
For these reasons, **we recommend to use data augmentation** when training models equivariant to continuous symmetries
on discretized data.


Downsampling can break equivariance to the exact symmetries of the grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the architecture involves **downsampling** of intermediate features (e.g. via *strided convolution*),
some design choices **may also break the equivariance** to the discrete symmetries of the grid.
For instance, using odd-sized convolutional filters with ``stride=2`` over an even-sized image will break the
equivariance to the 90 degrees rotations of the input, see Figure 2 `here <https://arxiv.org/abs/2004.09691>`_ for
more details.
We recommend working with *odd-sized images/features* in such cases.


Default steerable basis
~~~~~~~~~~~~~~~~~~~~~~~

The parameters ``sigma``, ``frequencies_cutoff`` and ``rings`` of
:class:`~escnn.nn.modules.conv._RdConv` (and its subclasses :class:`~escnn.nn.R2Conv` or
:class:`~escnn.nn.R3Conv`) are optional parameters used to control how the basis for the steerable filters is built,
how it is sampled on the filter grid and how it is expanded to build the filter.
These **default values** work well in most cases since we tuned them manually in a few practical settings
(odd-sized filters, with relatively small kernel size), so we don't recommend changing them.
In case you may want to adapt them for different use cases (e.g. even-sized or really wide filters), we provide
more details about them below.

The default steerable basis used by the convolution layers is built by using :class:`~escnn.kernels.CircularShellsBasis`
or :class:`~escnn.kernels.SphericalShellsBasis` so steerable filters are in general split in multiple rings,
see Figure 2 `here <https://arxiv.org/pdf/1711.07289.pdf>`_ .

The parameter ``rings`` is a list of float values, defining the radius of the different rings the filter is split into.
The parameter ``sigma`` defines the width of each of them.
You can pass a list containing a different sigma per radius or just a singular float value which is used for all rings.
The parameter ``frequencies_cutoff`` regulates the maximum frequency on each ring.
By default, it uses a policy we manually tuned.
If you pass a float value ``frequencies_cutoff=F``, the maximum frequency at radius ``r`` becomes ``int(r * F)``.
Alternative, you can pass a generic function accepting a radius ``r`` in input and return the maximum frequency
accepted at that radius.



Even-sized filters and ``kernel_size=2``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While even-sized convolutional filters (``kernel_size`` is an even number) are supported,
they are generally less stable to continuous rotations.
This is partially because the default parameters of the steerable basis were manually tuned for odd-sized filters.
In particular, if ``kernel_size=2``, the resulting grid is too small to sample any filter in the constructed
steerable basis: as a result, the sampled basis is empty and an error is raised.
**We don't recommend using this grid** if your goal is achieving equivariance to all rotations.

A ``kernel_size=2`` filter can be useful if you only care about equivariance to :math:`90\deg` rotations.
In that case, you can play with the ``sigma``,  ``rings`` and ``frequencies_cutoff`` parameters of
:class:`~escnn.nn.modules.conv.R2Conv` or :class:`~escnn.nn.modules.conv.R3Conv` to build a non-empty basis.
By tuning these parameters, you might be able to get a basis stable enough to continuous rotations for
your application, but we didn't experiment with this setting enough to provide any recommendation.
You can find a useful discussion about this in `this issue <https://github.com/QUVA-Lab/e2cnn/issues/18>`_ .

