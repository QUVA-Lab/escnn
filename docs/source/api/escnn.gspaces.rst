
.. automodule:: escnn.gspaces
   :no-members:

escnn.gspaces
====================================

This subpackage implements G-spaces as a user interface for defining spaces and their symmetries.
The user typically instantiates a subclass of :class:`~escnn.gspaces.GSpace` to specify the symmetries considered and uses
it to instantiate equivariant neural network modules (see :doc:`escnn.nn`).

Generally, the user does not need to manually instantiate :class:`~escnn.gspaces.GSpace` or its subclasses.
Instead, we provide a few :ref:`factory functions <factory-functions-gspaces>` with simpler interfaces which can be
used to directly instantiate the most commonly used symmetries.

This subpackage depends on :doc:`escnn.group` and :doc:`escnn.kernels`.



A point :math:`\bold{v} \in \R^n` is parametrized using an :math:`(X, Y, Z, \dots)` convention,
i.e. :math:`\bold{v} = (x, y, z, \dots)^T`.
The representation :attr:`escnn.gspaces.GSpace.basespace_action` also assumes this convention.

However, when working with voxel data, the :math:`(\cdots, -Z, -Y, X)` convention is used.
That means that, in a feature tensor of shape :math:`(B, C, D_1, D_2, \cdots, D_{n-2}, D_{n-1}, D_n)``,
the last dimension is the X axis but the :math:`i`-th last dimension is the *inverted* :math:`i`-th axis.
Note that this is consistent with 2D images, where a :math:`(-Y, X)` convention is used.

This is especially relevant when transforming a :class:`~escnn.nn.GeometricTensor` or when building
convolutional filters in :class:`~escnn.nn.R3Conv` which should be equivariant to subgroups of :math:`\O3`
(e.g. when choosing the rotation axis for :func:`~escnn.gspaces.rot2dOnR3`).



.. note::
    Note that gspaces describing planar rotational symmetries (e.g. :func:`~escnn.gspaces.rot2dOnR2`,
    :func:`~escnn.gspaces.rot2dOnR3` or :func:`~escnn.gspaces.flipRot2dOnR2`) have an argument `N` which can be used to
    specify a discretization of the continuous rotational symmetry to integer multiples of :math:`\frac{2\pi}{N}`.
    However, this is generally not possible in 3D gspaces (e.g. :func:`~escnn.gspaces.rot3dOnR3`).

    The reason why :func:`~escnn.gspaces.rot3dOnR3` (and similar gspaces) does not have an this argument is that,
    while in 2D there exists a discrete subgroup of rotations of any order `N`, this is not the case in 3D.
    Indeed, in 3D, the only discrete 3D rotational symmetry groups are the symmetries of the few platonic solids (the
    tetrahedron group, the octahedron group and the icosahedron group).

    Recall that the :ref:`factory functions <factory-functions-gspaces>` listed here are just convenient shortcuts
    to create most gspaces with a simpler interface.
    Because a discrete rotation group of any order `N` can be generated in 2D, it is convenient to provide all these
    groups under the same interface in :func:`~escnn.gspaces.rot2dOnR2`.
    Conversely, since there are only a few options in 3D, we provide a different method for each of them
    (e.g. :func:`~escnn.gspaces.icoOnR3` or :func:`~escnn.gspaces.octaOnR3`).

    Alternatively, it is always possible to leverage the :meth:`~escnn.gspaces.GSpace.restrict` method to generate
    smaller symmetries.
    For instance, one can build a gspace with all rotational symmetries (:math:`\SO3`) with
    :func:`~escnn.gspaces.rot3dOnR3` and then restrict it to a particular subgroup (identified by a subgroup id `sgid`,
    see Subgroup Structure in :class:`~escnn.group.SO3`) of symmetries.
    The factory methods above are equivalent to this sequence of operations.

    Instead, if you want to leverage a **more generic discretisation** of :math:`\SO3` (or other groups) while trying
    to **preserve the full rotational equivariance**, we recommend choosing :func:`~escnn.gspaces.rot3dOnR3` and
    discretize the group only when using non-linearities within a neural network.
    This is done by using *Fourier Transform*-based non-linearities like :class:`~escnn.nn.FourierPointwise` or
    :class:`~escnn.nn.FourierELU`.
    In this case, the steerable features of a network are interpreted as the Fourier coefficients of a continuous
    feature over the full :math:`\SO3` group (or another group of your choice); inside the non-linearity module,
    the continuous features are sampled on a finite number `N` of (uniformly spread) points (via an
    *inverse Fourier Transform*) and a non-linearity (like ReLU or ELU) is applied point-wise.
    Finally, this operation is followed by a *(discretized) Fourier Transform* to recover the coefficients of the
    new features.
    This strategy enables parameterizing convolution layers which are :math:`\SO3` equivariant while leveraging an
    arbitrary discretization of the group when taking non-linearities. Note also that the `N` points used for
    discretization don't have to form a subgroup and, therefore, are not limited to the choice of the platonic symmetry
    groups anymore.
    You can generate different grids discretizing :math:`\SO3` (as well as other groups) by using the
    :meth:`~escnn.group.Group.grid` method of each group.



.. contents:: Contents
    :local:
    :backlinks: top


.. _factory-functions-gspaces:

Factory Methods
---------------


(trivial) action on single point
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: escnn.gspaces.no_base_space

action on plane
~~~~~~~~~~~~~~~

.. contents::
    :local:
    :backlinks: top

.. autofunction:: escnn.gspaces.rot2dOnR2

.. autofunction:: escnn.gspaces.flipRot2dOnR2

.. autofunction:: escnn.gspaces.flip2dOnR2

.. autofunction:: escnn.gspaces.trivialOnR2

action on volume
~~~~~~~~~~~~~~~~

.. contents::
    :local:
    :backlinks: top

.. autofunction:: escnn.gspaces.flipRot3dOnR3

.. autofunction:: escnn.gspaces.rot3dOnR3

.. autofunction:: escnn.gspaces.fullIcoOnR3

.. autofunction:: escnn.gspaces.icoOnR3

.. autofunction:: escnn.gspaces.fullOctaOnR3

.. autofunction:: escnn.gspaces.octaOnR3

.. autofunction:: escnn.gspaces.dihedralOnR3

.. autofunction:: escnn.gspaces.rot2dOnR3

.. autofunction:: escnn.gspaces.conicalOnR3

.. autofunction:: escnn.gspaces.fullCylindricalOnR3

.. autofunction:: escnn.gspaces.cylindricalOnR3

.. autofunction:: escnn.gspaces.mirOnR3

.. autofunction:: escnn.gspaces.invOnR3

.. autofunction:: escnn.gspaces.trivialOnR3


Abstract Group Space
--------------------

.. autoclass:: escnn.gspaces.GSpace
    :members:
    :undoc-members:

Group Action (trivial) on single point
--------------------------------------

.. autoclass:: escnn.gspaces.GSpace0D
    :members:
    :undoc-members:
    :show-inheritance:

Group Actions on the Plane
--------------------------

.. autoclass:: escnn.gspaces.GSpace2D
    :members:
    :undoc-members:
    :show-inheritance:

Group Actions on the 3D Space
-----------------------------

.. autoclass:: escnn.gspaces.GSpace3D
    :members:
    :undoc-members:
    :show-inheritance:


