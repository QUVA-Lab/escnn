.. automodule:: escnn.kernels
   :no-members:

escnn.kernels
=============

This subpackage implements the complete analytical solutions of the equivariance constraints on the kernel space as
explained in
`A Program to Build E(N)-Equivariant Steerable CNNs <https://openreview.net/forum?id=WE4qe9xlnQw>`_.
The bases of equivariant kernels should be built through :ref:`factory functions <factory-functions-bases>`.

Typically, the user does not need to interact directly with this subpackage.
Instead, we suggest to use the interface provided in :doc:`escnn.gspaces`.

This subpackage depends only on :doc:`escnn.group`.

.. warning::
    This module has been largely refactored in version `1.0.0` and is not compatible with previous versions of the library.
    The interface of the other modules in not affected but some of the generated bases might be slightly different (e.g. basis
    elements being indexed in a different order). That means the weights trained using an older version of the library
    might not be compatible with newer instantiations of your models. For backward compatibility, we recommend using
    version `0.1.9` of the library.

Finally, note that a :class:`~escnn.kernels.KernelBasis` is a subclass of :class:`torch.nn.Module`.
As such, a :class:`~escnn.kernels.KernelBasis` can be treated as a PyTorch's module, e.g. it can be moved to a CUDA
enable device, the floating point precision of its parameters and buffers can be changed and its forward pass is
generally differentiable.

.. warning::
    Because non-zero circular and spherical harmonics are not well defined at the origin, the gradients of
    :class:`~escnn.kernels.CircularShellsBasis` and :class:`~escnn.kernels.SphericalShellsBasis` are difficult to compute.
    In 2D, the gradient at the origin of :class:`~escnn.kernels.CircularShellsBasis` is always zero (this could be solved
    in the most recent PyTorch versions by leveraging some complex powers).
    Instead, in 3D, :class:`~escnn.kernels.SphericalShellsBasis` should be able to compute reasonable estimates.


.. contents:: Contents
    :local:
    :backlinks: top


Generic Kernel Bases
--------------------

.. contents::
    :local:
    :backlinks: top


KernelBasis
~~~~~~~~~~~
.. autoclass:: escnn.kernels.KernelBasis
    :members:
    :show-inheritance:

AdjointBasis
~~~~~~~~~~~~
.. autoclass:: escnn.kernels.AdjointBasis
    :members:
    :show-inheritance:

UnionBasis
~~~~~~~~~~
.. autoclass:: escnn.kernels.UnionBasis
    :members:
    :show-inheritance:


EmptyBasisException
~~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.kernels.EmptyBasisException
    :members:
    :undoc-members:


General Steerable Basis for equivariant kernels
-----------------------------------------------

The following classes implement the logic behind the kernel constraint solutions described in
`A Program to Build E(N)-Equivariant Steerable CNNs <https://openreview.net/forum?id=WE4qe9xlnQw>`_.
:class:`~escnn.kernels.SteerableKernelBasis` solves the kernel constraint for a generic group and any pair of input and
output representations by decomposing them into irreducible representations and, then, solving the simpler constraints
on irreps independently, as described in `General E(2)-Equivariant Steerable CNNs <https://arxiv.org/abs/1911.08251>`_ .
:class:`~escnn.kernels.IrrepBasis` provides an interface for the methods which solve the kernel constraint for irreps.
We implement two such methods, :class:`~escnn.kernels.WignerEckartBasis` and
:class:`~escnn.kernels.RestrictedWignerEckartBasis`.


.. contents::
    :local:
    :backlinks: top


SteerableKernelBasis
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.kernels.SteerableKernelBasis
    :members:
    :show-inheritance:

IrrepBasis
~~~~~~~~~~
.. autoclass:: escnn.kernels.IrrepBasis
    :members:
    :show-inheritance:

WignerEckartBasis
~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.kernels.WignerEckartBasis
    :members:
    :show-inheritance:

RestrictedWignerEckartBasis
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.kernels.RestrictedWignerEckartBasis
    :members:
    :show-inheritance:



Steerable Filter Bases
----------------------

To solve the kernel constraint and generate a :class:`~escnn.kernels.SteerableKernelBasis`, one requires a steerable
basis for *scalar* filters over the base space, as derived in
`A Program to Build E(N)-Equivariant Steerable CNNs <https://openreview.net/forum?id=WE4qe9xlnQw>`_.
Different implementations and parameterizations of steerable convolutions effectively differ by the choice of steerable
filter basis.
The following classes implement different choices of :class:`~escnn.kernels.SteerableFilterBasis`.

.. contents::
    :local:
    :backlinks: top

SteerableFiltersBasis
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.kernels.SteerableFiltersBasis
    :members:
    :show-inheritance:

PointBasis
~~~~~~~~~~
.. autoclass:: escnn.kernels.PointBasis
    :members:
    :show-inheritance:

CircularShellsBasis
~~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.kernels.CircularShellsBasis
    :members:
    :show-inheritance:

SphericalShellsBasis
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.kernels.SphericalShellsBasis
    :members:
    :show-inheritance:

SparseOrbitBasis
~~~~~~~~~~~~~~~~
.. autoclass:: escnn.kernels.SparseOrbitBasis
    :members:
    :show-inheritance:

SparseOrbitBasisWithIcosahedralSymmetry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.kernels.SparseOrbitBasisWithIcosahedralSymmetry
    :members:
    :show-inheritance:

GaussianRadialProfile
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.kernels.GaussianRadialProfile
    :members:
    :show-inheritance:



.. _factory-functions-bases:


Generic Group acting on a single point
--------------------------------------

.. autofunction:: escnn.kernels.kernels_on_point


Bases for Group Actions on the Plane
------------------------------------

The following factory functions provide an interface to build the bases for kernels equivariant to groups acting on the
two dimensional plane (:math:`d=2`).
The names of the functions follow this convention `kernels_[G]_act_R[d]`, where :math:`G` is the origin-preserving isometry
group while :math:`\R^d` is the space on which it acts, interpreted as the domain of the
kernel :math:`\kappa: \R^d \to \R^{c_\text{out} \times c_\text{in}}`.
In the language of `Gauge Equivariant CNNs <https://arxiv.org/abs/1902.04615>`_ , the origin-preserving isometry
:math:`G` is called *structure group* (or, sometimes, *gauge group*).

.. contents:: R2 Bases
    :local:
    :backlinks: top


R2: Reflections
~~~~~~~~~~~~~~~

.. autofunction:: escnn.kernels.kernels_Flip_act_R2

R2: Reflections and Discrete Rotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: escnn.kernels.kernels_DN_act_R2

R2: Reflections and Continuous Rotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: escnn.kernels.kernels_O2_act_R2

R2: Trivial Action
~~~~~~~~~~~~~~~~~~

.. autofunction:: escnn.kernels.kernels_Trivial_act_R2

R2: Discrete Rotations
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: escnn.kernels.kernels_CN_act_R2

R2: Continuous Rotations
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: escnn.kernels.kernels_SO2_act_R2

R2: Generic Subgroups of Continuous Rotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: escnn.kernels.kernels_SO2_subgroup_act_R2

R2: Generic Subgroups of Continuous Rotations and Reflections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: escnn.kernels.kernels_O2_subgroup_act_R2

Bases for Group Actions in 3D Space
-----------------------------------

The following factory functions provide an interface to build the bases for kernels equivariant to groups acting on the
three dimensional Euclidean space (:math:`d=3`).
The names of the functions follow this convention `kernels_[G]_act_R[d]`, where :math:`G` is the origin-preserving isometry
group while :math:`\R^d` is the space on which it acts, interpreted as the domain of the
kernel :math:`\kappa: \R^d \to \R^{c_\text{out} \times c_\text{in}}`.
In the language of `Gauge Equivariant CNNs <https://arxiv.org/abs/1902.04615>`_ , the origin-preserving isometry
:math:`G` is called *structure group* (or, sometimes, *gauge group*).


.. contents:: R3 Bases
    :local:
    :backlinks: top



R3: Inversions and Continuous Rotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: escnn.kernels.kernels_O3_act_R3

R3: Continuous Rotations
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: escnn.kernels.kernels_SO3_act_R3

R3: Generic Subgroup of Continuous Rotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: escnn.kernels.kernels_SO3_subgroup_act_R3

R3: Generic Subgroup of Continuous Rotations and Inversions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: escnn.kernels.kernels_O3_subgroup_act_R3

R3: Icosahedral symmetries
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: escnn.kernels.kernels_Ico_act_R3

R3: Octahedral symmetries
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: escnn.kernels.kernels_Octa_act_R3

R3: Tetrahedral symmetries
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: escnn.kernels.kernels_Tetra_act_R3

R3: Full Icosahedral symmetries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: escnn.kernels.kernels_FullIco_act_R3

R3: Full Octahedral symmetries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: escnn.kernels.kernels_FullOcta_act_R3

R3: Full Tetrahedral symmetries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: escnn.kernels.kernels_FullTetra_act_R3

R3: Pyritohedral symmetries
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: escnn.kernels.kernels_Pyrito_act_R3

R3: Planar continuous Rotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: escnn.kernels.kernels_SO2_act_R3

R3: Planar discrete Rotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: escnn.kernels.kernels_CN_act_R3

R3: Continuous Cone Symmetries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: escnn.kernels.kernels_O2_conical_act_R3

R3: Discrete Cone Symmetries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: escnn.kernels.kernels_DN_conical_act_R3

R3: Continuous Dihedral Symmetries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: escnn.kernels.kernels_O2_dihedral_act_R3

R3: Discrete Dihedral Symmetries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: escnn.kernels.kernels_DN_dihedral_act_R3

R3: Inversions
~~~~~~~~~~~~~~

.. autofunction:: escnn.kernels.kernels_Inv_act_R3

R3: Trivial Action
~~~~~~~~~~~~~~~~~~

.. autofunction:: escnn.kernels.kernels_Trivial_act_R3



