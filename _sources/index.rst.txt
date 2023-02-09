
:github_url: https://github.com/QUVA-Lab/escnn

escnn documentation
=================================

*escnn* is a Pytorch based library for equivariant deep learning.

*Equivariant neural networks* guarantee a prespecified transformation behavior of their features under transformations of their input.
This package provides functionality for the equivariant processing of signals over Euclidean spaces (e.g. planar images or 3D signals).
It implements the *most general convolutional maps* which are equivariant under the isometries of the Euclidean space, that is, under translations, rotations and reflections.
Currently, it supports 2 and 3 dimensional Euclidean spaces.
The library also supports compact-group equivariant linear maps (interpreted as a special case of equivariant maps on a 0 dimensional Euclidean space) which can be used to construct equivariant MLPs.

.. warning::
    :doc:`escnn.kernels <api/escnn.kernels>` has been largely refactored in version `1.0.0`.
    While the interface of the other sub-packages in not affected, the weights trained using an older version of the library
    might not be compatible with newer instantiations of your models. For backward compatibility, we recommend using
    version `0.1.9` of the library.

Package Reference
-----------------

The library is structured into four subpackages with different high-level features:

* :doc:`escnn.group <api/escnn.group>`         implements basic concepts of group and representation theory
    
* :doc:`escnn.kernels <api/escnn.kernels>`     solves for spaces of equivariant convolution kernels
    
* :doc:`escnn.gspaces <api/escnn.gspaces>`     defines the Euclidean spaces and their symmetries
        
* :doc:`escnn.nn <api/escnn.nn>`               contains equivariant modules to build deep neural networks

Typically, only the high level functionalities provided in :doc:`escnn.gspaces <api/escnn.gspaces>` and
:doc:`escnn.nn <api/escnn.nn>` are needed to build an equivariant model.


Getting Started and Useful References
-------------------------------------

To get started, we provide an `introductory tutorial <https://github.com/QUVA-Lab/escnn/blob/master/examples/introduction.ipynb>`_
which introduces the basic functionality of the library.
A second `tutorial <https://github.com/QUVA-Lab/escnn/blob/master/examples/model.ipynb>`_ goes through building and training
an equivariant model on the rotated MNIST dataset.
Note that *escnn* also supports equivariant MLPs; see `these examples <https://github.com/QUVA-Lab/escnn/blob/master/examples/mlp.ipynb>`_.

Check also the `tutorial <https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Geometric_deep_learning/tutorial2_steerable_cnns.html>`_ on Steerable CNNs using our library in the *Deep Learning 2* course at the University of Amsterdam.

If you want to better understand the theory behind equivariant and steerable neural networks, you can check these references:
- Erik Bekkers' `lectures <https://uvagedl.github.io/>`_ on *Geometric Deep Learning* at in the Deep Learning 2 course at the University of Amsterdam
- The course material also includes a `tutorial <https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Geometric_deep_learning/tutorial1_regular_group_convolutions.html>`_ on *group convolution* and `another <https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Geometric_deep_learning/tutorial2_steerable_cnns.html>`_ about Steerable CNNs, using *this library*.
- My `thesis <https://gabri95.github.io/Thesis/thesis.pdf>`_ provides a brief overview of the essential mathematical ingredients needed to understand Steerable CNNs.



.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Package Reference
   :hidden:

   api/escnn.group
   api/escnn.kernels
   api/escnn.gspaces
   api/escnn.nn


Cite Us
-------

The development of this library was part of the work done in `our ICLR 22 paper <https://openreview.net/pdf?id=WE4qe9xlnQw>`_
and is an extension of the `e2cnn <https://github.com/QUVA-Lab/e2cnn>`_ library developed in `our previous NeurIPS 19 paper <https://arxiv.org/abs/1911.08251>`_ .
Please, cite us if you use this code in your own work::

   @inproceedings{cesa2022a,
        title={A Program to Build {E(N)}-Equivariant Steerable {CNN}s },
        author={Gabriele Cesa and Leon Lang and Maurice Weiler},
        booktitle={International Conference on Learning Representations (ICLR)},
        year={2022},
    }

    @inproceedings{e2cnn,
        title={{General {E(2)}-Equivariant Steerable CNNs}},
        author={Weiler, Maurice and Cesa, Gabriele},
        booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
        year={2019},
    }


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`

