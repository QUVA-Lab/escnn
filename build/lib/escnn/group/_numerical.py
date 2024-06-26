
########################################################################################################################
# Utils methods for decomposing or composing representations ###########################################################
########################################################################################################################

from __future__ import annotations

import escnn.group
from escnn.group import Group, GroupElement

from typing import Callable, Any, List, Tuple, Dict, Union, Iterable

import math
import numpy as np
import scipy as sp
from scipy import linalg, sparse
import scipy.sparse.linalg as slinalg
from scipy.sparse import find


try:
    import pymanopt
    from pymanopt.manifolds import Euclidean
    from pymanopt.optimizers import TrustRegions

except ImportError:
    pymanopt = None

try:
    import autograd.numpy as anp
except ImportError:
    anp = None

try:
    from sklearn.utils.extmath import randomized_svd
except ImportError:
    randomized_svd = None


########################################################################################################################
# Numerical utilities
########################################################################################################################


def null(A: Union[np.matrix, sparse.linalg.LinearOperator],
         use_sparse: bool,
         eps: float = 1e-12
         ) -> np.ndarray:
    """
    Compute a basis for the Kernel space of the matrix A.

    If ``use_sparse`` is ``True``, :meth:`scipy.sparse.linalg.svds` is used;
    otherwise, :meth:`scipy.linalg.svd` is used.

    Moreover, if the input is a sparse matrix, ``use_sparse`` has to be set to ``True``.

    Args:
        A: input matrix
        use_sparse: whether to use spare methods or not
        eps: threshold to consider a value zero. The default value is ``1e-12``

    Returns:
        A matrix whose columns are a basis of the kernel space

    """
    if use_sparse:
        k = min(A.shape) - 1
        u, s, vh = slinalg.svds(A, k=k)
    else:
        if randomized_svd is not None:
            k = min(A.shape)
            u, s, vh = randomized_svd(A, n_components=k, random_state=None)
        else:
            u, s, vh = linalg.svd(A, full_matrices=False)
    
    # print(u.shape, s.shape, vh.shape)
    # print(min(s))
    null_space = np.compress((s <= eps), vh, axis=0)
    return np.transpose(null_space)


def build_sylvester_constraint(rho_1: List[np.ndarray], rho_2: List[np.ndarray]) -> sparse.linalg.LinearOperator:
    
    assert len(rho_1) == len(rho_2)
    assert len(rho_1) > 0
    
    d1 = rho_1[0].shape[0]
    d2 = rho_2[0].shape[0]

    constraints = []
    for rho_1_g, rho_2_g in zip(rho_1, rho_2):
        
        assert rho_1_g.shape == (d1, d1)
        assert rho_2_g.shape == (d2, d2)

        # build the linear system corresponding to the Sylvester Equation with the current group element
        constraint = sparse.kronsum(rho_1_g, -rho_2_g.T, format='csc')
        constraints.append(constraint)

    # stack all equations in one unique matrix
    return sparse.vstack(constraints, format='csc')#.todense()


def find_intertwiner_basis_sylvester(rho_1: List[np.ndarray], rho_2: List[np.ndarray], eps: float = 1e-12) -> np.ndarray:
    
    constraint = build_sylvester_constraint(rho_1, rho_2)
    
    # the kernel space of this matrix contains the solutions of our problem
    
    if constraint.shape[1] == 1:
        if np.count_nonzero(constraint.todense()) == 0:
            return np.ones([1, 1])
        else:
            return np.zeros((1, 0))
    else:
        
        # compute the basis of the kernel
        
        # the sparse method can not compute the eigenspace associated with the smallest eigenvalue,
        # which is a problem when the null space is one dimensional
        
        # if len(rho_1) > 10:
        #     basis = null(constraint, True)
        # else:
        basis = null(constraint.todense(), False, eps=eps)
        
        assert np.allclose(constraint @ basis, 0.)
    
        return basis


def find_orthogonal_matrix(basis: np.ndarray, shape, verbose: bool = False) -> np.ndarray:

    # There is a bug in pygmanopt: a ZeroDivisionError is noted but not catched
    # This seems to happen when the basis contains some matrices like the identity and the anti-diagonal one.
    # (It is possible other bases cause the same issue, but I have not found out about them yet)
    # To avoid this error, we catch them before running the method

    if shape[0] == shape[1]:
        # if the identity matrix belongs to the span of the basis, return that
        eye = np.eye(*shape).reshape(-1, 1)
        w_eye = basis.T @ eye
        if np.allclose(eye, basis@w_eye):
            return eye.reshape(*shape), 0.

        # if the  anti-diagonal matrix belongs to the span of the basis, return that
        eye = np.eye(*shape)
        eye = np.fliplr(eye).reshape(-1, 1)

        w_eye = basis.T @ eye
        if np.allclose(eye, basis@w_eye):
            return eye.reshape(*shape), 0.

    if pymanopt is None:
        raise ImportError("Missing optional 'pymanopt' dependency. Install 'pymanopt' to use this function")
    
    if anp is None:
        raise ImportError("Missing optional 'autograd' dependency. Install 'autograd' to use this function")
    
    manifold = Euclidean(basis.shape[1])

    @pymanopt.function.autograd(manifold)
    def cost(X):
        d = anp.dot(basis, X).reshape(shape, order='F')
        if shape[0] < shape[1]:
            return anp.sum(anp.square(anp.dot(d, d.T) - anp.eye(shape[0])))
        elif shape[0] > shape[1]:
            return anp.sum(anp.square(anp.dot(d.T, d) - anp.eye(shape[1])))
        else:
            return anp.sum(
                anp.square(anp.dot(d, d.T) - anp.eye(*shape)) +
                anp.square(anp.dot(d.T, d) - anp.eye(*shape))
            )
    
    problem = pymanopt.Problem(manifold=manifold, cost=cost)
    
    # solver = TrustRegions(use_rand=True, miniter=10, mingradnorm=1e-10)
    # solver = ParticleSwarm(populationsize=500, maxcostevals=10000, logverbosity=0)
    # solver = ParticleSwarm(logverbosity=0)

    if not verbose:
        import os, sys
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    # Xopt = solver.solve(problem)
    # c = cost(Xopt)
    # print('PSO, Final Error:', c)
    #
    # x = Xopt

    solver = TrustRegions(min_gradient_norm=1e-10, log_verbosity=0)

    Xopt = solver.run(problem)  # , x=x) #, Delta_bar=np.sqrt(basis.shape[1])*2)

    c = Xopt.cost

    if not verbose:
        sys.stdout = old_stdout  # sys.__stdout__
    
    # print('TrustRegions, Final Error:', c)
    # print('Weights:', Xopt)

    D = np.dot(basis, Xopt.point).reshape(shape, order='F')
    
    return D, c


def sparse_allclose(A, B, atol=1e-8):
    diff = abs(A - B)
    _, _, v = find(diff)

    return np.less_equal(v, atol).all()


########################################################################################################################
# Numeric methods for irrep decomposition for FINITE GROUPS
########################################################################################################################

def compute_change_of_basis_finitegroup(
        representation: Dict[GroupElement, np.matrix],
        irreps: List[Tuple[Callable[[GroupElement], np.matrix], int]]
) -> np.matrix:
    r"""
    This method computes the change-of-basis matrix that decompose a representation of a *finite* group
    in a direct sum of irreps.

    Notice that the irreps are "stacked" in the same order as they are in the "irreps" list and consecutive copies
    of each irrep are added accordingly to the multiplicities specified.

    Args:
        representation: a dictionary mapping an element of "group" to a matrix
        irreps: a list of pairs (callable, integer). The callable implements an representation (takes an element as input and returns a matrix)
        and the integer is the multiplicity of this representation (i.e. how many times it has to appear in the decomposition)

    Returns:
        the change of _bases matrix

    """

    samples = list(representation.keys())
    representations = [representation[g] for g in samples]

    direct_sum_irreps = []

    for g in samples:
        blocks = []
        for irr, m in irreps:
            irr_g = irr(g)
            blocks += [irr_g]*m

        P = sparse.block_diag(blocks, format='csc')
        direct_sum_irreps.append(P)

    basis = find_intertwiner_basis_sylvester(representations, direct_sum_irreps)

    # reshape it to get the Change of Basis matrix
    shape = representations[0].shape

    # np.set_printoptions(precision=2, threshold=2 * len(representation)**2, suppress=True,
    #                     linewidth=len(representation) * 10 + 3)

    basis = linalg.orth(basis)

    # we could take any linear combination of the basis vectors to get the vectorized form of the Change of Basis matrix
    # d = basis @ np.random.randn(basis.shape[1], 1)

    # in case of CyclicGroup, if we have all the basis (i.e. we don't use the SparseSVD algorithm),
    # the sum of all basis vectors seems to always lead to an orthonormal matrix
    # d = basis @ np.ones((basis.shape[1], 1))
    # D = np.reshape(d, shape, order='F')

    # however, for large groups we can't use the dense SVD, so we need to find another orthonormal matrix in the
    # smaller space of solutions
    D, err = find_orthogonal_matrix(basis, shape)

    # print(D)
    # print(D @ D.T)
    # print(D.T @ D)

    # assert np.allclose(D @ D.T, np.eye(*shape))
    # assert np.allclose(D.T @ D, np.eye(*shape))

    # in case we take a random combination of the basis vectors, it is possible that the generated matrix is
    # singular. To be sure it is not we sample a few matrices and pick the one with the largest smallest singular
    # value. Anyway, the event of sampling a singular matrix should be unlikely enough to assume it never happens

    # max_sv = min(linalg.svd(D, compute_uv=False))
    # for i in range(50):
    #     # take any linear combination of them to get the vectorized form of the Change of Basis matrix
    #     d = _bases @ np.random.randn(_bases.shape[1], 1)
    #
    #     d = np.reshape(d, shape, order='F')
    #
    #     s = min(linalg.svd(d, compute_uv=False))
    #
    #     if s > max_sv:
    #         max_sv = s
    #         D = d

    # Check the change of basis found is right
    D_inv = linalg.inv(D)
    for element, rho in representation.items():
        # Build the direct sum of the irreps for this element
        blocks = []
        for (irrep, m) in irreps:
            repr = irrep(element)
            for i in range(m):
                blocks.append(repr)

        P = sparse.block_diag(blocks, format='csc')

        # if not np.allclose(rho, D @ P @ D_inv):
        #     print(element)
        #     print(rho)
        #     print(D @ P @ D_inv)

        assert (np.allclose(rho, D @ P @ D_inv)), "Error at element {}".format(element)

    return D


def find_irreps_multiplicities_finitegroup(
        representation: Dict[GroupElement, np.matrix],
        group: escnn.group.Group
) -> List[Tuple[Tuple, int]]:
        r"""
        The method computes the multiplicities of each irrep in the representation of a *finite* group using the
        inner product of their characters.

        It returns the decomposition in irreps as a list of "(irrep-name, multiplicity)" pairs,
        where "irrep-name" is the name of one of the irreps in ``group`` (a key in the :attr:`escnn.group.Group.irreps`
        dictionary) and "multiplicity" is the number of times this irrep appears in the decomposition.
        The order of this list follows the alphabetic order of the names.

        Args:
            representation: a dictionary associating to each group element a matrix representation
            group: the group whose irreps have to be used

        Returns:
            an ordered list of pairs (irrep-name, multiplicity)

        """

        # TODO - check also that they are all the generators possibly
        for g in representation.keys():
            assert g.group == group

        # compute the character of the representation w.r.t. the discrete group given
        character = {}
        for element, repr in representation.items():
            # the character maps an element to the trace of its representation
            character[element] = np.trace(repr)

        # compute the multiplicities of the irreps from the dot product between
        # their characters and the character of the representation
        multiplicities = []
        for irrep in group.irreps():
            # for each irrep
            multiplicity = 0.0

            # compute the inner product with the representation's character
            for element, char in character.items():
                multiplicity += char * irrep.character(~element)

            multiplicity /= len(character) * irrep.sum_of_squares_constituents

            # the result has to be an integer
            assert math.isclose(multiplicity, round(multiplicity), abs_tol=1e-9), \
                "Multiplicity of irrep [%s] is not an integer: %f" % (str(irrep.id), multiplicity)

            multiplicities.append((irrep.id, int(round(multiplicity))))

        # sort irreps by their name
        return sorted(multiplicities, key=lambda x: x[0])


def decompose_representation_finitegroup(
        representation: Dict[GroupElement, np.matrix],
        group: escnn.group.Group,
) -> Tuple[np.matrix, List[Tuple[Tuple, int]]]:
    r"""
    Decompose the input ``representation`` in a direct sum of irreps of the input *finite* ``group``.
    First, the method computes the multiplicities of each irrep in the representation using the inner product of their
    characters. Then, it computes the change-of-basis matrix which transforms the block-diagonal matrix coming from
    the direct sum of the irreps in the input representation.

    It returns the decomposition in irreps as a change-of-basis matrix and a list of "(irrep-name, multiplicity)" pairs,
    where "irrep-name" is the name of one of the irreps in ``group`` (a key in the :attr:`escnn.group.Group.irreps`
    dictionary) and "multiplicity" is the number of times this irrep appears in the decomposition.
    The order of this list follows the alphabetic order of the names and it represents the order in which the irreps
    have to be summed to build the block-diagonal representation.

    Args:
        representation: a dictionary associating to each group element a matrix representation
        group: the group whose irreps have to be used

    Returns:
        a tuple containing:

                - the change-of-basis matrix,

                - an ordered list of pairs (irrep-id, multiplicity)

    """

    assert group.generators is not None

    # build a list of representation instances with their multiplicities
    multiplicities = find_irreps_multiplicities_finitegroup(representation, group)

    # build a list of representation instances with their multiplicities
    irreps = [(group.irrep(*id), m) for (id, m) in multiplicities]

    representation = {
        g: representation[g] for g in group.generators
    }

    # compute te Change-Of-Basis matrix that transform the direct sum of irreps in the representation
    cob = compute_change_of_basis_finitegroup(representation, irreps)

    return cob, multiplicities



########################################################################################################################
# Numeric methods for irrep decomposition for GENERAL GROUPS
########################################################################################################################


class UnderconstrainedIrrepDecompositionSystem(Exception):

    def __init__(
            self,
            G: escnn.group.Group,
            j: Tuple,
            S: int,
            message: str = 'The algorithm to compute the Irrep Decomposition coefficients failed due to an unsufficient number of samples to constraint the problem',
    ):
        self.G = G
        self.j = j
        self.S = S
        super(UnderconstrainedIrrepDecompositionSystem, self).__init__(message)


class InsufficientIrrepsException(Exception):

    def __init__(
            self,
            G: escnn.group.Group,
            message: str = None,
    ):
        self.G = G

        if message is None:
            from textwrap import dedent
            message = dedent(f"""
                Error! Did not find sufficient irreps to complete the decomposition of the input representation.
                It is likely this happened because not sufficiently many irreps in '{G}' have been instantiated.
                Try instantiating more irreps and then repeat this call.
            """)
        super(InsufficientIrrepsException, self).__init__(message)


def _factor_out_endomorphisms(hombasis: np.ndarray, irrep: escnn.group.IrreducibleRepresentation) -> np.ndarray:

    endbasis = irrep.endomorphism_basis()
    d = endbasis.shape[0]

    assert hombasis.shape[1] % d == 0
    N = hombasis.shape[1] // d
    rho_size = hombasis.shape[0] // irrep.size

    if d == 1:
        # return hombasis.reshape(rho_size, irrep.size*N, order='F') * np.sqrt(irrep.size)
        return hombasis.reshape(irrep.size, rho_size, N)\
                       .transpose(1, 0, 2)\
                       .reshape(rho_size, N * irrep.size) * np.sqrt(irrep.size)

    embedding = []
    _hombasis = hombasis
    hombasis = hombasis.copy()

    eps = 1e-7

    for i in range(N):
        # v = hombasis[:, 0].reshape(rho_size, irrep.size, order='F')
        v = hombasis[:, 0].reshape(irrep.size, rho_size)

        # B = np.einsum('koi,ri->kro', endbasis, v).reshape(d, rho_size * irrep.size, order='F')
        B = np.einsum('koi,ir->kor', endbasis, v).reshape(d, rho_size * irrep.size)

        embedding.append(v)

        hombasis = hombasis - B.T @ (B @ hombasis)
        norms = np.linalg.norm(hombasis, axis=0, keepdims=True)
        mask = norms.reshape(-1) > eps

        assert mask[0] == False

        hombasis = hombasis[:, mask]
        norms = norms[:, mask]

        hombasis /= norms

    assert len(embedding) == N
    assert not mask.any(), mask

    # embedding = np.concatenate(embedding, axis=1) * np.sqrt(irrep.size)
    embedding = np.concatenate(embedding, axis=0).reshape(N, irrep.size, rho_size)\
                                                 .transpose(2, 1, 0)\
                                                 .reshape(rho_size, N * irrep.size) * np.sqrt(irrep.size)

    assert embedding.shape == (rho_size, N * irrep.size), embedding.shape

    return embedding


def _compute_irrep_embeddings(
        representation: Dict[GroupElement, np.matrix],
        irrep: escnn.group.IrreducibleRepresentation,
) -> np.ndarray:
    r"""
    This method computes the multiplciity of the input `irrep` in the input `representation` and returns an orthonormal
    embedding of the irrep in the representation for each of its occurrences.
    These embeddings can be used to form the change-of-basis matrix that decompose the representations into a direct
    sum of irreps.

    .. warning ::
        The method is *not* guaranteed to find only the embeddings.
        This depends on the samples used to instantiate the input `representation`.
        If the samples are not sufficient, the linear system might be underconstrained and the result might contain
        additional matrices which are not equivariant.

    Args:
        representation: a dictionary mapping an element of "group" to a matrix
        irrep (IrreducibleRepresentation): a callable implementing an representation (takes an element as input and returns a matrix)

    Returns:
        a matrix of shape `(S, I, m)`, where `S` is the size of `representation`, `I` is the size of the irrep and `m`
        its multiplicity

    """

    G = irrep.group

    assert len(representation) > 0, len(representation)

    samples = list(representation.keys())
    representations = [representation[g] for g in samples]

    irrep_values = []

    for g in samples:
        assert g.group == G
        irrep_values.append(irrep(g))

    rho_dim = representations[0].shape[0]
    psi_dim = irrep.size

    # compute a basis for the Homomorphism space Hom_G(psi, rho), where `irrep` is `psi` and `representation` is `rho`
    basis = find_intertwiner_basis_sylvester(representations, irrep_values)

    # warning! currently basis has memory layout (psi_dim, rho_dim, m), but reshaped to (psi_dim * rho_dim, m)

    # Note that Hom_G(psi, rho) ~= Hom_G(psi, psi^m) ~= End_G(psi) ^m
    # where `m` is the multiplicity of psi in rho
    # The dimension `dim_end_irrep` of End_G(psi) is irrep.sum_of_squares_constituents

    assert basis.shape[0] == rho_dim * psi_dim
    if basis.shape[1] % irrep.sum_of_squares_constituents != 0:
        raise UnderconstrainedIrrepDecompositionSystem(G, irrep.id, len(samples))

    # Therefore, we can compute the multiplicity `m` as  `m = N / dim_end_irrep`
    # where N is the dimensionality of the basis
    m = basis.shape[1] // irrep.sum_of_squares_constituents

    # If End_G(psi) is one dimensional, each element of the basis found above corresponds to a different embedding of
    # psi in rho
    # If End_G(psi) is not 1-dimensional, we need to factor out these endomorphisms from the basis found
    # To do so, we search in the space spanned by the basis for `m` homomorphisms such that, together, they form an
    # orthonormal matrix. In other words, we search for `m` homomorphisms whose images are orthogonal

    # shape of the matrix obtained by stacking horizontally the `m` homomorphisms
    shape = rho_dim, psi_dim * m

    if psi_dim * m > rho_dim:
        raise UnderconstrainedIrrepDecompositionSystem(G, irrep.id, len(samples))

    if m == 0:
        return np.zeros((rho_dim, psi_dim, m))
    else:

        basis = linalg.orth(basis)

        # warning! by using order='F' inside the following function, we implicitly fix the memory layout

        # D, err = find_orthogonal_matrix(basis, shape)
        D = _factor_out_endomorphisms(basis, irrep)

        assert D.shape == (rho_dim, psi_dim * m), (D.shape)

        # Check the change of basis found is right
        if not np.allclose(D.T @ D, np.eye(psi_dim*m)):
            # print(D)
            # print('-')
            # print(D.T @ D)
            # print('-----')
            raise UnderconstrainedIrrepDecompositionSystem(G, irrep.id, len(samples))
        # assert np.allclose(D.T @ D, np.eye(psi_dim*m))

        return D.reshape((rho_dim, psi_dim, m))


def compute_irrep_embeddings_general(
        representation: Callable[[GroupElement], np.matrix],
        irrep: escnn.group.IrreducibleRepresentation,
) -> np.ndarray:
    r"""
    This method computes the multiplciity of the input `irrep` in the input `representation` and returns an orthonormal
    embedding of the irrep in the representation for each of its occurrences.
    These embeddings can be used to form the change-of-basis matrix that decompose the representations into a direct
    sum of irreps.

    Args:
        representation (callable): a function mapping an element of "group" to a matrix.
                                   It should be possible to query the method with any element of the group `irrep.group`
        irrep (IrreducibleRepresentation): a callable implementing an representation (takes an element as input and returns a matrix)

    Returns:
        a matrix of shape `(S, I, m)`, where `S` is the size of `representation`, `I` is the size of the irrep and `m`
        its multiplicity

    """

    G = irrep.group

    try:
        generators = G.generators
        S = len(generators)

        if len(generators) == 0:
            assert G.order() == 1
            assert irrep == G.trivial_representation

            dim = representation(G.identity).shape[0]
            return np.eye(dim).reshape(dim, 1, dim)

    except ValueError:
        generators = []
        # number of samples to use to approximate the solutions
        # usually 4 are sufficient
        S = 4

    _S = S

    MAX_ATTEMPTS = 20

    for _ in range(MAX_ATTEMPTS):

        # Compute the orthogonal embeddings of the irrep in the representation using the samples

        try:
            # sometimes it might not converge, so we need to try a few times
            attepts = 5
            while True:
                try:
                    samples = generators + [G.sample() for _ in range(S - len(generators))]

                    end = _compute_irrep_embeddings(
                        {g: representation(g) for g in samples},
                        irrep
                    )

                except np.linalg.LinAlgError:
                    if attepts > 0:
                        attepts -= 1
                        continue
                    else:
                        raise
                else:
                    break
        except UnderconstrainedIrrepDecompositionSystem:
            # it is likely that the system was underconstrained and the basis found contained too many elements.
            #  we try again using more samples to build the constraint matrix
            S += 1
            continue

        # check that the solutions found are also in the kernel of the constraint matrix built with other random samples
        samples = generators + [G.sample() for _ in range(20)]

        rho_g = np.stack([
            representation(g) for g in samples
        ], axis=0)

        psi_g = np.stack([
            irrep(g) for g in samples
        ], axis=0)

        # check that the solution commutes with psi and rho
        end_psi = np.einsum('rim,gio->grom', end, psi_g)
        rho_end = np.einsum('goi,ipm->gopm', rho_g, end)

        if np.allclose(end_psi, rho_end):
            break

        # if this not the case, it means the solutions contained some elements which were not equivariant.
        # this is most likely due to an underconstrained system, so we try again using more samples to
        # build the constraint matrix
        S += 1

    else:
        # If after MAX_ATTEMPTS attemps no solution has been found, raise and error
        raise UnderconstrainedIrrepDecompositionSystem(G, irrep.id, S)

    return end


def decompose_representation_general(
        representation: Callable[[GroupElement], np.matrix],
        group: escnn.group.Group,
        irreps: List[escnn.group.IrreducibleRepresentation] = None
) -> Tuple[np.matrix, List[Tuple[Tuple, int]]]:
    r"""
    This method computes the multiplicity of each irrep of `group` in the input `representation` and an orthonormal
    embedding of each irrep in the representation for each of its occurrences.
    These embeddings are then used to form the change-of-basis matrix that decompose the representations into a direct
    sum of the irreps.

    .. warning ::
        This numerical method might be relatively expensive for large representations and groups.
        It is not recommended to call it multiple times on the same inputs.
        Instead, it is recommended to call this method once and cache its result, to make it available for immediate
        usage.

    Args:
        representation (callable): a function mapping an element of "group" to a matrix.
                                   It should be possible to query the method with any element of the group `irrep.group`
        group (Group): the group whose irreps have to be used
        irreps (list, optional): list of irreps of `group` to use, rather than looping over all the irrep of `group`
                                 available in `group.irreps()`

    Returns:
        a tuple containing:

                - the change-of-basis matrix,

                - an ordered list of pairs (irrep-id, multiplicity)

    """

    rho_size = representation(group.identity).shape[0]

    change_of_basis = np.empty((rho_size, rho_size))

    if irreps is None:
        irreps = group.irreps()

    size = 0
    irreps_multiplicities = []
    for psi in irreps:
        end_psi = compute_irrep_embeddings_general(representation, psi)

        # multiplicity of psi
        m = end_psi.shape[2]

        if m > 0:
            irreps_multiplicities.append((psi.id, m))

            # swap the last two axes to fit it in the change of basis in the right format
            end_psi = end_psi.reshape(rho_size, psi.size, m).transpose(0, 2, 1).reshape(rho_size, psi.size*m)
            change_of_basis[:, size:size+psi.size*m] = end_psi

        size += psi.size * m

    # check that size == rho_size
    if size < rho_size:
        raise InsufficientIrrepsException(group)

    assert size <= rho_size, f"""
        Error! Found too many irreps in the the decomposition of the input representation.
        This should never happen!
    """

    # check that the matrix is orthogonal
    change_of_basis_t = change_of_basis.T
    assert np.allclose(change_of_basis @ change_of_basis_t, np.eye(rho_size))
    assert np.allclose(change_of_basis_t @ change_of_basis, np.eye(rho_size))

    # check that the solution commutes with the representations
    for _ in range(10):
        g = group.sample()
        # Build the direct sum of the irreps for this element
        blocks = []
        for (irrep, m) in irreps_multiplicities:
            repr = group.irrep(*irrep)(g)
            for i in range(m):
                blocks.append(repr)

        P = sparse.block_diag(blocks, format='csc')

        rho = representation(g)

        assert (np.allclose(rho, change_of_basis @ P @ change_of_basis_t)), "Error at element {}".format(g)

    return change_of_basis, irreps_multiplicities


