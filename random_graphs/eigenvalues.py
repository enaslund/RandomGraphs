import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix
from . import covers


def generate_loop_extremal_eigs(*, size, deg, number, eig_type, simple):
    output = []

    if eig_type == "max_positive":
        id_mult = 1
    elif eig_type == "max_negative":
        id_mult = -1
    elif eig_type == "max_magnitude":
        id_mult = 0
    else:
        raise ValueError(
            f"size_type '{eig_type}' is invalid. Must be one of "
            f"'max_positive', 'max_negative', 'max_magnitude' "
        )

    identity_shift = id_mult * int(deg / 2)

    if simple:
        graph_function = covers.random_simple_graph
    else:
        graph_function = covers.random_graph

    # With small probability, the graph will be disconnected, and return the
    # value 4 twice.
    count = 0
    while count < number:
        B = graph_function(size=size, deg=deg, identity_shift=identity_shift)

        shifted_eigs = eigsh(B, k=2, return_eigenvectors=False)
        eigs = [x - identity_shift for x in shifted_eigs]

        #            print([np.min(abs(base_eigs - eig)) for eig in eigs])
        eigs = [eig for eig in eigs if abs(eig - deg) > 10 ** (-10)]
        if len(eigs) > 0:
            output.append(eigs[0])
            count += 1

    return output


def generate_matrix_rep_eigs(*, size, deg, number, matrix_func, eig_type):
    """Returns eigenvalues from generating random graphs of a particular
    size and degree, and replacing ones with random matrices according to
    the matrix_func"""
    if eig_type == "max_positive":
        id_mult = 1
    elif eig_type == "max_negative":
        id_mult = -1
    elif eig_type == "max_magnitude":
        id_mult = 0
    else:
        raise ValueError(
            f"size_type '{eig_type}' is invalid. Must be one of "
            f"'max_positive', 'max_negative', 'max_magnitude' "
        )

    identity_shift = id_mult * int(deg / 2)

    output = []
    for i in range(0, number):
        base_graph = covers.random_simple_graph(size=size, deg=deg, identity_shift=0)

        B = covers.random_cover_matrix_rep(
            base_graph=base_graph,
            matrix_func=matrix_func,
            identity_shift=identity_shift,
        )

        eig = eigsh(B, k=1, return_eigenvectors=False)[0] - identity_shift
        output.append(eig)
    return output


def generate_new_extremal_eigs(
    *, base_graph, cover_deg, permutation_func, number, trivial_eig, eig_type
):
    # We want to find the base eigenvalues that we should ignore.
    # If the base_graph is enormous, we need to be careful and
    # compute a smaller number of eigenvalues using a sparse routine.
    if base_graph.shape[0] < 1000:
        if not isinstance(base_graph, np.matrix):
            dense_base_graph = base_graph.todense()
        else:
            dense_base_graph = base_graph
        base_eigs = np.linalg.eigh(dense_base_graph)[0]
        base_eig_mag_cutoff = 0
    else:

        # If the base graph is very large, we compute the top k eigenvalues.
        # This should overestimate the number we need. Convergence to
        # TW has std ~C*n**(-2/3), while k/n eigs has std ~C_k*n**(-1/2)
        if base_graph.shape[0] <= 20000:
            num_base_eigs = 50
        else:
            num_base_eigs = 20

        base_graph = csr_matrix(base_graph)
        base_eigs = eigsh(base_graph, k=num_base_eigs, return_eigenvectors=False)

        if eig_type == "max_positive":
            base_eig_mag_cutoff = np.min([abs(eig) for eig in base_eigs if eig > 0])
        elif eig_type == "max_negative":
            base_eig_mag_cutoff = np.min([abs(eig) for eig in base_eigs if eig < 0])
        elif eig_type == "max_magnitude":
            base_eig_mag_cutoff = np.min([abs(eig) for eig in base_eigs])

    # The random_cover function runs much faster if we first turn the base graph
    # into a dok_matrix
    base_graph = dok_matrix(base_graph)

    output = []

    # base_eig_mag_cutoff is the smallest magnitude of the relevant base eigenvalues
    # This quantity only matters when we do not compute all of the base eigenvalues.
    if eig_type == "max_positive":
        id_mult = 1
    elif eig_type == "max_negative":
        id_mult = -1
    elif eig_type == "max_magnitude":
        id_mult = 0
    else:
        raise ValueError(
            f"size_type '{eig_type}' is invalid. Must be one of "
            f"'max_positive', 'max_negative', 'max_magnitude' "
        )

    identity_shift = id_mult * int(trivial_eig / 2)

    # If our number of iterations is smaller, the cost of having to compute
    # eigenvalues more than once is higher.
    # Remark: A more faster approach would be to increase num_eigs_to_get by
    # the number of relevant base_graph eigs larger than the radius of the
    # universal cover. This function applies to irregular graphs, so we
    # chose not to split into 2 functions in order to do this for reg graphs
    if number < 10:
        num_eigs_to_get = 5
    elif number < 20:
        num_eigs_to_get = 4
    else:
        num_eigs_to_get = 2

    for i in range(0, number):
        B = covers.random_cover(
            base_graph=base_graph,
            cover_deg=cover_deg,
            permutation_func=permutation_func,
            identity_shift=identity_shift,
        )

        found_new_eig = False

        while found_new_eig is not True:
            shifted_eigs = eigsh(B, k=num_eigs_to_get, return_eigenvectors=False)
            eigs = [x - identity_shift for x in shifted_eigs]

            #            print([np.min(abs(base_eigs - eig)) for eig in eigs])
            eigs = [eig for eig in eigs if np.min(abs(base_eigs - eig)) > 10 ** (-10)]

            if len(eigs) == 0:  # In this case, recompute with more eigenvalues.
                # Note: If k is increased for a single graph in the loop, it increases
                # for all others. With high probability, if k needs to be
                # larger, it is due to the base graph and not the cover.
                num_eigs_to_get += 2
            else:
                # Append the largest magnitude eigenvalues not from the base_graph
                lm_new_eig = sorted(eigs, key=lambda x: abs(x))[-1]
                output.append(lm_new_eig)
                found_new_eig = True

                # One way a new eigenvalue could fail to be new, is that it is
                # a base graph eigenvalue, but is not in the list of eigenvalues
                # we calculated because we chose k too small.
                # The following code checks that we can guarantee that the new
                # eigenvalue is new. The entire function will break if we cannot do
                # this, as that indicates a fatal error by the author,
                # as then the program is not computing what was intended.

                if abs(lm_new_eig) < base_eig_mag_cutoff:
                    raise RuntimeError(
                        f"Largest New Eigenvalue {lm_new_eig} is smaller than the "
                        f"smallest magnitude relevant base eig {base_eig_mag_cutoff}."
                        f"Hard-coded value {num_base_eigs} of num_base_eigs too low."
                    )
    return output
