import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix
import covers


def generate_loop_extremal_eigs(*, deg, size, number, eig_type, simple):
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
        eigs = [eig for eig in eigs if abs(eig - deg) > 10 ** (-12)]
        if len(eigs) > 0:
            output.append(eigs[0])
            count += 1

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
    else:
        base_graph = csr_matrix(base_graph)
        # If the base graph is very large, we compute the top 100 eigenvalues.
        # This should overestimate the number we need by a lot.
        base_eigs = eigsh(base_graph, k=100, return_eigenvectors=False)

    # The random_cover function runs much faster if we first turn the base graph
    # into a dok_matrix
    base_graph = dok_matrix(base_graph)

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

    identity_shift = id_mult * int(trivial_eig / 2)

    k = 2
    for i in range(0, number):
        B = covers.random_cover(
            base_graph=base_graph,
            cover_deg=cover_deg,
            permutation_func=permutation_func,
            identity_shift=identity_shift,
        )

        found_new_eig = False

        while found_new_eig is not True:
            shifted_eigs = eigsh(B, k=k, return_eigenvectors=False)
            eigs = [x - identity_shift for x in shifted_eigs]

            #            print([np.min(abs(base_eigs - eig)) for eig in eigs])
            eigs = [eig for eig in eigs if np.min(abs(base_eigs - eig)) > 10 ** (-12)]

            if len(eigs) > 0:
                # Append the largest magnitude eigenvalues not from the base_graph
                output.append(sorted(eigs, key=lambda x: abs(x))[-1])
                found_new_eig = True
            else:
                # Note: If k is increased for a single graph in the loop, it increases
                # for all others. With high probability, if k needs to be
                # larger, it is due to the base graph and not the cover.
                k = k + 2
    return output
