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
        # If the base graph is very large, we compute the top 20 eigenvalues.
        # This should overestimate the number we need. Convergence to
        # TW has std ~C*n**(-2/3), while 20/n eigs has std ~C*n**(-1/2)
        base_eigs = eigsh(base_graph, k=20, return_eigenvectors=False)

    # The following will be need to make sure that the new eigenvalue generated
    # is indeed new. Only relevant when not all the base eigs are calculated.
    base_eig_smallest_pos = np.min([eig for eig in base_eigs if eig > 0])
    base_eig_largest_neg = np.max([eig for eig in base_eigs if eig < 0])
    base_eig_smallest_abs = np.min([abs(eig) for eig in base_eigs])
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

    min_dist = 50

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
            eigs = [eig for eig in eigs if np.min(abs(base_eigs - eig)) > 10 ** (-10)]

            if len(eigs) == 0:  # In this case, recompute with more eigenvalues.
                # Note: If k is increased for a single graph in the loop, it increases
                # for all others. With high probability, if k needs to be
                # larger, it is due to the base graph and not the cover.
                k = k + 2
            else:
                # Append the largest magnitude eigenvalues not from the base_graph
                output.append(sorted(eigs, key=lambda x: abs(x))[-1])
                found_new_eig = True

                # One way a new eigenvalue could fail to be new, is that it is
                # a base graph eigenvalue, but is not in the list of eigenvalues
                # we calculated because we chose k too small.
                # The following code checks that we can guarantee that the new
                # eigenvalue is new. The entire function will break if we cannot do
                # this, as that indicates a fatal error by the author,
                # as then the program is not computing what was intended.
                if eig_type == "max_positive":

                    if min_dist > np.max(eigs) - base_eig_smallest_pos:
                        min_dist = np.max(eigs) - base_eig_smallest_pos

                    if np.max(eigs) < base_eig_smallest_pos:
                        raise ValueError(
                            f"Largest New Eigenvalue {np.max(eigs)} smaller than "
                            f" smallest base eig {base_eig_smallest_pos}"
                        )
                elif eig_type == "max_negative":
                    if np.min(eigs) > base_eig_largest_neg:
                        raise ValueError(
                            f"Smallest New Eigenvalue {np.min(eigs)} larger than "
                            f" largest base eig {base_eig_largest_neg}"
                        )
                elif eig_type == "max_magnitude":
                    if np.max(np.abs(np.array(eigs))) < base_eig_smallest_abs:
                        raise ValueError(
                            f"Largest magnitude new Eigenvalue {np.min(eigs)} smaller "
                            f" than smallest base eig {base_eig_smallest_abs}"
                        )
    print(min_dist)
    return output
