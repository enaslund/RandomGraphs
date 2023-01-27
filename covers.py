import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix
from .permutations import random_derangement


def random_cover(*, base_graph, cover_deg, permutation_func, identity_shift):
    """Returns a random cover of the base graph, according to the permutation_func given

    Args:
        base_graph (np.array or csr_matrix): The adjacency matrix of the base graph.
            This must be a symmetric square matrix.
        cover_deg (int): The degree of the cover.
        permutation_func (int -> np.array): The function used to generate permutations
            of {1,...,degree}. Must take in an int, and output an array of that length.
            E.g. Use np.random.permutation for the general case of the symmetric group.
        identity_shift (float): Shift the resulting adjacency matrix by x*I for some
            x, where I is the identity matrix. This helps when computing the most
            negative or the most positive eigenvalues of graphs that where these are
            similar in magnitude.

    Returns:
        scipy.sparse.csr_matrix: A sparse square matrix of size len(base_graph)*degree.
    """
    n = base_graph.shape[0] * cover_deg
    # Originally row_ind and col_ind were numpy arrays, where we appended in each loop
    # However for taking covers of extremely large graphs that ran slowly on my machine
    # and it was significantly faster to store everything in lists and append at the end
    row_ind = []
    col_ind = []

    # This turns the matrix A into a sparse list of entries
    # For covers of extremely large graphs, this is the slowest step
    if not isinstance(base_graph, dok_matrix):
        base_graph = dok_matrix(base_graph)

    ones = np.ones(cover_deg).astype(int)
    arange = np.arange(cover_deg)
    for (i, j), entry in base_graph.items():
        if i < j:
            continue
        elif i == j:
            entry = int(entry / 2)
        else:
            entry = int(entry)

        for k in range(0, entry):
            random_perm = permutation_func(cover_deg)

            x_coords = cover_deg * i * ones + arange
            y_coords = cover_deg * j * ones + random_perm
            row_ind.append(x_coords)
            col_ind.append(y_coords)

            col_ind.append(x_coords)
            row_ind.append(y_coords)

    row_ind.append(np.arange(n))
    col_ind.append(np.arange(n))

    row_ind = np.concatenate(row_ind)
    col_ind = np.concatenate(col_ind)

    data = np.append(
        np.ones(cover_deg * int(sum(base_graph.values()))), identity_shift * np.ones(n)
    )  # .astype(int)

    return csr_matrix((data, (row_ind, col_ind)), shape=(n, n))


def random_graph(*, deg, size, identity_shift=0, simple=False):
    """Generates a random graph from the loop. To make simple
    graphs the algorithm is inefficient and tries randomly until it succeeds.
    A different implementation is required for higher degree.

    Args:
        deg (int): The degree of the graph. Must be even since we are taking covers
            of the loop.
        size (int): The number of vertices of the output graph.
        identity_shift (float): Shift the resulting adjacency matrix by x*I for some
            real number x, where I is the identity matrix. This helps when computing the
            most negative or positive eigenvalues of graphs that where these are
            similar in magnitude.
        simple (boolean): Whether the output should be a simple graph.

    Returns:
        scipy.sparse.csr_matrix: A sparse square adjacencymatrix of size size
    """
    if deg % 2 != 0:
        raise ValueError("2 must divid degree since we are taking covers of the loop")
    base_graph = np.array([np.array[deg]])

    output_is_not_simple = True
    while simple and (output_is_not_simple):
        output_graph = random_cover(
            base_graph=base_graph,
            cover_deg=size,
            permutation_func=random_derangement,  # Derangements remove self loops
            identity_shift=identity_shift,
        )

        if np.max(output_graph) == 1:
            output_is_not_simple = False

    return output_graph
