import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix
from . import permutations


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


def random_cover_matrix_rep(*, base_graph, matrix_func, identity_shift):
    """Returns a random cover of the base graph, according to the permutation_func given

    Args:
        base_graph (np.array or csr_matrix): The adjacency matrix of the base graph.
            This must be a symmetric square matrix. Must be simple with no self loops.
        matrix_func (None -> np.ndarray): The function returns a random matrix of a
            specific pre-determined size.
        identity_shift (float): Shift the resulting adjacency matrix by x*I for some
            x, where I is the identity matrix. This helps when computing the most
            negative or the most positive eigenvalues of graphs that where these are
            similar in magnitude.

    Returns:
        scipy.sparse.csr_matrix: Size matrix_size*len(base_graph)
    """
    if np.max(base_graph) > 1:
        raise ValueError("Base Graph must be simple.")

    matrix_size = matrix_func().shape[0]
    n = base_graph.shape[0] * matrix_size

    row_ind = []
    col_ind = []
    data = []

    # This turns the matrix A into a sparse list of entries
    # For covers of extremely large graphs, this is the slowest step
    if not isinstance(base_graph, dok_matrix):
        base_graph = dok_matrix(base_graph)

    ones = np.ones(matrix_size).astype(int)
    arange = np.arange(matrix_size)
    for (i, j), entry in base_graph.items():
        if i <= j:
            continue
        else:
            random_matrix = matrix_func()
            data.append(random_matrix.flatten())
            data.append(random_matrix.flatten())

            for k in range(0, matrix_size):
                x_coords = matrix_size * i * ones + arange
                y_coords = matrix_size * j * ones + k * ones
                row_ind.append(x_coords)
                col_ind.append(y_coords)

            for k in range(0, matrix_size):
                x_coords = matrix_size * i * ones + arange
                y_coords = matrix_size * j * ones + k * ones
                row_ind.append(y_coords)
                col_ind.append(x_coords)

    # Add a multiple of the identity
    row_ind.append(np.arange(n))
    col_ind.append(np.arange(n))
    data.append(identity_shift * np.ones(n))

    row_ind = np.concatenate(row_ind)
    col_ind = np.concatenate(col_ind)
    data = np.concatenate(data)

    return csr_matrix((data, (row_ind, col_ind)), shape=(n, n))


def random_graph(*, deg, size, identity_shift=0):
    """Generates a random graph from the loop. Graph will not necessarily be simple,
    but will have no self-loops.

    Args:
        deg (int): The degree of the graph. Must be even since we are taking covers
            of the loop.
        size (int): The number of vertices of the output graph.
        identity_shift (float): Shift the resulting adjacency matrix by x*I for some
            real number x, where I is the identity matrix. This helps when computing the
            most negative or positive eigenvalues of graphs that where these are
            similar in magnitude.

    Returns:
        scipy.sparse.csr_matrix: A sparse square adjacency matrix of size size
    """
    if deg % 2 != 0:
        raise ValueError("2 must divide degree since we are taking covers of the loop")
    base_graph = np.array([np.array([deg])])

    output_graph = random_cover(
        base_graph=base_graph,
        cover_deg=size,
        # We use derangements to remove self loops
        permutation_func=permutations.derangement,
        identity_shift=identity_shift,
    )
    return output_graph


def random_simple_graph(*, deg, size, identity_shift=0):
    """Generates a random simple graph as a cover of the loop. The permutations used
    are generated to avoid overlap with any of the previously generated permutations.
    This occurs in the permutation function derangement_avoiding_others

    Args:
        deg (int): The degree of the graph. Must be even since we are taking covers
            of the loop.
        size (int): The number of vertices of the output graph.
        identity_shift (float): Shift the resulting adjacency matrix by x*I for some
            real number x, where I is the identity matrix. This helps when computing the
            most negative or positive eigenvalues of graphs that where these are
            similar in magnitude.

    Returns:
        scipy.sparse.csr_matrix: A sparse square adjacency matrix of size size
    """
    if deg % 2 != 0:
        raise ValueError("2 must divide deg since we are taking covers of the loop")

    row_ind = []
    col_ind = []
    arange = np.arange(size)

    random_perms = []
    for k in range(0, int(deg / 2)):
        random_perm = permutations.derangement_avoiding_others(size, random_perms)
        random_perms.append(random_perm)

    for random_perm in random_perms:
        x_coords = arange
        y_coords = random_perm
        row_ind.append(x_coords)
        col_ind.append(y_coords)

        col_ind.append(x_coords)
        row_ind.append(y_coords)

    row_ind.append(np.arange(size))
    col_ind.append(np.arange(size))

    row_ind = np.concatenate(row_ind)
    col_ind = np.concatenate(col_ind)

    data = np.append(
        np.ones(size * deg), identity_shift * np.ones(size)
    )  # .astype(int)

    return csr_matrix((data, (row_ind, col_ind)), shape=(size, size))
