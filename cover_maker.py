import numpy as np
from scipy.sparse import csr_matrix


def random_cover(base_graph, degree, permutation_func, identity_shift):
    """Returns a random cover of the base graph, according to the permutation_func given

    Args:
        base_graph (np.array): The adjacency matrix of the base graph.
            This must be a symmetric square matrix.
        degree (int): The degree of the cover.
        permutation_func: The function used to generate permutations of {1,...,degree}
            Use np.random.permutation for the general case of the symmetric group.
        identity_shift: Shift the resulting adjacency matrix by x*I for some real number
            x, where I is the identity matrix. This helps when computing the most
            negative or the most positive eigenvalues of graphs that where these are
            similar in magnitude.

    Returns:
        scipy.sparse.csr_matrix: A sparse square matrix of size len(base_graph)*degree.
    """
    n = len(base_graph) * degree
    row_ind = np.array([]).astype(int)
    col_ind = np.array([]).astype(int)

    ones = np.ones(degree).astype(int)
    arange = np.arange(degree)
    for i in range(0, len(base_graph)):
        for j in range(i, len(base_graph)):
            if i == j:
                entry = int(base_graph[i, j] / 2)
            else:
                entry = int(base_graph[i, j])

            permutations = []
            for k in range(0, entry):
                permutations.append(permutation_func(degree))

            for random_perm in permutations:
                x_coords = degree * i * ones + arange
                y_coords = degree * j * ones + random_perm
                row_ind = np.append(row_ind, x_coords)
                col_ind = np.append(col_ind, y_coords)

                col_ind = np.append(col_ind, x_coords)
                row_ind = np.append(row_ind, y_coords)

    row_ind = np.append(row_ind, np.arange(n))
    col_ind = np.append(col_ind, np.arange(n))
    data = np.append(
        np.ones(degree * int(np.sum(base_graph))), identity_shift * np.ones(n)
    )  # .astype(int)

    return csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
