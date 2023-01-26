import numpy as np


def random_derangement(n):
    """Returns a random permutation that has no fixed points"""
    new_perm = np.random.permutation(n)
    to_avoid = np.arange(n)

    coords_to_change = np.where(to_avoid == new_perm)[0]
    while len(coords_to_change) != 0:
        if len(coords_to_change) == 1:
            swap = np.random.randint(0, n)
            new_perm[[coords_to_change[0], swap]] = new_perm[
                [swap, coords_to_change[0]]
            ]
        else:
            swap = coords_to_change[np.random.permutation(len(coords_to_change))]
            new_perm[coords_to_change] = new_perm[swap]
        coords_to_change = np.where(to_avoid == new_perm)[0]
    return new_perm


def random_abelian_cycle(n):
    """Returns one of n possible cycles that permute 1 to n.
    Corresponds to the group action of the elements of an abelian group.
    """
    x = np.arange(n)
    y = np.random.randint(n)
    return np.append(x[y:], x[:y])
