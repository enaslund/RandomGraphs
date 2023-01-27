import numpy as np


def derangement(n):
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


def abelian_cycle(n):
    """Returns one of n possible cycles that permute 1 to n.
    Corresponds to the group action of the elements of an abelian group.
    """
    x = np.arange(n)
    y = np.random.randint(n)
    return np.append(x[y:], x[:y])


def simple_avoiding_others(N, perms_to_avoid):
    """Creates a random permutation that avoids others
    Used to force a graph to be simple."""
    new_perm = np.random.permutation(N)

    num_bad_coords = N
    while num_bad_coords > 0:
        # Numpy argsort corresponds to inverting the permutation
        # For covers of the loop, because the adjacency matrix is forced
        # to be symmetric, we are effectively using perm and perm_inv.
        new_perm_inv = np.argsort(new_perm)

        # If new_perm is equal to new_perm_inv anywhere, it means
        # we have either a fixed point, or a cycle of length 2. Both of these
        # are unacceptable, since they make the resulting graph non-simple
        cond = new_perm == new_perm_inv
        for perm in perms_to_avoid:
            # This condition is True in any coordinates where the new permutation
            # is equal to any of the permutations to avoid, or their
            cond = (
                cond
                | (new_perm == perm)
                | (new_perm == np.argsort(perm))
                | (new_perm_inv == perm)
                | (new_perm_inv == np.argsort(perm))
            )

        good_coords = np.where(~cond)[0]
        coords_to_change = np.where(cond)[0]
        num_bad_coords = len(coords_to_change)

        # Add random elements to be able to escape certain loops
        if len(coords_to_change) > 0:
            if len(good_coords) > 0:
                coords_to_change = np.append(
                    coords_to_change, np.random.choice(good_coords, 1)
                )
            swap = coords_to_change[np.random.permutation(len(coords_to_change))]
            new_perm[coords_to_change] = new_perm[swap]
    return new_perm
