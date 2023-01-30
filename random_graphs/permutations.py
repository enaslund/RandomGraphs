import numpy as _np  # Underscore to not polute namespace


def abelian_cycle(n):
    """Returns one of n possible cycles that permute 1 to n.
    Corresponds to the group action of the elements of an abelian group.
    """
    x = _np.arange(n)
    y = _np.random.randint(n)
    return _np.append(x[y:], x[:y])


def derangement(n):
    """Returns a random permutation that has no fixed points"""
    new_perm = _np.random.permutation(n)
    to_avoid = _np.arange(n)

    coords_to_change = _np.where(to_avoid == new_perm)[0]
    while len(coords_to_change) != 0:
        if len(coords_to_change) == 1:
            swap = _np.random.randint(0, n)
            new_perm[[coords_to_change[0], swap]] = new_perm[
                [swap, coords_to_change[0]]
            ]
        else:
            swap = coords_to_change[_np.random.permutation(len(coords_to_change))]
            new_perm[coords_to_change] = new_perm[swap]
        coords_to_change = _np.where(to_avoid == new_perm)[0]
    return new_perm


def derangement_avoiding_others(N, perms_to_avoid):
    """Creates a random permutation that avoids others
    Used to force a graph to be simple."""
    new_perm = _np.random.permutation(N)

    num_bad_coords = N
    while num_bad_coords > 0:
        # Numpy argsort corresponds to inverting the permutation
        # For covers of the loop, because the adjacency matrix is forced
        # to be symmetric, we are effectively using perm and perm_inv.
        new_perm_inv = _np.argsort(new_perm)

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
                | (new_perm == _np.argsort(perm))
                | (new_perm_inv == perm)
                | (new_perm_inv == _np.argsort(perm))
            )

        good_coords = _np.where(~cond)[0]
        coords_to_change = _np.where(cond)[0]
        num_bad_coords = len(coords_to_change)

        # Add random elements to be able to escape certain loops
        if len(coords_to_change) > 0:
            if len(good_coords) > 0:
                coords_to_change = _np.append(
                    coords_to_change, _np.random.choice(good_coords, 1)
                )
            swap = coords_to_change[_np.random.permutation(len(coords_to_change))]
            new_perm[coords_to_change] = new_perm[swap]
    return new_perm


def quaternion_matrix():
    """This function returns on of 8 4x4 matrices corresponding to the
    Real irreducible representation of the Quaternion group"""
    i = _np.array([[0.0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]])
    j = _np.array([[0.0, 0, -1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, -1, 0, 0]])
    k = _np.array([[0.0, 0, 0, -1], [0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
    rep_list = [_np.eye(4), -_np.eye(4), i, -i, j, -j, k, -k]
    return rep_list[_np.random.randint(0, len(rep_list))]
