import numpy as np


def random_permutation(N, perms_to_avoid):
    """Creates a random permutation that avoids others
    Can be used to force a graph to be simple."""
    new_perm = np.random.permutation(N)

    num_bad_coords = N
    while num_bad_coords > 0:
        cond = None
        new_perm_inv = np.argsort(new_perm)
        for perm in perms_to_avoid:
            if cond is None:
                cond = (new_perm == perm) | (new_perm == np.argsort(perm))
                cond = (
                    cond | (perm == new_perm_inv) | (np.argsort(perm) == new_perm_inv)
                )
            else:
                cond = cond | (new_perm == perm) | (new_perm == np.argsort(perm))
                cond = (
                    cond | (perm == new_perm_inv) | (np.argsort(perm) == new_perm_inv)
                )
        cond = cond | (new_perm == new_perm_inv)
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
