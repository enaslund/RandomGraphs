import numpy as _np


def random_quaternion_matrix():
    """This function returns on of 8 4x4 matrices corresponding to the
    Real irreducible representation of the Quaternion group"""
    i = _np.array([[0.0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]])
    j = _np.array([[0.0, 0, -1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, -1, 0, 0]])
    k = _np.array([[0.0, 0, 0, -1], [0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
    rep_list = [_np.eye(4), -_np.eye(4), i, -i, j, -j, k, -k]
    return rep_list[_np.random.randint(0, len(rep_list))]
