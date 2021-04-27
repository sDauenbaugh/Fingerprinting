import numpy as np


def smooth(arr, half_size):
    win_size = 2 * half_size + 1
    window = np.ones((win_size, 1))
    window[half_size] = 0

    arr = np.copy(arr)
    arr = np.pad(arr, (half_size, half_size), 'edge')

    new_arr = []
    for i in range(0, arr.shape[0] - win_size + 1):
        vals = window * arr[i:i + win_size]
        mid = np.median(vals)
        count = vals[vals == mid].shape[0]
        if count >= half_size + 1:
            new_arr.append(int(mid))
        else:
            new_arr.append(int(arr[i + half_size]))
    new_arr = np.asarray(new_arr)
    return new_arr


if __name__ == "__main__":
    arr1 = np.array([6, 6, 6, 5, 5, 5, 5, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5,
                    4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 9, 9, 9, 9, 9,
                    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 11, 11, 11, 11,
                    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
                    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 13, 13,
                    13, 13, 13, 13, 13, 3, 3, 3, 3, 13, 13, 13, 37, 37, 37, 37, 37,
                    37, 37, 37, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                    7, 7, 7, 8, 8, 8]).reshape((125, 1))
    res = smooth(arr1, 4)
    print(res)
