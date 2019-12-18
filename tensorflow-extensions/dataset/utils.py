import numpy as np

def get_batch(array, batchsize, axis=0):
    """
    Creates a generator for an array, returning a batch along the first axis.

    Parameters
    ----------
    array : np.ndarray
    batchsize : int
        Batch size
    axis : int, optional
        Axis to use. Default is `0`.

    Returns
    -------
    Iterator

    Yields
    ------
    np.ndarray
        batch of shape `(batchsize, ...)`.
    """
    i = 0
    n_samples = len(array)
    array = np.moveaxis(array, axis, 0)
    while True:
        if i + batchsize >= n_samples:
            yield np.vstack([array[i:], array[:i+batchsize-n_samples]])
            i = (i+batchsize)- n_samples
        else:
            yield array[i:i+batchsize]
            i += batchsize

def test_get_batch():
    data = np.arange(20).reshape((-1,1))
    reps = 10
    batchsize = 3

    it = get_batch(data, batchsize)
    test = np.array([next(it) for i in range(reps)]).flatten()
    ref = (np.tile(data,np.ceil(batchsize * reps  / len(data)).astype(int)).T.flatten())[:test.size]
    np.testing.assert_array_equal(test, ref)


def get_a_random_sample_from_list(list_data, percentage=None, n_samples=None, dtype=np.float64, replace=True):
    """
    Draw a random sample from a list of data.

    Parameters
    ----------
    list_data : List[np.ndarray]
        List of numpy arrays
    percentage : float or None, optional
        Percentage of the dataset to take.
    n_samples : int or None, optional
        Number of samples to draw.
    dtype : np.dtype, optional
        Data type of the samples. Default is `np.float64`.
    replace : bool, optional
        Draw samples with replacement or not. Default is `False`.

    Returns
    -------
    sample : np.ndarray
            Sample of shape `(n_samples, *n_dim)`
    """
    if percentage is not None and n_samples is not None:
        raise UserWarning("Can't use both options!")
    list_shapes = np.array([l.shape for l in list_data])
    assert len(set(list_shapes[:,1])), "Dimensions do not fit!"
    n_dims = list_shapes[0,1]

    index_bounds = np.cumsum(list_shapes[:,0])
    n_samples_max = index_bounds[-1]

    if percentage is None and n_samples is None:
        print('Use all Samples')
        n_samples = n_samples_max
    elif n_samples is None:
        n_samples = int(n_samples_max*percentage/100)
    assert n_samples <= n_samples_max, "More Samples required then present in the data set"
    print('Draw {} Samples'.format(n_samples))

    bincount = np.bincount(np.random.choice(len(list_shapes), size=n_samples))
    data = np.empty((n_samples, n_dims), dtype=dtype)
    offset = 0
    for i, count in enumerate(bincount):
        print("\rDraw {} samples from {}/{}".format(count, i+1, len(bincount)), end='')
       # print("Draw {} samples from {}/{}".format(count, i+1, len(bincount)))
        data[offset:offset+count] = list_data[i][np.random.choice(list_data[i].shape[0],
                                                                    size=count, replace=replace)]
        offset += count
    print('\rFinished                             ')
    return data