import numpy as np

class NumpyDataset:
    def __init__(self, npyfiles, dtype=None, verbose=False):
        """
        Class that implements a numpy dataset

        Parameters
        ----------
        npyfiles : str
            list of npy files
        dtype : np.dtype or None, optional
            dtype of the data
        verbose : bool, optional
            Turns on Verbosity
        """
        self.verbose = verbose
        if isinstance(npyfiles, str):
            npyfiles = (npyfiles,)

        self.npyfiles = npyfiles

        self.list_parameter = []
        for f in npyfiles:
            offset, header = self._read_npy_header(f)
            assert not header['fortran_order'], "NOT IMPLEMENTED"
            # CODE: use fortran_order
            #    if fortran_order:
            #        array.shape = shape[::-1]
            #        array = array.transpose()
            #    else:
            #        array.shape = shape
            self.list_parameter.append(
                (offset, np.dtype(header['descr']), header['fortran_order'], header['shape'],))

        # get the list of shapes
        self.list_shapes = np.array([s for o, d, f, s in self.list_parameter])
        assert len(set(self.list_shapes[:, 1])), "Dimensions do not fit!"
        if dtype is None:
            dtype = set([d for o, d, f, s in self.list_parameter])
            assert len(dtype) == 1, \
                "Multiple dtypes found. \n{}".format(dtype)
            self.dtype = dtype.pop()

    def get_sample(self, percentage=None, n_samples=None, size=None, replace=False):
        """
        Get a sample of the dataset.

        Only of the options `['percentage', 'n_samples', 'size']` is usable at one time.

        Parameters
        ----------
        percentage : float or None, optional
            Percentage of the dataset to take.
        n_samples : int or None, optional
            Number of samples to draw.
        size : int or None, optional
            Draw X samples based on the given number of memory.
        replace : bool, optional
            Draw samples with replacement or not. Default is `False`.

        Returns
        -------
        sample : np.ndarray
            Sample of shape `(n_samples, *n_dim)`
        """
        if [percentage, n_samples, size].count(None) < 2:
            raise UserWarning("Can't use both options!")

        n_dims = np.multiply.reduce(self.list_shapes[0, 1:])

        index_bounds = np.cumsum(self.list_shapes[:, 0])
        n_samples_max = index_bounds[-1]
        if [percentage, n_samples, size].count(None) == 3:
            if self.verbose:
                print('Use all Samples')
            n_samples = n_samples_max
        elif size is not None:
            n_samples = size // np.dtype(dtype).itemsize // n_dims
            if n_samples >= n_samples_max:
                if self.verbose:
                    print('Use all Samples')
                n_samples = n_samples_max
            assert n_samples != 0, "No Samples where selected"
        elif n_samples is None:
            n_samples = int(n_samples_max * percentage / 100.0)
        assert n_samples <= n_samples_max, "More Samples required then present in the data set"
        if self.verbose:
            print('Draw {} Samples'.format(n_samples))

        data = np.empty((n_samples, n_dims), dtype=self.dtype)

        bincount = np.bincount(np.random.choice(len(self.list_shapes), size=n_samples))
        offset = 0
        for i, (count, npyfile, param) in enumerate(zip(bincount, self.npyfiles, self.list_parameter)):
            # create random indices
            indices = np.random.choice(self.list_shapes[i, 0], size=count, replace=replace)
            if self.verbose:
                print("\rDraw {} samples from {}/{}".format(count, i + 1, len(bincount)), end='')
            data[offset:offset + count] = self.take(npyfile, indices, *param)
            offset += count
        if self.verbose:
            print('\rShuffle                              ', end='')
        np.random.shuffle(data)
        if self.verbose:
            print('\rFinished                             ')
        return data

    def _read_npy_header(self, npy_path):
        # returns offset, dict_header
        with open(str(npy_path), 'rb') as f:
            if f.read(6) != b'\x93NUMPY':
                raise ValueError('Invalid NPY file.')
            version_major, version_minor = f.read(2)
            if version_major == 1:
                header_len_size = 2
            elif version_major == 2:
                header_len_size = 4
            else:
                raise ValueError('Unknown NPY file version {}.{}.'.format(version_major, version_minor))

            header_len = sum(b << (8 * i) for i, b in enumerate(f.read(header_len_size)))
            header = f.read(header_len)
            offset = f.tell()
            dict_header = np.lib.safe_eval(header.decode())
            if not header.endswith(b'\n'):
                raise ValueError('Invalid NPY file.')
            return offset, dict_header

    def _take_npmmap(self, npyfile, indices, *args):
        tmp = np.load(npyfile, mmap_mode='r')
        return np.take(tmp, indices, axis=0)

    def _take_npfrombuffer(self, npyfile, indices, offset, dtype, fortran_order, shape):
        shape_line = np.multiply.reduce(shape[1:])
        len_line = shape_line * dtype.itemsize

        data = np.zeros((len(indices), shape_line), dtype=dtype)
        with open(npyfile, 'rb') as fp:
            for i, idx in enumerate(indices):
                fp.seek(offset + idx * len_line)
                data[i] = np.frombuffer(fp.read(len_line), dtype=dtype)
        return data

    def _take_pymemoryview(self, npyfile, indices, offset, dtype, fortran_order, shape):
        shape_line = np.multiply.reduce(shape[1:])
        len_line = shape_line * dtype.itemsize

        data = np.zeros((len(indices), shape_line), dtype=dtype)
        buf = memoryview(data)
        buf = buf.cast('B')
        with open(npyfile, 'rb') as fp:
            for i, idx in enumerate(indices):
                fp.seek(offset + idx * len_line)
                fp.readinto(buf[i * len_line:(i + 1) * len_line])
        return data

    def _take_raw_npyfile(self, npyfile, indices):
        offset, header = self._read_npy_header(npyfile)

        assert not header['fortran_order'], "NOT IMPLEMENTED"

        dtype = np.dtype(header['descr'])
        itemsize = dtype.itemsize
        shape_line = np.multiply.reduce(header['shape'][1:])
        len_line = shape_line * itemsize

        data = np.zeros((len(indices), shape_line), dtype=dtype)
        with open(npyfile, 'rb') as fp:
            for i, idx in enumerate(indices):
                fp.seek(offset + idx * len_line)
                data[i] = np.frombuffer(fp.read(len_line), dtype=dtype)
        return data

    take = _take_npmmap