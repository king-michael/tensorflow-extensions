import tensorflow as tf
import numpy as np

# Creates a dataset from a list of npy files
def create_dataset_from_listnpy(npy_files, num_features=None, dtype = tf.float32):
    """
    Creates a TFDataset from a list of npy files.

    Parameters
    ----------
    npy_files : List[str]
        List of npy files.
    num_features : int
        List of features.
    dtype : tf.dtype
        Data type of the data.

    Returns
    -------
    dataset : TFDataset
    """

    # give the header
    def npy_header(npy_path):
        """
        Reads the header of a npy file.

        Parameters
        ----------
        npy_path : str
            npy file

        Returns
        -------
        offset : int
            Byte offset
        header : dict
            Header of the npy file.
        """
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
            dict_header = np.lib.safe_eval(header.decode())
            if not header.endswith(b'\n'):
                raise ValueError('Invalid NPY file.')
            return f.tell(), dict_header

    if num_features is None:
        list_header = [npy_header(f)[1] for f in npy_files]
        list_features = [np.prod(h['shape'][1:]) for h in list_header]
        assert len(set(list_features)) == 1, "Not all files have the same number of features"
        num_features = list_features[0]

    for i,npy_file in enumerate(npy_files):
        header_offset = npy_header(npy_file)[0]
        dataset = tf.data.FixedLengthRecordDataset([npy_file], num_features * dtype.size, header_bytes=header_offset)
        dataset = dataset.map(lambda s: tf.reshape(tf.io.decode_raw(s, dtype), (-1,num_features)))
        if i == 0:
            dataset_comb = dataset
        else:
            dataset_comb = dataset_comb.concatenate(dataset)
        return dataset_comb
