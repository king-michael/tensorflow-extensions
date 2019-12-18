import numpy as np

__all__ = ('npy_header', 'npy_header_offset')

# give the header
def npy_header_offset(npy_path):
    """
    Get the byte offset from the header of a npy file.
    Parameters
    ----------
    npy_path : str
        Path to a npy file.

    Returns
    -------
    offset : int
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
        if not header.endswith(b'\n'):
            raise ValueError('Invalid NPY file.')
        return f.tell()


def npy_header(npy_path):
    """
    Get the header of a npy file.

    Parameters
    ----------
    npy_path : str
            Path to a npy file.

    Returns
    -------
    header : dict
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
        return dict_header
