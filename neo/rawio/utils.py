import io
import mmap
import numpy as np


def get_memmap_shape(filename, dtype, num_channels=None, offset=0):
    dtype = np.dtype(dtype)
    with open(filename, mode="rb") as f:
        f.seek(0, 2)
        flen = f.tell()
        bytes = flen - offset
        if bytes % dtype.itemsize != 0:
            raise ValueError("Size of available data is not a multiple of the data-type size.")
        size = bytes // dtype.itemsize
        if num_channels is None:
            shape = (size,)
        else:
            shape = (size // num_channels, num_channels)
        return shape


def get_memmap_chunk_from_opened_file(fid, num_channels, start, stop, dtype, file_offset=0):
    """
    Utility function to get a chunk as a memmap array directly from an opened file.
    Using this instead memmap can avoid memmory consumption when multiprocessing.

    Similar mechanism is used in spikeinterface.

    """
    bytes_per_sample = num_channels * dtype.itemsize

    # Calculate byte offsets
    start_byte = file_offset + start * bytes_per_sample
    end_byte = file_offset + stop * bytes_per_sample

    # Calculate the length of the data chunk to load into memory
    length = end_byte - start_byte

    # The mmap offset must be a multiple of mmap.ALLOCATIONGRANULARITY
    memmap_offset, start_offset = divmod(start_byte, mmap.ALLOCATIONGRANULARITY)
    memmap_offset *= mmap.ALLOCATIONGRANULARITY

    # Adjust the length so it includes the extra data from rounding down
    # the memmap offset to a multiple of ALLOCATIONGRANULARITY
    length += start_offset

    memmap_obj = mmap.mmap(fid.fileno(), length=length, access=mmap.ACCESS_READ, offset=memmap_offset)

    arr = np.ndarray(
        shape=((stop - start), num_channels),
        dtype=dtype,
        buffer=memmap_obj,
        offset=start_offset,
    )

    return arr


def read_records_from_opened_file(
    fid: io.IOBase, dtype: np.typing.DTypeLike, offset: int, count: int = 1
) -> np.ndarray:
    """
    Read a specific number of records from an opened file.

    Parameters
    ----------
    fid : file object
        The opened file object.
    dtype : data-type
        The data type of the records to read.
    offset : int
        The byte offset in the file to start reading from.
    count : int, optional
        The number of records to read. Default is 1.

    Returns
    -------
    np.ndarray
        An array containing the read records.
    """
    dtype = np.dtype(dtype)
    fid.seek(offset, 0)
    nbytes = count * dtype.itemsize
    data = fid.read(nbytes)
    if len(data) != nbytes:
        raise ValueError(f"Expected {nbytes} bytes at offset {offset}, got {len(data)}")
    return np.frombuffer(data, dtype=dtype)


def read_strided_from_opened_file(
    fid: io.IOBase,
    dtype: np.typing.DTypeLike,
    offset: int,
    stride: int,
    count: int,
    num_values: int | None = None,
) -> np.ndarray:
    """
    Read `count` evenly spaced rows from an opened file with one seek and one read.

    Row `i` starts at `offset + i * stride`. Between rows the file may hold other
    fields (for example, the per-sample header of a Blackrock PTP packet); only the
    bytes from the first row to the end of the last row are read, and the rows are
    returned as a strided view over them.

    Parameters
    ----------
    fid : file object
        The opened file object. It needs only `seek` and `read`, so it can be remote.
    dtype : data-type
        The data type of each value.
    offset : int
        The byte offset of the first value of the first row.
    stride : int
        The number of bytes from the start of one row to the start of the next.
    count : int
        The number of rows to read.
    num_values : int | None, default: None
        Values per row. None returns a 1D array of one value per row.

    Returns
    -------
    np.ndarray
        Shape (count,) when `num_values` is None, else (count, num_values).
    """
    dtype = np.dtype(dtype)
    row_bytes = dtype.itemsize * (1 if num_values is None else num_values)
    shape = (count,) if num_values is None else (count, num_values)
    strides = (stride,) if num_values is None else (stride, dtype.itemsize)
    if count <= 0:
        return np.empty((0,) + shape[1:], dtype=dtype)
    nbytes = (count - 1) * stride + row_bytes
    fid.seek(offset, 0)
    data = fid.read(nbytes)
    if len(data) != nbytes:
        raise ValueError(f"Expected {nbytes} bytes at offset {offset}, got {len(data)}")
    return np.ndarray(shape=shape, dtype=dtype, buffer=data, strides=strides)
