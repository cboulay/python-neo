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


def find_timestamp_gaps_in_opened_file(
    fid: io.IOBase,
    offset: int,
    packet_size: int,
    num_packets: int,
    timestamp_resolution: float,
    threshold_s: float,
    packets_per_piece: int = 1 << 20,
) -> dict:
    """
    Find gaps between consecutive uint64 timestamps that are `packet_size` bytes apart,
    reading them in pieces so memory does not grow with the file.

    A gap is a difference between neighboring timestamps, in seconds, larger than
    `threshold_s`. The arithmetic is the same as `np.diff(timestamps / resolution)` on
    the whole array, so the result matches it exactly; the last timestamp of each piece
    is carried into the next so a gap that falls between two pieces is still found.

    Parameters
    ----------
    fid : file object
        The opened file object. It needs only `seek` and `read`, so it can be remote.
    offset : int
        The byte offset of the first timestamp.
    packet_size : int
        The number of bytes from one timestamp to the next.
    num_packets : int
        The number of timestamps.
    timestamp_resolution : float
        Timestamp ticks per second.
    threshold_s : float
        A difference larger than this, in seconds, is a gap.
    packets_per_piece : int, default: 1 << 20
        How many timestamps to read at once. Memory use is about
        `packets_per_piece * packet_size` bytes.

    Returns
    -------
    dict
        "indices": for each gap, the index of the timestamp just before it (as
        `np.flatnonzero(np.diff(...) > threshold_s)` would give);
        "sizes_s": each gap's size in seconds;
        "before_s": the timestamp before each gap, in seconds;
        "first_s": the first timestamp in seconds, or None if there are none.
    """
    indices, sizes, before = [], [], []
    first_s = None
    last_s = None
    for start in range(0, num_packets, packets_per_piece):
        count = min(packets_per_piece, num_packets - start)
        ticks = read_strided_from_opened_file(fid, np.uint64, offset + start * packet_size, packet_size, count)
        seconds = ticks / timestamp_resolution
        if first_s is None:
            first_s = float(seconds[0])
        # With the previous piece's last timestamp in front, diffs[k] is the gap
        # after timestamp (start - 1 + k); without it, after timestamp (start + k).
        if last_s is None:
            joined, base = seconds, start
        else:
            joined, base = np.concatenate(([last_s], seconds)), start - 1
        diffs = np.diff(joined)
        found = np.flatnonzero(diffs > threshold_s)
        indices.append(found + base)
        sizes.append(diffs[found])
        before.append(joined[found])
        last_s = seconds[-1]

    def _join(parts, dtype):
        return np.concatenate(parts).astype(dtype) if parts else np.empty(0, dtype=dtype)

    return {
        "indices": _join(indices, np.int64),
        "sizes_s": _join(sizes, np.float64),
        "before_s": _join(before, np.float64),
        "first_s": first_s,
    }
