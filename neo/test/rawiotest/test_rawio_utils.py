import io

import numpy as np
import pytest

from neo.rawio.utils import (
    read_records_from_opened_file,
    read_strided_from_opened_file,
)


def test_read_matches_fromfile(tmp_path):
    file_path = tmp_path / "testfile"
    dt = np.dtype([("file_id", "S8"), ("ver_major", "uint8"), ("ver_minor", "uint8")])
    records = np.array([(b"NEURALCD", 2, 3), (b"NEURALCD", 3, 0), (b"BRSMPGRP", 3, 0)], dtype=dt)
    records.tofile(file_path)
    with open(file_path, "rb") as f:
        read_records = read_records_from_opened_file(f, dt, offset=0, count=3)
    np.testing.assert_array_equal(read_records, records)
    with open(file_path, "rb") as f:
        read_records = read_records_from_opened_file(f, dt, offset=10, count=2)
    np.testing.assert_array_equal(read_records, records[1:3])


def test_read_works_on_iobytes():
    dt = np.dtype([("file_id", "S8"), ("ver_major", "uint8"), ("ver_minor", "uint8")])
    records = np.array([(b"NEURALCD", 2, 3), (b"NEURALCD", 3, 0), (b"BRSMPGRP", 3, 0)], dtype=dt)
    iobytes = io.BytesIO(records.tobytes())
    read_records = read_records_from_opened_file(iobytes, dt, offset=0, count=3)
    np.testing.assert_array_equal(read_records, records)
    read_records = read_records_from_opened_file(iobytes, dt, offset=10, count=2)
    np.testing.assert_array_equal(read_records, records[1:3])


def test_short_read_raises(tmp_path):
    file_path = tmp_path / "testfile"
    dt = np.dtype([("file_id", "S8"), ("ver_major", "uint8"), ("ver_minor", "uint8")])
    records = np.array([(b"NEURALCD", 2, 3), (b"NEURALCD", 3, 0), (b"BRSMPGRP", 3, 0)], dtype=dt)
    records.tofile(file_path)

    with open(file_path, "rb") as f, pytest.raises(ValueError):
        read_records_from_opened_file(f, dt, offset=10, count=3)


def _ptp_packets(timestamps_ns, num_channels=2):
    """Bytes of fake PTP packets: one per timestamp, samples all zero."""
    dt = np.dtype(
        [
            ("reserved", "u1"),
            ("timestamps", "u8"),
            ("num_data_points", "u4"),
            ("samples", "i2", (num_channels,)),
        ]
    )
    packets = np.zeros(len(timestamps_ns), dtype=dt)
    packets["timestamps"] = timestamps_ns
    packets["num_data_points"] = 1
    return packets.tobytes(), dt


def test_read_strided_matches_structured_fields():
    # Samples and timestamps of PTP packets are fixed strides apart; reading rows
    # 3 to 7 with strides must give exactly those fields of those packets.
    timestamps = np.arange(10, dtype=np.uint64) * 33_333
    data, dt = _ptp_packets(timestamps, num_channels=3)
    packets = np.frombuffer(data, dtype=dt).copy()
    packets["samples"] = np.arange(30, dtype=np.int16).reshape(10, 3)
    fid = io.BytesIO(packets.tobytes())

    samples = read_strided_from_opened_file(
        fid, "int16", offset=3 * dt.itemsize + dt.fields["samples"][1], stride=dt.itemsize, count=5, num_values=3
    )
    np.testing.assert_array_equal(samples, packets["samples"][3:8])

    ticks = read_strided_from_opened_file(
        fid, np.uint64, offset=3 * dt.itemsize + dt.fields["timestamps"][1], stride=dt.itemsize, count=5
    )
    np.testing.assert_array_equal(ticks, timestamps[3:8])

    assert read_strided_from_opened_file(fid, "int16", 0, dt.itemsize, count=0, num_values=3).shape == (0, 3)
    with pytest.raises(ValueError):
        read_strided_from_opened_file(fid, "int16", 8 * dt.itemsize, dt.itemsize, count=5, num_values=3)
