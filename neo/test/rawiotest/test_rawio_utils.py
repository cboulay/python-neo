import io

import numpy as np
import pytest

from neo.rawio.utils import (
    find_timestamp_gaps_in_opened_file,
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


def _reference_gaps(timestamps_ns, resolution, threshold_s):
    """What BlackrockRawIO computes today, on the whole array at once."""
    seconds = np.asarray(timestamps_ns) / resolution
    diffs = np.diff(seconds)
    indices = np.flatnonzero(diffs > threshold_s)
    return indices, diffs[indices]


@pytest.mark.parametrize("packets_per_piece", [1, 7, 100, 1000, 5000])
def test_timestamp_gaps_match_whole_array(packets_per_piece):
    # 1000 packets at 30 kHz (33,333 ns apart), then three 10 ms gaps planted.
    timestamps = np.arange(1000, dtype=np.uint64) * 33_333
    for after in (9, 99, 500):  # 99 -> 100 crosses a piece boundary when packets_per_piece is 100
        timestamps[after + 1 :] += 10_000_000
    data, dt = _ptp_packets(timestamps)

    gaps = find_timestamp_gaps_in_opened_file(
        io.BytesIO(data),
        offset=dt.fields["timestamps"][1],
        packet_size=dt.itemsize,
        num_packets=len(timestamps),
        timestamp_resolution=1e9,
        threshold_s=2 / 30_000,
        packets_per_piece=packets_per_piece,
    )

    expected_indices, expected_sizes = _reference_gaps(timestamps, 1e9, 2 / 30_000)
    np.testing.assert_array_equal(expected_indices, [9, 99, 500])  # the reference itself is right
    np.testing.assert_array_equal(gaps["indices"], expected_indices)
    np.testing.assert_array_equal(gaps["sizes_s"], expected_sizes)


def test_timestamp_gaps_none_found():
    # Evenly spaced, and 7 does not divide 1000, so the last piece is a partial one.
    timestamps = np.arange(1000, dtype=np.uint64) * 33_333
    data, dt = _ptp_packets(timestamps)

    gaps = find_timestamp_gaps_in_opened_file(
        io.BytesIO(data),
        offset=dt.fields["timestamps"][1],
        packet_size=dt.itemsize,
        num_packets=len(timestamps),
        timestamp_resolution=1e9,
        threshold_s=2 / 30_000,
        packets_per_piece=7,
    )

    assert len(gaps["indices"]) == 0
    assert len(gaps["sizes_s"]) == 0


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
