import io

import numpy as np
import pytest

from neo.rawio.utils import read_records_from_opened_file


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
