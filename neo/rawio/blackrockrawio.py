"""
Module for reading data from files in the Blackrock in raw format.

This work is based on:
  * Chris Rodgers - first version
  * Michael Denker, Lyuba Zehl - second version
  * Samuel Garcia - third version
  * Lyuba Zehl, Michael Denker - fourth version
  * Samuel Garcia, Julia Srenger - fifth version
  * Chadwick Boulay - FileSpec 3.0 and 3.0-PTP
  * Heberto Mayorquin - Time segmentation fixes, reporting and refactoring

This IO supports reading only.
This IO is able to read:
  * the nev file which contains spikes
  * ns1, ns2, .., ns6 files that contain signals at different sampling rates

This IO can handle the following Blackrock file specifications:
  * 2.1
  * 2.2
  * 2.3
  * 3.0
  * 3.0 with PTP timestamps (Gemini systems)

The neural data channels are 1 - 128.
The analog inputs are 129 - 144. (129 - 137 AC coupled, 138 - 144 DC coupled)

spike- and event-data; 30000 Hz in NEV file.
"ns1": "analog data: 500 Hz",
"ns2": "analog data: 1000 Hz",
"ns3": "analog data: 2000 Hz",
"ns4": "analog data: 10000 Hz",
"ns5": "analog data: 30000 Hz",
"ns6": "analog data: 30000 Hz (no digital filter)"


The possible file extensions of the Cerebus system and their content:
    ns1: contains analog data; sampled at 500 Hz (+ digital filters)
    ns2: contains analog data; sampled at 1000 Hz (+ digital filters)
    ns3: contains analog data; sampled at 2000 Hz (+ digital filters)
    ns4: contains analog data; sampled at 10000 Hz (+ digital filters)
    ns5: contains analog data; sampled at 30000 Hz (+ digital filters)
    ns6: contains analog data; sampled at 30000 Hz (no digital filters)
    nev: contains spike- and event-data; sampled at 30000 Hz
    sif: contains institution and patient info (XML)
    ccf: contains Cerebus configurations

TODO:
  * videosync events (file spec 2.3)
  * tracking events (file spec 2.3)
  * buttontrigger events (file spec 2.3)
  * config events (file spec 2.3)
  * check left sweep settings of Blackrock
  * check nsx offsets (file spec 2.1)
  * add info of nev ext header (NSASEXEX) to non-neural events
    (file spec 2.1 and 2.2)
  * read sif file information
  * read ccf file information
  * fix reading of periodic sampling events (non-neural event type)
    (file spec 2.1 and 2.2)
"""

import datetime
import io
import mmap
import os
import re
import warnings

import numpy as np
import quantities as pq

from .baserawio import (
    BaseRawIO,
    _signal_channel_dtype,
    _signal_stream_dtype,
    _signal_buffer_dtype,
    _spike_channel_dtype,
    _event_channel_dtype,
)
from .utils import (
    find_timestamp_gaps_in_opened_file,
    read_records_from_opened_file,
    read_strided_from_opened_file,
)

from neo.core import NeoReadWriteError

# Central sends up to 300 samples (10 ms) per UDP datagram, and the recording PC can write datagrams out of
# order. A packet stamped at most this much earlier than the one written before it is such a packet, not a
# clock reset: it is read in timestamp order.
REORDER_WINDOW_S = 0.010


class BlackrockRawIO(BaseRawIO):
    """
    Class for reading data in from a file set recorded by the Blackrock (Cerebus) recording system.
    Upon initialization, the class is linked to the available set of Blackrock files.

    Parameters
    ----------
    filename: str, default: ''
        File name (without extension) of the set of Blackrock files to associate with.
        Any .nsX or .nev, .sif, or .ccf extensions are ignored when parsing this parameter.
    nsx_override: str | None, default: None
        File name of the .nsX files (without extension). If None, filename is used.
    nev_override: str | None, default: None
        File name of the .nev file (without extension). If None, filename is used.
    nsx_to_load: int | list | 'max' | 'all' | None, default: None
        IDs of nsX file from which to load data, e.g., if set to 5 only data from the ns5 file are loaded.
        If 'all', then all nsX will be loaded. Contrary to previous version of the IO  (<0.7), nsx_to_load
        must be set at the init before parse_header().
    load_nev: bool, default: True
        Load (or not) events/spikes by ignoring or not the nev file.
    gap_tolerance_ms : float | None, default: None
        Maximum acceptable gap size in milliseconds for automatic segmentation.

        **Default behavior (None)**: If timestamp gaps are detected, an error is raised
        with a detailed gap report. This ensures users are aware of data discontinuities.

        **Opt-in segmentation**: Provide a value to automatically segment data at gaps
        larger than this threshold. Gaps smaller than the threshold are ignored (data
        treated as continuous).

        Examples:

        - None (default): Error on any detected gaps
        - 1.0: Tolerate gaps up to 1 ms, segment on larger gaps
        - 10.0: Tolerate gaps up to 10 ms (filters buffer artifacts)
        - 100.0: Tolerate gaps up to 100 ms (only major pauses create segments)

        Applies to:

        - PTP format (v3.0-ptp): Gaps in per-sample timestamps
        - Standard format (v2.2/2.3/3.0): Gaps between data blocks

    storage_options : dict | None, default: None
        Dictionary of storage options to pass to the underlying file system.
        This can include options such as credentials, caching behavior, etc.

    Notes
    -----
    * Note: This routine will handle files according to specification 2.1, 2.2,
    2.3, 3.0 and 3.0-ptp. Recording pauses that may occur in file specifications
    2.2 and 2.3 are automatically extracted and the data set is split into different
    segments.

    * The times of spikes and events are compared with `t_start` and `t_stop` as float64 seconds, and
    `t_stop` is exclusive. To include a spike at `t_stop`, such as the last of a segment, pass
    `t_stop + 1e-5`.

    * Packets written out of their timestamp order (as when the recording PC receives Central's datagrams out
    of order) are read in timestamp order when each is within 10 ms of its place: in a PTP .nsX, a step back
    whose neighbourhood sorts into an even grid is neither a gap nor a new segment; in a .nev, a packet stamped
    up to 10 ms earlier than the last of its type is no clock reset, and each unit's spikes are returned in
    timestamp order. They are counted in `nsx_reordered_packets` and `nev_reordered_packets`, with a warning.

    * The Blackrock data format consists not of a single file, but a set of
    different files. This constructor associates itself with a set of files
    that constitute a common data set. By default, all files belonging to
    the file set have the same base name, but different extensions.
    However, by using the override parameters, individual filenames can
    be set.

    Examples
    --------
    >>> import neo.rawio
    >>> # Inspect a set of file consisting of files FileSpec2.3001.ns5 and FileSpec2.3001.nev
    >>> reader = neo.rawio.BlackrockRawIO(filename='FileSpec2.3001', nsx_to_load=5)
    >>> reader.parse_header()
    >>> print(reader)


    """

    extensions = ["ns" + str(_) for _ in range(1, 7)]
    extensions.extend(["nev", "sif", "ccf"])  # 'sif', 'ccf' not yet supported
    rawmode = "multi-file"

    # We need to document the origin of this value
    main_sampling_rate = 30000.0

    def __init__(
        self,
        filename=None,
        nsx_override=None,
        nev_override=None,
        nsx_to_load=None,
        load_nev=True,
        verbose=False,
        gap_tolerance_ms=None,
        storage_options=None,
    ):
        BaseRawIO.__init__(self)

        self.gap_tolerance_ms = gap_tolerance_ms
        # Packets read at once when a remote NEV is scanned in pieces (about 27 MB at 104-byte packets).
        self._nev_packets_per_piece = 1 << 18
        # Summary of a NEV's spikes when it was scanned in pieces instead of read into memory.
        self._nev_spikes = None
        # NEV data packets that match no packet type for the file's spec, by packet id: they are
        # not read, so a caller can refuse a file whose data would otherwise be lost silently.
        self.nev_unmatched_packets = {}
        # Packets the file holds out of their timestamp order (within REORDER_WINDOW_S), read in timestamp
        # order: steps back in the NEV (per packet type), and per nsX number in PTP .nsX files.
        self.nev_reordered_packets = 0
        self.nsx_reordered_packets = {}

        self.filename = str(filename)
        if "://" in self.filename:
            try:
                import fsspec
            except ImportError:
                raise ImportError("fsspec is required to handle URLs, run pip install neo[remote]")
            self._fs, _ = fsspec.core.url_to_fs(self.filename, **(storage_options or {}))
        else:
            self._fs = None

        # remove extension from base _filenames
        for ext in self.extensions:
            if self.filename.endswith(os.path.extsep + ext):
                self.filename = self.filename.replace(os.path.extsep + ext, "")

        self.nsx_to_load = nsx_to_load

        # remove extensions from overrides
        self._filenames = {}
        if nsx_override:
            self._filenames["nsx"] = re.sub(os.path.extsep + "ns[1,2,3,4,5,6]$", "", nsx_override)
        else:
            self._filenames["nsx"] = self.filename
        if nev_override:
            self._filenames["nev"] = re.sub(os.path.extsep + "nev$", "", nev_override)
        else:
            self._filenames["nev"] = self.filename

        self._filenames["sif"] = self.filename
        self._filenames["ccf"] = self.filename

        # check which files are available
        self._avail_files = dict.fromkeys(self.extensions, False)
        self._avail_nsx = []
        for ext in self.extensions:
            if ext.startswith("ns"):
                file2check = "".join([self._filenames["nsx"], os.path.extsep, ext])
            else:
                file2check = "".join([self._filenames[ext], os.path.extsep, ext])

            if self._exists(file2check):
                self._avail_files[ext] = True
                if ext.startswith("ns"):
                    self._avail_nsx.append(int(ext[-1]))

        if not load_nev:
            self._avail_files["nev"] = False

        if not self._avail_files["nev"] and not self._avail_nsx:
            raise IOError("No Blackrock files found in specified path")

        # These dictionaries are used internally to map the file specification
        # revision of the nsx and nev files to one of the reading routines
        # NEV
        self._waveform_size = {
            "2.1": self._get_waveform_size_spec_v21,
            "2.2": self._get_waveform_size_spec_v21,
            "2.3": self._get_waveform_size_spec_v22_30,
            "3.0": self._get_waveform_size_spec_v22_30,
        }
        self._channel_labels = {
            "2.1": self._get_channel_labels_spec_v21,
            "2.2": self._get_channel_labels_spec_v22_30,
            "2.3": self._get_channel_labels_spec_v22_30,
            "3.0": self._get_channel_labels_spec_v22_30,
        }
        self._nonneural_evdicts = {
            "2.1": self._get_nonneural_evdicts_spec_v21_22,
            "2.2": self._get_nonneural_evdicts_spec_v21_22,
            "2.3": self._get_nonneural_evdicts_spec_v23,
            "3.0": self._get_nonneural_evdicts_spec_v23,
        }
        self._comment_evdict = {
            "2.1": self._get_comment_evdict_spec_v21_22,
            "2.2": self._get_comment_evdict_spec_v21_22,
            "2.3": self._get_comment_evdict_spec_v21_22,
            "3.0": self._get_comment_evdict_spec_v21_22,
        }

    def _exists(self, path: str) -> bool:
        if self._fs is None:
            return os.path.exists(path)
        return self._fs.exists(path)

    def _open(self, path: str) -> io.IOBase:
        if self._fs is None:
            return open(path, "rb")
        return self._fs.open(path, "rb")

    def _parse_header(self):

        event_channels = []
        spike_channels = []
        signal_buffers = []
        signal_streams = []
        signal_channels = []

        # Step1 NEV file
        if self._avail_files["nev"]:
            # Load file spec and headers of available

            # read nev file specification
            self._nev_spec = self._extract_nev_file_spec()

            # read nev headers
            nev_filename = f"{self._filenames['nev']}.nev"
            self._nev_basic_header, self._nev_ext_header = self._read_nev_header(self._nev_spec)

            self.nev_data = self._read_nev_data(self._nev_spec, nev_filename)

            # scan all channel to get number of Unit
            spike_channels = []
            self.internal_unit_ids = []  # pair of chan['packet_id'], spikes['unit_class_nb']
            for i in range(len(self._nev_ext_header[b"NEUEVWAV"])):

                # electrode_id values are stored at uint16 which can overflow when
                # multiplying by 1000 below. We convert to a regular python int which
                # won't overflow
                channel_id = int(self._nev_ext_header[b"NEUEVWAV"]["electrode_id"][i])

                # all `unit_class_nb` is uint8. Also will have issues with overflow
                # cast this to python int
                all_unit_id = self._nev_unit_ids(channel_id)
                for u, unit_id in enumerate(all_unit_id):
                    unit_id = int(unit_id)
                    self.internal_unit_ids.append((channel_id, unit_id))
                    name = f"ch{channel_id}#{unit_id}"
                    _id = f"Unit {1000 * channel_id + unit_id}"
                    wf_gain = self._nev_params("digitization_factor")[channel_id] / 1000.0
                    wf_offset = 0.0
                    wf_units = "uV"
                    # TODO: Double check if this is the correct assumption (10 samples)
                    # default value: threshold crossing after 10 samples of waveform
                    wf_left_sweep = 10
                    wf_sampling_rate = self.main_sampling_rate
                    spike_channels.append((name, _id, wf_units, wf_gain, wf_offset, wf_left_sweep, wf_sampling_rate))

            # scan events
            # NonNeural: serial and digital input
            events_data, event_segment_ids = self.nev_data["NonNeural"]
            ev_dict = self._nonneural_evdicts[self._nev_spec](events_data)
            if "Comments" in self.nev_data:
                comments_data, comments_segment_ids = self.nev_data["Comments"]
                ev_dict.update(self._comment_evdict[self._nev_spec](comments_data))
            for ev_name in ev_dict:
                event_channels.append((ev_name, "", "event"))
            # TODO: TrackingEvents
            # TODO: ButtonTrigger
            # TODO: VideoSync

        # Step2 NSX file
        # Load file spec and headers of available nsx files
        self._nsx_spec = {}
        self._nsx_basic_header = {}
        self._nsx_ext_header = {}
        self._nsx_data_header = {}
        self._nsx_sampling_frequency = {}

        # Read headers
        for nsx_nb in self._avail_nsx:
            spec_version = self._nsx_spec[nsx_nb] = self._extract_nsx_file_spec(nsx_nb)
            # read nsx headers
            self._nsx_basic_header[nsx_nb], self._nsx_ext_header[nsx_nb] = self._read_nsx_header(spec_version, nsx_nb)

            # The Blackrock defines period as the number of  1/30_000 seconds between data points
            # E.g. it is 1 for 30_000, 3 for 10_000, etc
            nsx_period = self._nsx_basic_header[nsx_nb]["period"]
            sampling_rate = 30_000.0 / nsx_period
            self._nsx_sampling_frequency[nsx_nb] = float(sampling_rate)

        # nsx_to_load can be either int, list, 'max', 'all' (aka None)
        # here make a list only
        if self.nsx_to_load is None or self.nsx_to_load == "all":
            self.nsx_to_load = list(self._avail_nsx)
        elif self.nsx_to_load == "max":
            if len(self._avail_nsx):
                self.nsx_to_load = [max(self._avail_nsx)]
            else:
                self.nsx_to_load = []
        elif isinstance(self.nsx_to_load, int):
            self.nsx_to_load = [self.nsx_to_load]
        elif isinstance(self.nsx_to_load, list):
            pass
        else:
            raise (ValueError("nsx_to_load is wrong"))

        missing_nsx_files = [nsx_nb for nsx_nb in self.nsx_to_load if nsx_nb not in self._avail_nsx]
        if missing_nsx_files:
            missing_list = ", ".join(f"ns{nsx_nb}" for nsx_nb in missing_nsx_files)
            raise FileNotFoundError(
                f"Requested NSX file(s) not found: {missing_list}. Available NSX files: {self._avail_nsx}"
            )

        # check that all files come from the same specification
        all_spec = [self._nsx_spec[nsx_nb] for nsx_nb in self.nsx_to_load]
        if self._avail_files["nev"]:
            all_spec.append(self._nev_spec)
        if not all(all_spec[0] == spec for spec in all_spec):
            raise NeoReadWriteError("Files don't have the same internal version")

        if len(self.nsx_to_load) > 0 and self._nsx_spec[self.nsx_to_load[0]] == "2.1" and not self._avail_files["nev"]:
            pass
            # Because rescaling to volts requires information from nev file (dig_factor)
            # Remove if raw loading becomes possible
            # raise IOError("For loading Blackrock file version 2.1 .nev files are required!")

        # Keep public attribute for backward compatibility but let's use the private one and maybe deprecate this at some point
        self.sig_sampling_rates = {
            nsx_number: self._nsx_sampling_frequency[nsx_number] for nsx_number in self.nsx_to_load
        }
        ptp_files_seen = False
        if len(self.nsx_to_load) > 0:
            for nsx_nb in self.nsx_to_load:
                basic_header = self._nsx_basic_header[nsx_nb]
                spec_version = self._nsx_spec[nsx_nb]
                # The only way to know if it is the Precision Time Protocol of file spec 3.0
                # is to check for nanosecond timestamp resolution.
                is_ptp_variant = (
                    "timestamp_resolution" in basic_header.dtype.names
                    and basic_header["timestamp_resolution"] == 1_000_000_000
                )
                if is_ptp_variant:
                    data_spec = "3.0-ptp"
                    ptp_files_seen = True
                else:
                    data_spec = spec_version

                # Parse data blocks (extracts offsets, sample counts, timestamps)
                parsed_data_headers = self._parse_nsx_data(data_spec, nsx_nb)

                # Segment the data (analyzes gaps, creates per-segment metadata)
                self._nsx_data_header[nsx_nb] = self._segment_nsx_data(parsed_data_headers, nsx_nb)

                # Match NSX and NEV segments for v2.3
                if self._avail_files["nev"]:
                    self._match_nsx_and_nev_segment_ids(nsx_nb)

                sr = self._nsx_sampling_frequency[nsx_nb]

                if spec_version in ["2.2", "2.3", "3.0"]:
                    ext_header = self._nsx_ext_header[nsx_nb]
                elif spec_version == "2.1":
                    # v2.1 has no extended headers - construct from NEV digitization factors
                    ext_header = self._build_nsx_v21_ext_header(nsx_nb)

                if len(ext_header) > 0:
                    # in blackrock : one stream per buffer so same id
                    buffer_id = stream_id = str(nsx_nb)
                    stream_name = f"nsx{nsx_nb}"
                    signal_buffers.append((stream_name, buffer_id))
                    signal_streams.append((stream_name, stream_id, buffer_id))
                for i, chan in enumerate(ext_header):
                    if spec_version in ["2.2", "2.3", "3.0"]:
                        ch_name = chan["electrode_label"].decode()
                        ch_id = str(chan["electrode_id"])
                        units = chan["units"].decode()
                    elif spec_version == "2.1":
                        ch_name = chan["labels"]
                        ch_id = str(self._nsx_ext_header[nsx_nb][i]["electrode_id"])
                        units = chan["units"]
                    sig_dtype = "int16"
                    # max_analog_val/min_analog_val/max_digital_val/min_analog_val are int16!!!!!
                    # dangerous situation so cast to float everyone
                    if np.isnan(float(chan["min_analog_val"])):
                        gain = 1
                        offset = 0
                    else:
                        gain = (float(chan["max_analog_val"]) - float(chan["min_analog_val"])) / (
                            float(chan["max_digital_val"]) - float(chan["min_digital_val"])
                        )
                        offset = -float(chan["min_digital_val"]) * gain + float(chan["min_analog_val"])
                    buffer_id = stream_id = str(nsx_nb)
                    signal_channels.append((ch_name, ch_id, sr, sig_dtype, units, gain, offset, stream_id, buffer_id))

            # check nb segment per nsx
            nb_segments_for_nsx = [len(self._nsx_data_header[nsx_nb]) for nsx_nb in self.nsx_to_load]
            if not all(nb == nb_segments_for_nsx[0] for nb in nb_segments_for_nsx):
                raise NeoReadWriteError("Segment nb not consistent across nsX files")
            self._nb_segment = nb_segments_for_nsx[0]

            self._delete_empty_segments()

            # t_start/t_stop for segment are given by nsx limits or nev limits
            self._sigs_t_starts = {nsx_nb: [] for nsx_nb in self.nsx_to_load}
            self._seg_t_starts, self._seg_t_stops = [], []
            for data_bl in range(self._nb_segment):
                t_stop = 0.0
                for nsx_nb in self.nsx_to_load:
                    spec = self._nsx_spec[nsx_nb]
                    if "timestamp_resolution" in self._nsx_basic_header[nsx_nb].dtype.names:
                        ts_res = self._nsx_basic_header[nsx_nb]["timestamp_resolution"]
                    elif spec == "2.1":
                        ts_res = 30_000  # v2.1 always uses 30kHz timestamp resolution
                    else:
                        ts_res = 30_000
                    period = self._nsx_basic_header[nsx_nb]["period"]
                    sec_per_samp = period / 30_000  # Maybe 30_000 should be ['sample_resolution']
                    seg_header = self._nsx_data_header[nsx_nb][data_bl]
                    length = seg_header["nb_data_points"]
                    timestamp = seg_header["timestamp"]

                    if "timestamps_memmap_kwargs" in seg_header:
                        # FileSpec 3.0 with PTP -- read first/last timestamps on demand
                        ts_kw = seg_header["timestamps_memmap_kwargs"]
                        first_ts = self._read_ptp_ticks(nsx_nb, ts_kw, 0, 1)[0]
                        last_ts = self._read_ptp_ticks(nsx_nb, ts_kw, length - 1, length)[0]
                        t_start = float(first_ts) / ts_res
                        t_stop = max(t_stop, float(last_ts) / ts_res + sec_per_samp)
                    elif timestamp is None:
                        # V2.1 format has no timestamps
                        t_start = 0.0
                        t_stop = max(t_stop, length / self._nsx_sampling_frequency[nsx_nb])
                    else:
                        # Standard format with scalar timestamp
                        t_start = timestamp / ts_res
                        t_stop = max(t_stop, t_start + length / self._nsx_sampling_frequency[nsx_nb])
                    self._sigs_t_starts[nsx_nb].append(t_start)

                if self._avail_files["nev"]:
                    max_nev_time = 0
                    for k, (data, ev_ids) in self.nev_data.items():
                        segment_mask = ev_ids == data_bl
                        if data[segment_mask].size > 0:
                            t = data[segment_mask]["timestamp"].max() / self._nev_basic_header["timestamp_resolution"]

                            max_nev_time = max(max_nev_time, t)
                    if self._nev_spikes is not None and data_bl in self._spike_times_by_segment():
                        t = self._spike_times_by_segment()[data_bl][3] / self._nev_basic_header["timestamp_resolution"]
                        max_nev_time = max(max_nev_time, t)
                    if max_nev_time > t_stop:
                        t_stop = max_nev_time
                    min_nev_time = max_nev_time
                    for k, (data, ev_ids) in self.nev_data.items():
                        segment_mask = ev_ids == data_bl
                        if data[segment_mask].size > 0:
                            t = data[segment_mask]["timestamp"].min() / self._nev_basic_header["timestamp_resolution"]
                            min_nev_time = min(min_nev_time, t)
                    if self._nev_spikes is not None and data_bl in self._spike_times_by_segment():
                        t = self._spike_times_by_segment()[data_bl][2] / self._nev_basic_header["timestamp_resolution"]
                        min_nev_time = min(min_nev_time, t)
                    if min_nev_time < t_start:
                        t_start = min_nev_time
                self._seg_t_starts.append(t_start)
                self._seg_t_stops.append(float(t_stop))

        else:
            # When only nev is available, only segments that are documented in nev can be detected

            max_nev_times = {}
            min_nev_times = {}

            # Find maximal and minimal time for each nev segment
            for k, (data, ev_ids) in self.nev_data.items():
                for i in np.unique(ev_ids):
                    curr_data = data[ev_ids == i]
                    if curr_data.size > 0:
                        if max(curr_data["timestamp"]) >= max_nev_times.get(i, 0):
                            max_nev_times[i] = max(curr_data["timestamp"])
                        if min(curr_data["timestamp"]) <= min_nev_times.get(i, max_nev_times[i]):
                            min_nev_times[i] = min(curr_data["timestamp"])
            if self._nev_spikes is not None:
                for i, (_first, _last, lowest, highest) in self._spike_times_by_segment().items():
                    if highest >= max_nev_times.get(i, 0):
                        max_nev_times[i] = highest
                    if lowest <= min_nev_times.get(i, max_nev_times[i]):
                        min_nev_times[i] = lowest

            # Calculate t_start and t_stop for each segment in seconds
            resolution = self._nev_basic_header["timestamp_resolution"]
            self._seg_t_starts = [v / float(resolution) for k, v in sorted(min_nev_times.items())]
            self._seg_t_stops = [v / float(resolution) for k, v in sorted(max_nev_times.items())]
            self._nb_segment = len(self._seg_t_starts)
            self._sigs_t_starts = [None] * self._nb_segment

        # If no ptp files were seen, enable the buffer description API.
        if not ptp_files_seen:
            self._has_buffer_description_api = True
        # finalize header
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)
        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)
        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)
        signal_buffers = np.array(signal_buffers, dtype=_signal_buffer_dtype)

        self.header = {}
        self.header["nb_block"] = 1
        self.header["nb_segment"] = [self._nb_segment]
        self.header["signal_buffers"] = signal_buffers
        self.header["signal_streams"] = signal_streams
        self.header["signal_channels"] = signal_channels
        self.header["spike_channels"] = spike_channels
        self.header["event_channels"] = event_channels

        rec_datetime = self._nev_params("rec_datetime") if self._avail_files["nev"] else None

        # Put annotations at some places for compatibility
        # with previous BlackrockIO version
        self._generate_minimal_annotations()

        block_ann = self.raw_annotations["blocks"][0]
        block_ann["description"] = "Block of data from Blackrock file set."
        block_ann["file_origin"] = self.filename
        block_ann["name"] = "Blackrock Data Block"
        block_ann["rec_datetime"] = rec_datetime
        block_ann["avail_file_set"] = [k for k, v in self._avail_files.items() if v]
        block_ann["avail_nsx"] = self._avail_nsx
        block_ann["avail_nev"] = self._avail_files["nev"]
        #  'sif' and 'ccf' files not yet supported
        # block_ann['avail_sif'] = self._avail_files['sif']
        # block_ann['avail_ccf'] = self._avail_files['ccf']
        block_ann["rec_pauses"] = False

        for seg_index in range(self._nb_segment):
            seg_ann = block_ann["segments"][seg_index]
            seg_ann["file_origin"] = self.filename
            seg_ann["name"] = f"Segment {seg_index}"
            seg_ann["description"] = "Segment containing data from t_start to t_stop"
            if seg_index == 0:
                # if more than 1 segment means pause
                # so datetime is valid only for seg_index=0
                seg_ann["rec_datetime"] = rec_datetime

            for c in range(signal_streams.size):
                sig_ann = seg_ann["signals"][c]
                stream_id = signal_streams["id"][c]
                nsx_nb = int(stream_id)
                sig_ann["description"] = f"AnalogSignal from  nsx{nsx_nb}"
                sig_ann["file_origin"] = self._filenames["nsx"] + ".ns" + str(nsx_nb)
                sig_ann["nsx"] = nsx_nb
                # handle signal array annotations from nsx header
                if self._nsx_spec[nsx_nb] in ["2.2", "2.3"] and nsx_nb in self._nsx_ext_header:
                    mask = signal_channels["stream_id"] == stream_id
                    channels = signal_channels[mask]
                    nsx_header = self._nsx_ext_header[nsx_nb]
                    for key in (
                        "physical_connector",
                        "connector_pin",
                        "hi_freq_corner",
                        "lo_freq_corner",
                        "hi_freq_order",
                        "lo_freq_order",
                        "hi_freq_type",
                        "lo_freq_type",
                    ):
                        values = []
                        for chan_id in channels["id"]:
                            chan_id = int(chan_id)
                            idx = list(nsx_header["electrode_id"]).index(chan_id)
                            values.append(nsx_header[key][idx])
                        values = np.array(values)
                        sig_ann["__array_annotations__"][key] = values

            for c in range(spike_channels.size):
                st_ann = seg_ann["spikes"][c]
                channel_id, unit_id = self.internal_unit_ids[c]
                st_ann["channel_id"] = channel_id
                st_ann["unit_id"] = unit_id
                if unit_id == 0:
                    st_ann["unit_classification"] = "unclassified"
                elif 1 <= unit_id <= 16:
                    st_ann["unit_classification"] = "sorted"
                elif unit_id == 255:
                    st_ann["unit_classification"] = "noise"
                else:  # 17-254 are reserved
                    st_ann["unit_classification"] = "reserved"
                st_ann["unit_tag"] = st_ann["unit_classification"]
                st_ann["description"] = f"SpikeTrain channel_id: {channel_id}, unit_id: {unit_id}"
                st_ann["file_origin"] = self._filenames["nev"] + ".nev"

            if self._avail_files["nev"]:
                ev_dict = self._nonneural_evdicts[self._nev_spec](events_data)
                if "Comments" in self.nev_data:
                    ev_dict.update(self._comment_evdict[self._nev_spec](comments_data))
                    color_codes = ["#{:08X}".format(code) for code in comments_data["color"]]
                    color_codes = np.array(color_codes, dtype="S9")
                for c in range(event_channels.size):
                    # Next line makes ev_ann a reference to seg_ann['events'][c]
                    ev_ann = seg_ann["events"][c]
                    name = event_channels["name"][c]
                    ev_ann["description"] = ev_dict[name]["desc"]
                    ev_ann["file_origin"] = self._filenames["nev"] + ".nev"
                    if name == "comments":
                        ev_ann["color_codes"] = color_codes

    def _source_name(self):
        return self.filename

    def _segment_t_start(self, block_index, seg_index):
        return self._seg_t_starts[seg_index]

    def _segment_t_stop(self, block_index, seg_index):
        return self._seg_t_stops[seg_index]

    def _get_signal_size(self, block_index, seg_index, stream_index):
        stream_id = self.header["signal_streams"][stream_index]["id"]
        nsx_nb = int(stream_id)
        return self._nsx_data_header[nsx_nb][seg_index]["nb_data_points"]

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        stream_id = self.header["signal_streams"][stream_index]["id"]
        nsx_nb = int(stream_id)
        return self._sigs_t_starts[nsx_nb][seg_index]

    @staticmethod
    def _create_mmap_view(fid, dtype, offset, num_samples, num_channels=None, packet_size=None, item_offset=0):
        """
        Create an np.ndarray view over a raw mmap buffer from an open file.

        When packet_size is None, creates a standard contiguous view (for
        standard/v2.1 formats where samples are stored contiguously as int16).

        When packet_size is provided, creates a strided view that extracts
        interleaved fields from PTP packets. Each packet has a fixed size,
        and the target field starts at item_offset bytes into the packet.
        The stride between rows equals packet_size, allowing the view to
        skip over other fields (timestamps, reserved bytes, etc.).

        Parameters
        ----------
        fid : file-like
            Open file object (must support .fileno()).
        dtype : str or np.dtype
            Data type of the target field (e.g. "int16" for samples,
            "uint64" for timestamps).
        offset : int
            Byte offset in the file where the data region starts.
        num_samples : int
            Number of samples (rows) to read.
        num_channels : int or None
            Number of channels (columns). None for 1D arrays (e.g. timestamps).
        packet_size : int or None
            Stride between consecutive rows in bytes. None for contiguous data.
        item_offset : int
            Byte offset of the target field within each packet.

        Returns
        -------
        np.ndarray
            View into the memory-mapped file. Shape is (num_samples, num_channels)
            when num_channels is provided, or (num_samples,) otherwise.
        """
        dtype = np.dtype(dtype)

        if num_channels is not None:
            shape = (num_samples, num_channels)
        else:
            shape = (num_samples,)

        if packet_size is None:
            bytes_per_sample = dtype.itemsize * (num_channels if num_channels is not None else 1)
            start_byte = offset
            length = num_samples * bytes_per_sample
        else:
            start_byte = offset + item_offset
            length = (num_samples - 1) * packet_size + (
                dtype.itemsize * num_channels if num_channels is not None else dtype.itemsize
            )

        # mmap offset must be aligned to ALLOCATIONGRANULARITY
        mmap_offset, start_remainder = divmod(start_byte, mmap.ALLOCATIONGRANULARITY)
        mmap_offset *= mmap.ALLOCATIONGRANULARITY
        length += start_remainder

        raw_mmap = mmap.mmap(fid.fileno(), length=length, access=mmap.ACCESS_READ, offset=mmap_offset)

        if packet_size is not None:
            strides = (packet_size, dtype.itemsize) if num_channels is not None else (packet_size,)
        else:
            strides = None  # default contiguous strides

        return np.ndarray(
            shape=shape,
            dtype=dtype,
            buffer=raw_mmap,
            offset=start_remainder,
            strides=strides,
        )

    def _get_nsx_fid(self, nsx_nb):
        """Open NSX file on demand and cache the file descriptor for reuse."""
        if not hasattr(self, "_nsx_fids"):
            self._nsx_fids = {}
        if nsx_nb not in self._nsx_fids:
            filename = f"{self._filenames['nsx']}.ns{nsx_nb}"
            self._nsx_fids[nsx_nb] = self._open(filename)
        return self._nsx_fids[nsx_nb]

    def _get_nev_fid(self):
        """Open NEV file on demand and cache the file descriptor for reuse."""
        if not hasattr(self, "_nev_fid"):
            filename = f"{self._filenames['nev']}.nev"
            self._nev_fid = self._open(filename)
        return self._nev_fid

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, stream_index, channel_indexes):
        stream_id = self.header["signal_streams"][stream_index]["id"]
        nsx_nb = int(stream_id)
        seg = self._nsx_data_header[nsx_nb][seg_index]
        fid = self._get_nsx_fid(nsx_nb)
        memmap_kwargs = seg["memmap_kwargs"]
        channels = int(self._nsx_basic_header[nsx_nb]["channel_count"])
        num_channels = memmap_kwargs.get("num_channels", channels)
        if self._fs is None:
            rows = self._create_mmap_view(
                fid=fid,
                dtype=memmap_kwargs["dtype"],
                offset=memmap_kwargs["offset"],
                num_samples=memmap_kwargs["num_samples"],
                num_channels=num_channels,
                packet_size=memmap_kwargs.get("packet_size"),
                item_offset=memmap_kwargs.get("item_offset", 0),
            )
            data = self._in_timestamp_order(rows[i_start:i_stop], memmap_kwargs, i_start, lambda r: rows[r])
        else:
            data = self._read_rows(
                fid=fid, memmap_kwargs=memmap_kwargs, num_channels=num_channels, i_start=i_start, i_stop=i_stop
            )
            data = self._in_timestamp_order(
                data,
                memmap_kwargs,
                i_start,
                lambda r: np.concatenate([self._read_rows(fid, memmap_kwargs, num_channels, i, i + 1) for i in r]),
            )
        if channel_indexes is None:
            channel_indexes = slice(None)
        return data[:, channel_indexes]

    @staticmethod
    def _in_timestamp_order(rows, kw, i_start, read):
        """
        Rows [i_start, i_start + len(rows)) of a PTP segment in timestamp order: `kw["reorder"]`, if
        present, holds the positions the file has out of order and the rows to read there instead
        (see _find_reordered_samples). `read(rows)` reads the given rows of the segment.
        """
        if "reorder" not in kw:
            return rows
        positions, sources = kw["reorder"]
        i_start = i_start or 0
        here = (positions >= i_start) & (positions < i_start + len(rows))
        if not here.any():
            return rows
        rows = np.array(rows)  # a copy: memory maps are read only
        rows[positions[here] - i_start] = read(sources[here])
        return rows

    def _read_rows(
        self,
        fid: io.IOBase,
        memmap_kwargs: dict,
        num_channels: int,
        i_start: int | None,
        i_stop: int | None,
    ) -> np.ndarray:
        """
        Read a chunk of analog signal data from the file. Standard files
        have no `packet_size`, so a row is `num_channels x itemsize` bytes.

        Parameters
        ----------
        fid : io.IOBase
            File descriptor of the NSX file.
        memmap_kwargs : dict
            Dictionary containing memory-mapping information.
        num_channels : int
            Number of channels in the data.
        i_start : int | None, optional
            Starting row index, by default None.
        i_stop : int | None, optional
            Ending row index, by default None.

        Returns
        -------
        np.ndarray
            Array of shape (n_rows, num_channels) containing the requested data.
        """

        dtype = np.dtype(memmap_kwargs["dtype"])
        # Standard files have no packet_size: rows sit back to back.
        packet_size = memmap_kwargs.get("packet_size") or num_channels * dtype.itemsize
        item_offset = memmap_kwargs.get("item_offset", 0)
        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = memmap_kwargs["num_samples"]
        return read_strided_from_opened_file(
            fid,
            dtype,
            offset=memmap_kwargs["offset"] + i_start * packet_size + item_offset,
            stride=packet_size,
            count=i_stop - i_start,
            num_values=num_channels,
        )

    def _read_ptp_ticks(self, nsx_nb: int, ts_kw: dict, i_start: int, i_stop: int) -> np.ndarray:
        """
        Raw PTP timestamps (uint64 ticks) for samples [i_start, i_stop) of one segment.

        Local files take them from a memory map; remote files read only that range.
        """
        fid = self._get_nsx_fid(nsx_nb)
        if self._fs is None:
            ticks = self._create_mmap_view(
                fid=fid,
                dtype=ts_kw["dtype"],
                offset=ts_kw["offset"],
                num_samples=ts_kw["num_samples"],
                packet_size=ts_kw.get("packet_size"),
                item_offset=ts_kw.get("item_offset", 0),
            )
            return self._in_timestamp_order(ticks[i_start:i_stop], ts_kw, i_start, lambda r: ticks[r])

        def read(start, stop):
            return read_strided_from_opened_file(
                fid,
                ts_kw["dtype"],
                offset=ts_kw["offset"] + ts_kw.get("item_offset", 0) + start * ts_kw["packet_size"],
                stride=ts_kw["packet_size"],
                count=stop - start,
            )

        return self._in_timestamp_order(
            read(i_start, i_stop), ts_kw, i_start, lambda r: np.concatenate([read(i, i + 1) for i in r])
        )

    def _get_analogsignal_buffer_description(self, block_index: int, seg_index: int, buffer_id: str) -> dict:
        """Get the description of the analog signal buffer.

        This includes information necessary to memory-map the data, such as
        file path, data type, shape, and file offset. We currently support only
        standard (non-PTP) NSX files. PTP streams are never described. Their
        samples use per-sample timestamps, which this format can't express. The
        buffer API is disabled when one is loaded.

        Parameters
        ----------
        block_index : int
            Block index.
        seg_index : int
            Segment index.
        buffer_id : str
            Buffer ID.

        Returns
        -------
        buffer_description : dict
            Dictionary containing the buffer description necessary for memory-mapping the data.
        """
        nsx_nb = int(buffer_id)
        seg = self._nsx_data_header[nsx_nb][seg_index]
        channels = int(self._nsx_basic_header[nsx_nb]["channel_count"])
        memmap_kwargs = seg["memmap_kwargs"]
        samples = memmap_kwargs["num_samples"]
        buffer_description = {
            "type": "raw",
            "file_path": memmap_kwargs["filename"],
            "dtype": memmap_kwargs["dtype"],
            "order": "C",
            "file_offset": memmap_kwargs["offset"],
            "shape": (samples, channels),
        }
        return buffer_description

    def _get_blackrock_timestamps(self, block_index, seg_index, i_start, i_stop, stream_index):
        """
        Return timestamps in seconds for analog signal samples in the given range.

        The behavior depends on the file format:

        - **PTP format (FileSpec 3.0-ptp):** Each packet in the file contains
          exactly one sample with its own PTP hardware timestamp at nanosecond
          resolution. This method returns those actual timestamps converted to
          seconds. Because each NSx file (e.g. ns2 at 1kHz, ns6 at 30kHz) stores
          its own independent PTP packets, every sample has a real timestamp.
          Note that PTP timestamps exhibit natural clock jitter at the
          nanosecond scale, so consecutive sample intervals are not perfectly
          uniform.

        - **Standard formats (FileSpec 2.2, 2.3, 3.0 non-PTP):** Each data
          block has a single scalar timestamp for its first sample. All
          subsequent samples within the block are interpolated as
          ``t_start + sample_index / sampling_rate``, assuming uniform spacing.

        - **FileSpec 2.1:** No timestamps are stored in the file. All samples
          are interpolated from ``t_start=0`` using the sampling rate.

        Parameters
        ----------
        block_index : int
            Block index.
        seg_index : int
            Segment index.
        i_start : int | None
            First sample index. None means 0.
        i_stop : int | None
            Stop sample index (exclusive). None means end of segment.
        stream_index : int
            Stream index.

        Returns
        -------
        timestamps : np.ndarray (float64)
            Timestamps in seconds for each sample in [i_start, i_stop).
        """
        stream_id = self.header["signal_streams"][stream_index]["id"]
        nsx_nb = int(stream_id)
        seg = self._nsx_data_header[nsx_nb][seg_index]

        # Resolve None to concrete indices
        size = seg["nb_data_points"]
        i_start = i_start if i_start is not None else 0
        i_stop = i_stop if i_stop is not None else size

        # PTP format: read timestamps on demand via strided mmap view
        if "timestamps_memmap_kwargs" in seg:
            ts_kw = seg["timestamps_memmap_kwargs"]
            ts_res = float(self._nsx_basic_header[nsx_nb]["timestamp_resolution"])
            return self._read_ptp_ticks(nsx_nb, ts_kw, i_start, i_stop).astype("float64") / ts_res
        else:
            # Non-PTP: reconstruct from t_start + index / sampling_rate
            t_start = self._sigs_t_starts[nsx_nb][seg_index]
            sr = self._nsx_sampling_frequency[nsx_nb]
            return t_start + np.arange(i_start, i_stop, dtype="float64") / sr

    def _spike_count(self, block_index, seg_index, unit_index):
        channel_id, unit_id = self.internal_unit_ids[unit_index]
        if self._nev_spikes is not None:
            # Counted while the NEV was scanned at open, so no further pass is needed.
            return sum(
                n
                for (raw_seg, ch, unit), n in self._nev_spikes["counts"].items()
                if ch == channel_id and unit == unit_id and self._spike_segment_ids([raw_seg])[0] == seg_index
            )

        all_spikes, event_segment_ids = self._spike_records(channel_id, unit_id)
        # Selected by segment id, as _get_spike_timestamps does, so the count always equals
        # the number of timestamps it returns.
        mask = (
            (all_spikes["packet_id"] == channel_id)
            & (all_spikes["unit_class_nb"] == unit_id)
            & (event_segment_ids == seg_index)
        )
        return int(np.sum(mask))

    def _get_spike_timestamps(self, block_index, seg_index, unit_index, t_start, t_stop):
        channel_id, unit_id = self.internal_unit_ids[unit_index]

        all_spikes, event_segment_ids = self._spike_records(channel_id, unit_id)

        # select by channel_id and unit_id
        mask = (
            (all_spikes["packet_id"] == channel_id)
            & (all_spikes["unit_class_nb"] == unit_id)
            & (event_segment_ids == seg_index)
        )
        unit_spikes = self._in_time_order(all_spikes[mask])

        timestamp = unit_spikes["timestamp"]
        sl = self._get_timestamp_slice(timestamp, seg_index, t_start, t_stop)
        timestamp = timestamp[sl]

        return timestamp

    @staticmethod
    def _in_time_order(spikes):
        """Spike records in timestamp order (stable): the NEV can hold spikes out of order (REORDER_WINDOW_S)."""
        return spikes[np.argsort(spikes["timestamp"], kind="stable")]

    def _get_timestamp_slice(self, timestamp, seg_index, t_start, t_stop):
        """
        Slice of `timestamp` (one segment's, in timestamp order) within [t_start, t_stop), in seconds.

        The segment itself is already selected by segment id, so None means no limit.

        The bounds are compared with the times neo reports (the timestamps in float64 seconds), so a
        time obtained from neo, such as `segment_t_start`, selects the item it came from. Converting the
        bounds to ticks instead does not work at the 1 ns resolution of PTP files: a float64 number of
        seconds (about 1.75e9) is only precise to about 240 ns, so the converted bound can fall a few
        ticks past the item and drop it, and np.searchsorted compares uint64 with a Python int as float64.

        `t_stop` is exclusive. To include an item at `t_stop`, such as the last of a segment,
        pass `t_stop + 1e-5` (10 us, shorter than the interval between samples at 30 kHz and longer
        than the precision of float64 seconds).
        """
        if t_start is None and t_stop is None:
            return slice(None)
        times = timestamp.astype("float64") / self._nev_basic_header["timestamp_resolution"]
        ind_start = None if t_start is None else np.searchsorted(times, t_start, side="left")
        ind_stop = None if t_stop is None else np.searchsorted(times, t_stop, side="left")
        return slice(ind_start, ind_stop)

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        spike_times = spike_timestamps.astype(dtype)
        spike_times /= self._nev_basic_header["timestamp_resolution"]
        return spike_times

    def _get_spike_raw_waveforms(self, block_index, seg_index, unit_index, t_start, t_stop):
        channel_id, unit_id = self.internal_unit_ids[unit_index]
        all_spikes, event_segment_ids = self._spike_records(channel_id, unit_id)

        mask = (
            (all_spikes["packet_id"] == channel_id)
            & (all_spikes["unit_class_nb"] == unit_id)
            & (event_segment_ids == seg_index)
        )
        unit_spikes = self._in_time_order(all_spikes[mask])
        waveforms = self._spike_waveforms(unit_spikes, channel_id)

        timestamp = unit_spikes["timestamp"]
        sl = self._get_timestamp_slice(timestamp, seg_index, t_start, t_stop)
        waveforms = waveforms[sl]

        return waveforms

    def _event_count(self, block_index, seg_index, event_channel_index):
        name = self.header["event_channels"]["name"][event_channel_index]
        if name == "comments":
            events_data, event_segment_ids = self.nev_data["Comments"]
            ev_dict = self._comment_evdict[self._nev_spec](events_data)[name]
        else:
            events_data, event_segment_ids = self.nev_data["NonNeural"]
            ev_dict = self._nonneural_evdicts[self._nev_spec](events_data)[name]
        # Selected by segment id, as _get_event_timestamps does.
        mask = ev_dict["mask"] & (event_segment_ids == seg_index)
        return int(np.sum(mask))

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        name = self.header["event_channels"]["name"][event_channel_index]
        if name == "comments":
            events_data, event_segment_ids = self.nev_data["Comments"]
            ev_dict = self._comment_evdict[self._nev_spec](events_data)[name]
            # If immediate decoding is desired:
            encoding = {0: "latin_1", 1: "utf_16", 255: "latin_1"}
            labels = [data[ev_dict["field"]].decode(encoding[data["char_set"]]) for data in events_data]
            labels = np.array(labels, dtype="U")
        else:
            events_data, event_segment_ids = self.nev_data["NonNeural"]
            ev_dict = self._nonneural_evdicts[self._nev_spec](events_data)[name]
            labels = events_data[ev_dict["field"]].astype("U")

        mask = ev_dict["mask"] & (event_segment_ids == seg_index)
        timestamp = events_data[mask]["timestamp"]
        labels = labels[mask]

        # time clip
        sl = self._get_timestamp_slice(timestamp, seg_index, t_start, t_stop)
        timestamp = timestamp[sl]
        labels = labels[sl]
        durations = None

        return timestamp, durations, labels

    def _rescale_event_timestamp(self, event_timestamps, dtype, event_channel_index):
        ev_times = event_timestamps.astype(dtype)
        ev_times /= self._nev_basic_header["timestamp_resolution"]
        return ev_times

    ###################################################
    ###################################################

    # Above here code from Lyuba Zehl, Michael Denker
    # coming from previous BlackrockIO

    def _extract_nsx_file_spec(self, nsx_nb):
        """
        Extract file specification from an .nsx file.
        """
        # Header structure of files specification 2.2 and higher. For files 2.1
        # and lower, the entries ver_major and ver_minor are not supported.
        dt0 = [("file_id", "S8"), ("ver_major", "uint8"), ("ver_minor", "uint8")]

        fid = self._get_nsx_fid(nsx_nb)
        nsx_file_id = read_records_from_opened_file(fid, dt0, offset=0, count=1)[0]
        if nsx_file_id["file_id"].decode() == "NEURALSG":
            spec = "2.1"
        elif nsx_file_id["file_id"].decode() in ["NEURALCD", "BRSMPGRP"]:
            spec = f"{nsx_file_id['ver_major']}.{nsx_file_id['ver_minor']}"
        else:
            raise IOError("Unsupported NSX file type.")

        return spec

    def _extract_nev_file_spec(self):
        """
        Extract file specification from an .nev file
        """
        # Header structure of files specification 2.2 and higher. For files 2.1
        # and lower, the entries ver_major and ver_minor are not supported.
        dt0 = [("file_id", "S8"), ("ver_major", "uint8"), ("ver_minor", "uint8")]

        fid = self._get_nev_fid()
        nev_file_id = read_records_from_opened_file(fid, dt0, offset=0, count=1)[0]
        if nev_file_id["file_id"].decode() in ["NEURALEV", "BREVENTS"]:
            spec = f"{nev_file_id['ver_major']}.{nev_file_id['ver_minor']}"
        else:
            raise IOError(f"NEV file type {nev_file_id['file_id'].decode()} is not supported")

        return spec

    def _read_nsx_header(self, spec, nsx_nb):
        """
        Extract nsx header information for any specification version.

        Parameters
        ----------
        spec : str
            The specification version (e.g., "2.1", "2.2", "2.3", "3.0")
        nsx_nb : int
            The NSX file number (e.g., 5 for ns5)

        Returns
        -------
        nsx_basic_header : numpy structured array
            Basic header information
        nsx_ext_header : numpy memmap
            Extended header information
        """

        # Get basic header structure for this spec
        basic_header_dtype = NSX_BASIC_HEADER_TYPES[spec]

        fid = self._get_nsx_fid(nsx_nb)
        nsx_basic_header = read_records_from_opened_file(fid, dtype=basic_header_dtype, count=1, offset=0)[0]

        # Get extended header structure for this spec
        ext_header_dtype = NSX_EXT_HEADER_TYPES[spec]
        offset_dt0 = np.dtype(basic_header_dtype).itemsize
        channel_count = int(nsx_basic_header["channel_count"])
        nsx_ext_header = read_records_from_opened_file(
            fid, dtype=ext_header_dtype, count=channel_count, offset=offset_dt0
        )

        return nsx_basic_header, nsx_ext_header

    def _read_nsx_dataheader(self, spec, nsx_nb, offset):
        """
        Reads data header following the given offset of an nsx file.

        Parameters
        ----------
        spec : str
            The specification version (e.g., "2.2", "2.3", "3.0")
        nsx_nb : int
            The NSX file number
        offset : int
            Offset position in the file
        """

        # Get data header structure for this spec
        data_header_dtype = NSX_DATA_HEADER_TYPES[spec]
        if data_header_dtype is None:
            return None  # v2.1 has no data headers

        fid = self._get_nsx_fid(nsx_nb)
        nsx_basic_header = read_records_from_opened_file(fid, dtype=data_header_dtype, count=1, offset=offset)[0]

        return nsx_basic_header

    def _parse_nsx_data(self, spec, nsx_nb):
        """
        Parse NSX data blocks from file and extract data with timestamps.

        This is the main router function for NSX data parsing. It creates a memory-mapped
        view of the file internally and extracts data blocks with their associated timestamps.

        The function handles three different NSX file format variants, each with different
        internal structure for storing data and timestamps.

        NSX FILE FORMAT VARIANTS
        ========================

        STANDARD FORMAT (v2.2, v2.3, v3.0 non-PTP)
        ------------------------------------------
        File structure:
        ┌─────────────────────────────────────────────┐
        │ BASIC HEADER (fixed size)                   │  ← File metadata
        │ - file_id, version, period, etc.            │
        ├─────────────────────────────────────────────┤
        │ EXTENDED HEADER (per channel)               │  ← Channel info
        │ - electrode_id, label, units, etc.          │
        ├─────────────────────────────────────────────┤
        │ DATA BLOCK 1 HEADER                         │  ← Block metadata
        │ - header_flag=1                             │
        │ - timestamp (scalar, e.g., 0)               │
        │ - nb_data_points (e.g., 1000)               │
        ├─────────────────────────────────────────────┤
        │ DATA BLOCK 1 DATA                           │  ← Actual samples
        │ - 1000 samples x N channels                 │
        │ - int16 values                              │
        ├─────────────────────────────────────────────┤
        │ DATA BLOCK 2 HEADER                         │  ← Next block
        │ - header_flag=1                             │
        │ - timestamp (scalar, e.g., 30000)           │
        │ - nb_data_points (e.g., 1000)               │
        ├─────────────────────────────────────────────┤
        │ DATA BLOCK 2 DATA                           │
        │ - 1000 samples x N channels                 │
        └─────────────────────────────────────────────┘

        Key characteristics:
        - Headers are EXPLICIT and SPARSE (only at block boundaries)
        - Each block has ONE scalar timestamp for ALL samples in that block
        - Reader LOOPS through file, finding headers

        PTP FORMAT (v3.0 with Precision Time Protocol)
        -----------------------------------------------
        File structure:
        ┌─────────────────────────────────────────────┐
        │ BASIC HEADER (fixed size)                   │
        │ - timestamp_resolution = 1,000,000,000      │  ← Nanosecond precision!
        ├─────────────────────────────────────────────┤
        │ EXTENDED HEADER (per channel)               │
        ├─────────────────────────────────────────────┤
        │ PACKET 0:                                   │  ← Each sample = packet
        │ - reserved (1 byte)                         │
        │ - timestamp (8 bytes, e.g., 1000)           │
        │ - num_data_points (always 1)                │
        │ - samples (N channels x int16)              │
        ├─────────────────────────────────────────────┤
        │ PACKET 1:                                   │
        │ - reserved                                  │
        │ - timestamp (e.g., 1033)                    │
        │ - num_data_points (1)                       │
        │ - samples (N channels x int16)              │
        ├─────────────────────────────────────────────┤
        │ PACKET 2:                                   │
        │ - reserved                                  │
        │ - timestamp (e.g., 1066)                    │
        │ - num_data_points (1)                       │
        │ - samples (N channels x int16)              │
        ├─────────────────────────────────────────────┤
        │ ...thousands more packets...                │
        │ PACKET 500:                                 │
        │ - timestamp (e.g., 50000)                   │  ← BIG GAP!
        │ ...                                         │
        └─────────────────────────────────────────────┘

        Key characteristics:
        - NO separate headers and data - they're INTERLEAVED
        - EVERY sample has its own timestamp (per-sample nanosecond precision)
        - File is ONE CONTINUOUS ARRAY of uniform packets

        V2.1 FORMAT
        -----------
        Simplest format:
        - Single continuous data block
        - No timestamps in data section
        - No multiple blocks (no pause/resume support)

        Parameters
        ----------
        spec : str
            File specification version ("2.1", "2.2", "2.3", "3.0", "3.0-ptp")
        nsx_nb : int
            NSX file number (e.g., 5 for ns5 file)

        Returns
        -------
        list[dict]
            List of parsed data block headers, one per block in the file:
            [
                {
                    "timestamp": scalar or None,
                        - Standard format: scalar (one timestamp per block)
                        - v2.1 format: None (no timestamps)
                    "memmap_kwargs": dict,
                        Recipe for on-demand mmap creation (offset, num_samples, dtype, etc.)
                    "ptp_timestamps_memmap_kwargs": dict (PTP only),
                        Strided access recipe for per-sample PTP timestamps
                },
                ...
            ]

        Notes
        -----
        - No memmaps are stored; only lightweight metadata for on-demand creation
        - Returned data is NOT YET SEGMENTED (segmentation happens in a separate step)
        - For standard format, each block from the file is one list entry
        - For PTP format, all data is in a single entry
        """
        if spec == "2.1":
            return self._parse_nsx_data_v21(nsx_nb)
        elif spec == "3.0-ptp":
            return self._parse_nsx_data_v30_ptp(nsx_nb)
        else:  # 2.2, 2.3, 3.0 standard
            return self._parse_nsx_data_v22_v30(spec, nsx_nb)

    def _parse_nsx_data_v21(self, nsx_nb):
        """
        Parse v2.1 NSX data blocks.

        V2.1 format is the simplest:
        - Single continuous data block
        - No timestamps
        - No pause/resume support

        Returns
        -------
        list[dict]
            Single-element list: [{"timestamp": None, "memmap_kwargs": dict}]
        """
        filename = f"{self._filenames['nsx']}.ns{nsx_nb}"

        # Calculate header size and data points for v2.1
        channels = int(self._nsx_basic_header[nsx_nb]["channel_count"])
        bytes_in_headers = (
            self._nsx_basic_header[nsx_nb].dtype.itemsize + self._nsx_ext_header[nsx_nb].dtype.itemsize * channels
        )
        filesize = self._get_file_size(filename)
        num_samples = int((filesize - bytes_in_headers) / (2 * channels) - 1)

        return [
            {
                "timestamp": None,
                "memmap_kwargs": {
                    "filename": filename,
                    "dtype": "int16",
                    "offset": bytes_in_headers,
                    "num_samples": num_samples,
                },
            }
        ]

    def _parse_nsx_data_v22_v30(self, spec, nsx_nb):
        """
        Parse standard format NSX data blocks (v2.2, 2.3, 3.0).

        Standard format has:
        - Explicit block headers in file
        - Each block has scalar timestamp
        - Multiple blocks when recording paused/resumed

        Returns
        -------
        list[dict]
            [{"timestamp": scalar, "memmap_kwargs": dict}, ...]
        """
        filename = f"{self._filenames['nsx']}.ns{nsx_nb}"

        # Get file parameters
        filesize = self._get_file_size(filename)
        channels = int(self._nsx_basic_header[nsx_nb]["channel_count"])
        current_offset = int(self._nsx_basic_header[nsx_nb]["bytes_in_headers"])

        parsed_data_headers = []

        # Loop through file, reading block headers
        while current_offset < filesize:
            # Read header at current position
            header = self._read_nsx_dataheader(spec, nsx_nb, current_offset)

            if header["header_flag"] != 1:
                raise ValueError(
                    f"Invalid NSX data block header at offset {current_offset:#x} "
                    f"in ns{nsx_nb} file. Expected header_flag=1, got {header['header_flag']}."
                )

            num_samples = int(header["nb_data_points"])
            data_offset = current_offset + header.dtype.itemsize
            timestamp = header["timestamp"]

            parsed_data_headers.append(
                {
                    "timestamp": timestamp,
                    "memmap_kwargs": {
                        "filename": filename,
                        "dtype": "int16",
                        "offset": data_offset,
                        "num_samples": num_samples,
                    },
                }
            )

            # Jump to next block
            data_size_bytes = num_samples * channels * 2  # int16 = 2 bytes
            current_offset = data_offset + data_size_bytes

        return parsed_data_headers

    def _parse_nsx_data_v30_ptp(self, nsx_nb):
        """
        Parse PTP format NSX data (v3.0 with Precision Time Protocol).

        PTP format has:
        - Interleaved structure (timestamp + sample per packet)
        - Array of timestamps (one per sample)
        - Continuous data (segmentation inferred from timestamp gaps)

        Returns
        -------
        list[dict]
            Single-element list: [{"timestamp": None, "memmap_kwargs": dict,
                                    "ptp_timestamps_memmap_kwargs": dict}]
        """
        filename = f"{self._filenames['nsx']}.ns{nsx_nb}"

        # Get file parameters
        filesize = self._get_file_size(filename)
        header_size = int(self._nsx_basic_header[nsx_nb]["bytes_in_headers"])
        channel_count = int(self._nsx_basic_header[nsx_nb]["channel_count"])

        ptp_dt = NSX_DATA_HEADER_TYPES["3.0-ptp"](channel_count)
        ptp_dtype = np.dtype(ptp_dt)
        npackets = int((filesize - header_size) / ptp_dtype.itemsize)

        # Verify this is truly PTP: the first packet should carry exactly one sample.
        # PTP files are homogeneous (every packet holds a single sample), so the first
        # packet is diagnostic for the whole file and we avoid mapping all of it.
        fid = self._get_nsx_fid(nsx_nb)
        fid.seek(header_size)
        first_packet = np.frombuffer(fid.read(ptp_dtype.itemsize), dtype=ptp_dtype, count=1)[0]
        if first_packet["num_data_points"] != 1:
            # Not actually PTP! Fall back to standard format
            return self._parse_nsx_data_v22_v30("3.0", nsx_nb)

        # Compute strided access parameters from the structured dtype
        packet_size = ptp_dtype.itemsize
        samples_item_offset = ptp_dtype.fields["samples"][1]
        timestamps_item_offset = ptp_dtype.fields["timestamps"][1]

        return [
            {
                "timestamp": None,
                "memmap_kwargs": {
                    "filename": filename,
                    "dtype": "int16",
                    "offset": header_size,
                    "num_samples": npackets,
                    "num_channels": channel_count,
                    "packet_size": packet_size,
                    "item_offset": samples_item_offset,
                },
                "ptp_timestamps_memmap_kwargs": {
                    "filename": filename,
                    "dtype": "uint64",
                    "offset": header_size,
                    "num_samples": npackets,
                    "packet_size": packet_size,
                    "item_offset": timestamps_item_offset,
                },
            }
        ]

    def _format_gap_report(self, gap_indices, gap_sizes_s, gap_before_s, first_s, nsx_nb):
        """
        Format a detailed gap report showing where timestamp discontinuities occur.

        Parameters
        ----------
        gap_indices : np.ndarray
            Indices where gaps were detected
        gap_sizes_s : np.ndarray
            Size of each gap in seconds
        gap_before_s : np.ndarray
            Timestamp, in seconds, of the sample just before each gap
        first_s : float
            First timestamp of the file in seconds
        nsx_nb : int
            NSX file number for the report

        Returns
        -------
        str
            Formatted gap report with table
        """
        # Calculate gap details
        gap_durations_ms = gap_sizes_s * 1000
        gap_positions_seconds = gap_before_s - first_s

        # Build gap detail table
        gap_detail_lines = [
            f"| {index:>15,} | {pos:>21.6f} | {dur:>21.3f} |\n"
            for index, pos, dur in zip(gap_indices, gap_positions_seconds, gap_durations_ms)
        ]

        return (
            f"Gap Report for ns{nsx_nb}:\n"
            f"Found {len(gap_indices)} timestamp gaps (detection threshold: 2 x sampling period)\n\n"
            "Gap Details:\n"
            "+-----------------+-----------------------+-----------------------+\n"
            "| Sample Index    | Sample at (Seconds)   | Gap Size (ms)         |\n"
            "+-----------------+-----------------------+-----------------------+\n"
            + "".join(gap_detail_lines)
            + "+-----------------+-----------------------+-----------------------+\n"
        )

    def _segment_nsx_data(self, parsed_data_headers, nsx_nb):
        """
        Segment NSX data based on timestamp gaps.

        Takes the parsed data headers returned by _parse_nsx_data() and creates segments.
        Segmentation logic depends on the file format:

        - Standard format (multiple blocks): Each block IS a segment
        - PTP format (single block with timestamp array): Detect gaps in timestamps
        - V2.1 format (no timestamps): Single segment

        Parameters
        ----------
        parsed_data_headers : list[dict]
            List from _parse_nsx_data():
            [{"timestamp": scalar/None, "memmap_kwargs": dict, ...}, ...]
        nsx_nb : int
            NSX file number

        Returns
        -------
        list[dict]
            [
                {
                    "timestamp": scalar or None,
                    "nb_data_points": int,
                    "header": int or None,
                    "offset_to_data_block": None (deprecated but kept for compatibility),
                    "memmap_kwargs": dict,
                    "timestamps_memmap_kwargs": dict (PTP only),
                },
                ...
            ]
        """
        segments = []

        # Case 1: Multiple blocks (Standard format) - each block is a segment
        if len(parsed_data_headers) > 1:
            for block_info in parsed_data_headers:
                segments.append(
                    {
                        "timestamp": block_info["timestamp"],
                        "nb_data_points": block_info["memmap_kwargs"]["num_samples"],
                        "header": 1,  # Standard format has headers
                        "offset_to_data_block": None,  # Not needed
                        "memmap_kwargs": block_info["memmap_kwargs"],
                    }
                )

        # Case 2: Single block - check if PTP (has ptp_timestamps_memmap_kwargs) or simple
        elif len(parsed_data_headers) == 1:
            block_info = parsed_data_headers[0]

            # PTP format: read timestamps on demand and detect gaps
            if "ptp_timestamps_memmap_kwargs" in block_info:
                ts_kw = block_info["ptp_timestamps_memmap_kwargs"]
                samples_kw = block_info["memmap_kwargs"]

                sampling_rate = self._nsx_sampling_frequency[nsx_nb]
                # Detection threshold: use strict 2x sampling period to find ALL gaps
                detection_threshold = 2.0 / sampling_rate
                timestamps_sampling_rate = self._nsx_basic_header[nsx_nb]["timestamp_resolution"]

                if self._fs is None:
                    # Read timestamps via strided mmap view for gap detection
                    fid = self._get_nsx_fid(nsx_nb)
                    timestamps = self._create_mmap_view(
                        fid=fid,
                        dtype=ts_kw["dtype"],
                        offset=ts_kw["offset"],
                        num_samples=ts_kw["num_samples"],
                        packet_size=ts_kw.get("packet_size"),
                        item_offset=ts_kw.get("item_offset", 0),
                    )
                    timestamps_in_seconds = timestamps / timestamps_sampling_rate
                    time_differences = np.diff(timestamps_in_seconds)
                    gap_indices = np.argwhere(time_differences > detection_threshold).flatten()
                    gap_sizes_s = time_differences[gap_indices]
                    gap_before_s = timestamps_in_seconds[gap_indices]
                    first_s = timestamps_in_seconds[0] if len(timestamps_in_seconds) else None
                    steps_back = np.flatnonzero(time_differences < 0)
                else:
                    # A remote file cannot be memory-mapped, and its timestamps are spread
                    # through the whole file: scan them in pieces, keeping only the gaps.
                    gaps = find_timestamp_gaps_in_opened_file(
                        self._get_nsx_fid(nsx_nb),
                        offset=ts_kw["offset"] + ts_kw.get("item_offset", 0),
                        packet_size=ts_kw["packet_size"],
                        num_packets=int(ts_kw["num_samples"]),
                        timestamp_resolution=float(timestamps_sampling_rate),
                        threshold_s=detection_threshold,
                    )
                    gap_indices = gaps["indices"]
                    gap_sizes_s = gaps["sizes_s"]
                    gap_before_s = gaps["before_s"]
                    first_s = gaps["first_s"]
                    steps_back = gaps["steps_back"]

                # Packets written out of order are read in timestamp order, and the steps they made in the
                # timestamps as written are no gaps.
                positions, sources, spans = self._find_reordered_samples(nsx_nb, ts_kw, steps_back)
                if len(spans):
                    span = np.searchsorted(spans[:, 0] - 1, gap_indices, side="right") - 1
                    sorted_out = (span >= 0) & (gap_indices <= spans[np.maximum(span, 0), 1])
                    gap_indices = gap_indices[~sorted_out]
                    gap_sizes_s = gap_sizes_s[~sorted_out]
                    gap_before_s = gap_before_s[~sorted_out]

                # If gaps found, check user's tolerance
                if len(gap_indices) > 0:
                    gap_report = self._format_gap_report(gap_indices, gap_sizes_s, gap_before_s, first_s, nsx_nb)

                    # Error by default - user must opt-in to segmentation
                    if self.gap_tolerance_ms is None:
                        raise ValueError(
                            f"Detected {len(gap_indices)} timestamp gaps in ns{nsx_nb} file.\n"
                            f"{gap_report}\n"
                            f"To load this data, provide gap_tolerance_ms parameter to automatically "
                            f"segment at gaps larger than the specified tolerance."
                        )

                    # User provided tolerance - filter gaps and segment
                    gap_tolerance_s = self.gap_tolerance_ms / 1000.0
                    significant_gap_mask = gap_sizes_s > gap_tolerance_s
                    significant_gap_indices = gap_indices[significant_gap_mask]

                    # Use significant gaps for segmentation (no warning - user opted in)
                    gap_indices = significant_gap_indices

                # Create segments based on gaps
                num_total_samples = ts_kw["num_samples"]
                segment_starts = np.hstack((0, gap_indices + 1))
                segment_boundaries = list(segment_starts) + [num_total_samples]

                packet_size = samples_kw["packet_size"]
                base_samples_offset = samples_kw["offset"]
                base_ts_offset = ts_kw["offset"]

                for seg_index, start in enumerate(segment_starts):
                    end = segment_boundaries[seg_index + 1]
                    seg_num_samples = end - start

                    # Compute new file offset for this segment slice
                    seg_samples_offset = base_samples_offset + int(start) * packet_size
                    seg_ts_offset = base_ts_offset + int(start) * packet_size

                    reorder = {}
                    here = (positions >= start) & (positions < end)
                    if here.any():
                        reorder = {"reorder": (positions[here] - start, sources[here] - start)}
                    segments.append(
                        {
                            "timestamp": None,  # PTP timestamps read on demand
                            "nb_data_points": seg_num_samples,
                            "header": None,  # PTP has no headers
                            "offset_to_data_block": None,
                            "memmap_kwargs": {
                                **samples_kw,
                                "offset": seg_samples_offset,
                                "num_samples": seg_num_samples,
                                **reorder,
                            },
                            "timestamps_memmap_kwargs": {
                                **ts_kw,
                                "offset": seg_ts_offset,
                                "num_samples": seg_num_samples,
                                **reorder,
                            },
                        }
                    )

            # V2.1 or single block standard format: no segmentation needed
            else:
                segments.append(
                    {
                        "timestamp": block_info["timestamp"],
                        "nb_data_points": block_info["memmap_kwargs"]["num_samples"],
                        "header": None,
                        "offset_to_data_block": None,
                        "memmap_kwargs": block_info["memmap_kwargs"],
                    }
                )

        return segments

    def _find_reordered_samples(self, nsx_nb, ts_kw, steps_back):
        """
        Packets of a PTP .nsX written out of their timestamp order (REORDER_WINDOW_S).

        For each step back in the timestamps (`steps_back`: the index of the packet before it), the packets
        around it are sorted by timestamp. If no packet moves by more than REORDER_WINDOW_S and the sorted
        packets lie on the sampling grid (each half a period to two periods after the one before: no
        duplicate, overlap or gap), they are read in that order. Any other step back is left as it is.
        The count of steps back sorted is kept in nsx_reordered_packets, with one warning.

        Returns
        -------
        positions, sources : np.ndarray
            Packet positions (from the start of the file) and the packet to read at each instead.
        spans : np.ndarray
            (first, last) position of each sorted run, shape (n, 2).
        """
        n = int(ts_kw["num_samples"])
        fs = self._nsx_sampling_frequency[nsx_nb]
        resolution = float(self._nsx_basic_header[nsx_nb]["timestamp_resolution"])
        period = resolution / fs
        reach = int(np.ceil(REORDER_WINDOW_S * fs)) + 2
        positions, sources, spans = [], [], []
        for step in np.asarray(steps_back).tolist():
            if spans and step < spans[-1][1]:
                continue  # sorted with an earlier step back
            lo, hi = max(0, step + 1 - reach), min(n, step + 1 + reach)
            local = self._read_ptp_ticks(nsx_nb, ts_kw, lo, hi).astype(np.int64)
            order = np.argsort(local, kind="stable")
            # The run sorted for this step: between the nearest places, on either side, before which
            # sorting moves no packet across. Other steps in the window have their own runs.
            closed = np.flatnonzero(np.maximum.accumulate(order) == np.arange(len(order)))
            i = step - lo
            first = int(closed[closed < i][-1]) + 1 if np.any(closed < i) else 0
            last = int(closed[closed >= i + 1][0])
            if (first == 0 and lo > 0) or (last == len(order) - 1 and hi < n):
                continue  # moved further than the window
            in_order = local[order]
            step_sizes = np.diff(in_order[max(first - 1, 0) : last + 2])
            if np.any(step_sizes < period / 2) or np.any(step_sizes > 2 * period):
                continue
            if np.max(np.abs(in_order[first : last + 1] - local[first : last + 1])) > REORDER_WINDOW_S * resolution:
                continue
            positions.append(lo + np.arange(first, last + 1))
            sources.append(lo + order[first : last + 1])
            spans.append((lo + first, lo + last))
        spans = np.array(spans, dtype=np.int64).reshape(-1, 2)
        steps = np.asarray(steps_back)
        count = int(sum(np.sum((steps >= a) & (steps < b)) for a, b in spans))
        self.nsx_reordered_packets[nsx_nb] = count
        if count:
            warnings.warn(
                f"{count} packets of ns{nsx_nb} were written before packets stamped earlier (within "
                f"{REORDER_WINDOW_S * 1000:g} ms) and are read in timestamp order. See nsx_reordered_packets."
            )
        if not spans.size:
            return np.empty(0, np.int64), np.empty(0, np.int64), spans
        return np.concatenate(positions), np.concatenate(sources), spans

    def _build_nsx_v21_ext_header(self, nsx_nb):
        """
        Build extended header structure for v2.1 NSX files.

        v2.1 NSX files don't have extended headers with analog/digital ranges.
        We estimate these from the digitization factor in the NEV file.
        dig_factor = max_analog_val / max_digital_val
        We set max_digital_val = 1000, so max_analog_val = dig_factor
        dig_factor is in nV, so units are uV.

        Information from Kian Torab, Blackrock Microsystems.
        """
        ext_header = []

        for i, elid in enumerate(self._nsx_ext_header[nsx_nb]["electrode_id"]):
            # Get digitization factor from NEV
            if self._avail_files["nev"]:
                # Workaround for DigitalFactor overflow in buggy Cerebus systems
                # Fix from NPMK toolbox (openNEV, line 464, git rev d0a25eac902704a3a29fa5dfd3aed0744f4733ed)
                dig_factor = self._nev_params("digitization_factor")[elid]
                if dig_factor == 21516:
                    dig_factor = 152592.547
                units = "uV"
            else:
                dig_factor = float("nan")
                units = ""
                if i == 0:  # Only warn once
                    warnings.warn("Cannot rescale to voltage, raw data will be returned.", UserWarning)

            # Generate label
            if elid < 129:
                label = f"chan{elid}"
            else:
                label = f"ainp{(elid - 129 + 1)}"

            ext_header.append(
                {
                    "labels": label,
                    "units": units,
                    "min_analog_val": -float(dig_factor),
                    "max_analog_val": float(dig_factor),
                    "min_digital_val": -1000,
                    "max_digital_val": 1000,
                }
            )

        return ext_header

    def _read_nev_header(self, spec):
        """
        Extract nev header information for any specification version.

        Parameters
        ----------
        spec : str
            The specification version (e.g., "2.1", "2.2", "2.3", "3.0")

        Returns
        -------
        nev_basic_header : np.ndarray
            Basic header information
        nev_ext_header : dict
            Extended header information by packet ID
        """
        # Note: This function only uses the passed parameters, not self attributes
        # This makes it easy to convert to @staticmethod later

        # basic header (same for all versions)
        dt0 = [
            # Set to "NEURALEV"
            ("file_type_id", "S8"),
            ("ver_major", "uint8"),
            ("ver_minor", "uint8"),
            # Flags
            ("additionnal_flags", "uint16"),
            # File index of first data sample
            ("bytes_in_headers", "uint32"),
            # Number of bytes per data packet (sample)
            ("bytes_in_data_packets", "uint32"),
            # Time resolution of time stamps in Hz
            ("timestamp_resolution", "uint32"),
            # Sampling frequency of waveforms in Hz
            ("sample_resolution", "uint32"),
            ("year", "uint16"),
            ("month", "uint16"),
            ("weekday", "uint16"),
            ("day", "uint16"),
            ("hour", "uint16"),
            ("minute", "uint16"),
            ("second", "uint16"),
            ("millisecond", "uint16"),
            ("application_to_create_file", "S32"),
            ("comment_field", "S256"),
            # Number of extended headers
            ("nb_ext_headers", "uint32"),
        ]

        fid = self._get_nev_fid()
        nev_basic_header = read_records_from_opened_file(fid, dtype=dt0, count=1, offset=0)[0]

        shape = nev_basic_header["nb_ext_headers"]
        offset_dt0 = np.dtype(dt0).itemsize

        # This is the common structure of the beginning of extended headers
        dt1 = [("packet_id", "S8"), ("info_field", "S24")]

        raw_ext_header = read_records_from_opened_file(fid, dtype=dt1, count=shape, offset=offset_dt0)

        # Get extended header types for this spec
        header_types = NEV_EXT_HEADER_TYPES_BY_SPEC[spec]

        # Parse extended headers by packet type
        # Strategy: view() entire array first, then mask for efficiency
        # Since all NEV extended header packets are fixed-width (32 bytes), temporarily
        # interpreting a "NEUEVWAV" packet as "ARRAYNME" structure is safe - the raw bytes
        # are just reinterpreted without copying. We immediately filter out mismatched packets
        # with the mask, keeping only those that actually belong to the current packet type.
        nev_ext_header = {}
        for packet_id, dtype_def in header_types.items():
            mask = raw_ext_header["packet_id"] == packet_id
            nev_ext_header[packet_id] = raw_ext_header.view(dtype_def)[mask]

        return nev_basic_header, nev_ext_header

    def _read_nev_data(self, spec, filename):
        """
        Extract nev data for any specification version.

        Parameters
        ----------
        spec : str
            The specification version (e.g., "2.1", "2.2", "2.3", "3.0")
        filename : str
            The NEV filename to read from
        """
        packet_size_bytes = self._nev_basic_header["bytes_in_data_packets"]
        # int(): a uint32 scalar overflows against file sizes > 4 GiB under NumPy 2 (NEP 50)
        header_size = int(self._nev_basic_header["bytes_in_headers"])

        if self._nev_basic_header["ver_major"] >= 3:
            ts_format = "uint64"
            header_skip = 10
        else:
            ts_format = "uint32"
            header_skip = 6

        # read all raw data packets and markers
        dt0 = [("timestamp", ts_format), ("packet_id", "uint16"), ("value", f"S{packet_size_bytes - header_skip}")]

        # expected number of data packets. We are not sure why, but it seems we can get partial data packets
        # based on blackrock's own code this is okay so applying an int to round down is necessary to obtain the
        # memory map of full packets and toss the partial packet.
        # See reference: https://github.com/BlackrockNeurotech/Python-Utilities/blob/fa75aa671680306788e10d3d8dd625f9da4ea4f6/brpylib/brpylib.py#L580-L587
        data_packages_in_bytes = self._get_file_size(filename) - header_size
        n_packets = int(data_packages_in_bytes / packet_size_bytes)

        self._nev_layout = {"dtype": dt0, "header_size": header_size, "n_packets": n_packets}
        self._nev_spikes = None

        if self._fs is None:
            # Local files stay memory-mapped, so the raw packets are not copied into memory.
            raw_data = np.memmap(
                filename,
                offset=header_size,
                dtype=dt0,
                shape=(n_packets,),
                mode="r",
            )
        elif spec == "2.3":
            # 2.3 moves spikes between segments by each spike's own timestamp
            # (_match_nsx_and_nev_segment_ids), which needs every packet at once.
            raw_data = read_records_from_opened_file(
                self._get_nev_fid(), dtype=dt0, offset=header_size, count=n_packets
            )
        else:
            # A remote NEV can be far larger than memory: scan it in pieces and keep
            # everything except the spike packets, which are summarized.
            return self._scan_nev_data(spec)

        packet_identifiers = NEV_PACKET_IDENTIFIERS_BY_SPEC[spec]
        masks, types = self._nev_masks_and_types(raw_data, spec)

        self._count_unmatched_packets(raw_data, masks)
        self._warn_unmatched_packets()
        event_segment_ids = self._get_event_segment_ids(raw_data, masks, spec)

        # Extract data for each packet type using view-then-mask pattern
        # Strategy: reinterpret entire raw_data array with each packet type's structure, then filter
        # All NEV data packets are fixed-width, so temporarily viewing "Spikes" data as "Comments"
        # structure is safe - we immediately filter to keep only packets that actually match.
        # This avoids creating copies of large data arrays during the parsing process.
        data = {}
        for data_type in packet_identifiers:
            mask = masks[data_type]
            data[data_type] = (raw_data.view(types[data_type])[mask], event_segment_ids[mask])

        return data

    def _nev_masks_and_types(self, raw_data, spec):
        """Per packet type: a mask of which packets are that type, and the type's dtype."""
        packet_size_bytes = self._nev_basic_header["bytes_in_data_packets"]
        data_types = NEV_PACKET_DATA_TYPES_BY_SPEC[spec]
        masks = {}
        types = {}
        for data_type, packet_id_spec in NEV_PACKET_IDENTIFIERS_BY_SPEC[spec].items():
            if isinstance(packet_id_spec, tuple):
                # Range check (min, max)
                min_val, max_val = packet_id_spec
                masks[data_type] = (min_val <= raw_data["packet_id"]) & (raw_data["packet_id"] <= max_val)
            else:
                # Equality check
                masks[data_type] = raw_data["packet_id"] == packet_id_spec
            types[data_type] = data_types[data_type](packet_size_bytes)
        return masks, types

    def _iter_nev_pieces(self):
        """Yield (index of the first packet, packets) over the whole NEV, a piece at a time."""
        layout = self._nev_layout
        packet_size = np.dtype(layout["dtype"]).itemsize
        step = self._nev_packets_per_piece
        for start in range(0, layout["n_packets"], step):
            count = min(step, layout["n_packets"] - start)
            raw = read_records_from_opened_file(
                self._get_nev_fid(),
                dtype=layout["dtype"],
                offset=layout["header_size"] + start * packet_size,
                count=count,
            )
            yield start, raw

    def _nev_piece_segment_ids(self, raw, masks, spec, start, state):
        """
        Segment ids for one piece of NEV packets (the whole file is one piece).

        A segment starts at a reset comment, or at a packet whose timestamp is lower than the
        last packet of the same type since the segment began: a clock reset sends every packet
        type's clock back. Packets of different types are not compared: the format does not
        order them, and comments can be stamped a few ms after the spikes written after them,
        which compared across types looked like a reset each time. Comments are not compared at all:
        they are not in time order among themselves either, so a comment gets the segment of the
        packets around it in the file (a reset comment still starts one). `state` carries the last
        timestamp and position of each type, and the boundaries so far, from piece to piece.

        A packet stamped earlier than the last of its type by at most REORDER_WINDOW_S was written out of
        order, not after a reset: it is counted in `state["reordered"]` and starts no segment.
        """
        n = len(raw)
        if spec in ["2.1", "2.2"]:
            # No pause or reset mechanism present for file version 2.1 and 2.2
            return np.zeros(n, dtype=int)
        reset_pos = np.flatnonzero(self._get_reset_event_mask(raw, masks, spec))

        # Candidate jumps per packet type: a packet earlier than the previous one of its type.
        # Each is kept with the position of the packet it was compared with.
        candidates = []  # (position, position compared with, is a reset comment)
        for data_type, mask in masks.items():
            pos = np.flatnonzero(mask)
            if len(pos) == 0 or data_type == "Comments":
                continue
            ts = raw["timestamp"][pos]
            if data_type in state["last"]:
                last_ts, last_pos = state["last"][data_type]
                prev_ts = np.concatenate(([last_ts], ts[:-1]))
                prev_pos = np.concatenate(([last_pos], pos[:-1] + start))
            else:
                prev_ts, prev_pos, ts, pos = ts[:-1], pos[:-1] + start, ts[1:], pos[1:]
            jumps = np.flatnonzero(ts < prev_ts)
            late = prev_ts[jumps] - ts[jumps] <= REORDER_WINDOW_S * self._nev_basic_header["timestamp_resolution"]
            state["reordered"] += int(np.sum(late))
            jumps = jumps[~late]
            candidates.extend(zip((pos[jumps] + start).tolist(), prev_pos[jumps].tolist(), [False] * len(jumps)))
            state["last"][data_type] = (raw["timestamp"][mask][-1], int(np.flatnonzero(mask)[-1]) + start)
        candidates.extend((int(r) + start, -1, True) for r in reset_pos)

        # Accept in file order. A jump counts only if the packet it was compared with is in the
        # current segment, so one reset is one boundary however many packet types it sends back.
        reset_at = {int(r) + start for r in reset_pos}
        boundaries = []
        for position, compared_with, is_reset in sorted(candidates):
            if boundaries and position == boundaries[-1]:
                continue  # a reset comment and a jump at the same packet are one boundary
            if is_reset or compared_with >= state["segment_start"]:
                boundaries.append(position)
                state["segment_start"] = position
                if position not in reset_at:
                    state["undocumented"].append(position)
        local = np.array(sorted(boundaries), dtype=np.int64) - start
        ids = state["boundaries"] + np.searchsorted(local, np.arange(n), side="right")
        state["boundaries"] += len(local)
        return ids

    @staticmethod
    def _new_nev_segment_state():
        return {"last": {}, "segment_start": -1, "boundaries": 0, "undocumented": [], "reordered": 0}

    def _scan_nev_data(self, spec):
        """
        Read a NEV in pieces. Returns the same dict as _read_nev_data, without "Spikes":
        spike packets are most of a NEV, so only a summary of them is kept, in
        self._nev_spikes (units per channel, and per segment the first, last, lowest and
        highest spike timestamp). Spikes themselves are read again on demand.
        """
        packet_identifiers = NEV_PACKET_IDENTIFIERS_BY_SPEC[spec]
        kept = {data_type: ([], []) for data_type in packet_identifiers if data_type != "Spikes"}
        kept_types = None
        units = {}
        seg_times = {}  # raw segment id -> [first, last, lowest, highest]
        counts = {}  # (raw segment id, channel id, unit id) -> spikes
        state = self._new_nev_segment_state()
        for start, raw in self._iter_nev_pieces():
            masks, types = self._nev_masks_and_types(raw, spec)
            self._count_unmatched_packets(raw, masks)
            kept_types = types
            ids = self._nev_piece_segment_ids(raw, masks, spec, start, state)
            for data_type, (records, record_ids) in kept.items():
                records.append(raw.view(types[data_type])[masks[data_type]])
                record_ids.append(ids[masks[data_type]])
            spikes = raw.view(types["Spikes"])[masks["Spikes"]]
            spike_ids = ids[masks["Spikes"]]
            if len(spikes) == 0:
                continue
            for channel_id, unit_id in np.unique(
                np.stack((spikes["packet_id"], spikes["unit_class_nb"]), axis=1), axis=0
            ):
                units.setdefault(int(channel_id), set()).add(int(unit_id))
            keys, n = np.unique(
                np.stack((spike_ids, spikes["packet_id"], spikes["unit_class_nb"]), axis=1), axis=0, return_counts=True
            )
            for (seg, channel_id, unit_id), count in zip(keys.tolist(), n.tolist()):
                counts[(seg, channel_id, unit_id)] = counts.get((seg, channel_id, unit_id), 0) + count
            for seg in np.unique(spike_ids):
                seg_ts = spikes["timestamp"][spike_ids == seg]
                times = seg_times.setdefault(int(seg), [seg_ts[0], seg_ts[-1], seg_ts.min(), seg_ts.max()])
                times[1] = seg_ts[-1]
                times[2] = min(times[2], seg_ts.min())
                times[3] = max(times[3], seg_ts.max())

        if state["undocumented"]:
            additional_ids = np.array(state["undocumented"])
            warnings.warn(
                f"Detected {len(additional_ids)} undocumented segments within "
                f"nev data after timestamps {additional_ids}."
                ""
            )
        if spec in ["2.3", "3.0"]:
            self._nb_segment_nev = state["boundaries"] + 1
        self._warn_reordered_nev_packets(state)
        self._warn_unmatched_packets()

        if kept_types is None:  # a NEV with no packets
            _, kept_types = self._nev_masks_and_types(np.empty(0, dtype=self._nev_layout["dtype"]), spec)
        data = {}
        for data_type, (records, record_ids) in kept.items():
            data[data_type] = (
                np.concatenate(records) if records else np.empty(0, dtype=kept_types[data_type]),
                np.concatenate(record_ids) if record_ids else np.empty(0, dtype=int),
            )
        self._nev_spikes = {
            "units": units,
            "seg_times": seg_times,
            "counts": counts,
            "deleted_segments": [],
            "types": kept_types,
        }
        return data

    def _spike_segment_ids(self, raw_ids):
        """Segment ids of scanned spikes after the same removals _delete_empty_segments made to the others."""
        ids = np.array(raw_ids, copy=True)
        for seg_index in self._nev_spikes["deleted_segments"]:
            ids[ids > seg_index] -= 1
        return ids

    def _spike_times_by_segment(self):
        """For a scanned NEV: per segment, the first, last, lowest and highest spike timestamp."""
        merged = {}
        for raw_seg in sorted(self._nev_spikes["seg_times"]):
            first, last, lowest, highest = self._nev_spikes["seg_times"][raw_seg]
            seg = int(self._spike_segment_ids([raw_seg])[0])
            if seg in merged:
                times = merged[seg]
                merged[seg] = [times[0], last, min(times[2], lowest), max(times[3], highest)]
            else:
                merged[seg] = [first, last, lowest, highest]
        return merged

    def _spike_records(self, channel_id, unit_id):
        """
        (spike records, segment ids) to select one unit's spikes from. In memory, every spike;
        for a scanned NEV, this unit's spikes, read with one more pass over the file.
        """
        if self._nev_spikes is None:
            return self.nev_data["Spikes"]
        records, record_ids = [], []
        state = self._new_nev_segment_state()
        for start, raw in self._iter_nev_pieces():
            masks, types = self._nev_masks_and_types(raw, self._nev_spec)
            ids = self._nev_piece_segment_ids(raw, masks, self._nev_spec, start, state)
            spikes = raw.view(types["Spikes"])
            mask = masks["Spikes"] & (spikes["packet_id"] == channel_id) & (spikes["unit_class_nb"] == unit_id)
            records.append(spikes[mask])
            record_ids.append(ids[mask])
        spike_type = self._nev_spikes["types"]["Spikes"]
        all_spikes = np.concatenate(records) if records else np.empty(0, dtype=spike_type)
        segment_ids = np.concatenate(record_ids) if record_ids else np.empty(0, dtype=int)
        return all_spikes, self._spike_segment_ids(segment_ids)

    def _nev_unit_ids(self, channel_id):
        """Sorted unit ids that have spikes on one channel."""
        if self._nev_spikes is not None:
            return np.array(sorted(self._nev_spikes["units"].get(channel_id, ())), dtype=np.uint8)
        spikes = self.nev_data["Spikes"][0]
        return np.unique(spikes[spikes["packet_id"] == channel_id]["unit_class_nb"])

    def _spike_waveforms(self, unit_spikes, channel_id):
        """Raw waveforms (n, 1, samples) of one channel's spike records."""
        wf_dtype = self._nev_params("waveform_dtypes")[channel_id]
        wf_size = self._nev_params("waveform_size")[channel_id]
        wf_byte_size = np.dtype(wf_dtype).itemsize * wf_size

        dt1 = [
            ("extra", "S{}".format(unit_spikes["waveform"].dtype.itemsize - wf_byte_size)),
            ("ch_waveform", "S{}".format(wf_byte_size)),
        ]

        waveforms = unit_spikes["waveform"].view(dt1)["ch_waveform"].flatten().view(wf_dtype)
        return waveforms.reshape(int(unit_spikes.size), 1, int(wf_size))

    def iter_spikes(self, seg_index=None, packets_per_piece=None):
        """
        Yield spikes a piece at a time, for every unit, reading the NEV once.

        neo's per-unit methods (get_spike_timestamps, get_spike_raw_waveforms) each have to
        look at every packet, because a unit's spikes are spread through the whole file; on a
        large remote NEV, calling them for every unit reads the file once per unit. This reads
        it once in total, for every segment (or only `seg_index`), and memory stays at about
        one piece.

        Each item is (seg_index, {unit_index: (timestamps, waveforms)}): raw timestamps and raw
        waveforms shaped (n, 1, samples), as the per-unit methods return them. Items come in
        timestamp order, so concatenating a unit's arrays over the items of one segment equals what
        those methods return for that segment.
        """
        step = packets_per_piece or self._nev_packets_per_piece
        unit_index_of = {pair: i for i, pair in enumerate(self.internal_unit_ids)}

        def pieces():
            if self._nev_spikes is None:
                spikes, ids = self.nev_data["Spikes"]
                for start in range(0, len(spikes), step):
                    yield spikes[start : start + step], ids[start : start + step]
                return
            state = self._new_nev_segment_state()
            saved, self._nev_packets_per_piece = self._nev_packets_per_piece, step
            try:
                for start, raw in self._iter_nev_pieces():
                    masks, types = self._nev_masks_and_types(raw, self._nev_spec)
                    ids = self._nev_piece_segment_ids(raw, masks, self._nev_spec, start, state)
                    yield raw.view(types["Spikes"])[masks["Spikes"]], self._spike_segment_ids(ids[masks["Spikes"]])
            finally:
                self._nev_packets_per_piece = saved

        def in_time_order(pieces):
            # A spike can be written after spikes stamped up to REORDER_WINDOW_S later, in the next piece too:
            # sort each piece and hold back the spikes a later piece could still hold an earlier one than.
            window = REORDER_WINDOW_S * self._nev_basic_header["timestamp_resolution"]
            held = None
            for spikes, ids in pieces:
                if held is not None:
                    spikes, ids = np.concatenate((held[0], spikes)), np.concatenate((held[1], ids))
                order = np.lexsort((spikes["timestamp"], ids))
                spikes, ids = spikes[order], ids[order]
                hold = np.zeros(len(ids), dtype=bool)
                if len(ids):
                    hold = (ids == ids[-1]) & (spikes["timestamp"] >= spikes["timestamp"][-1] - window)
                yield spikes[~hold], ids[~hold]
                held = spikes[hold], ids[hold]
            if held is not None:
                yield held

        for spikes, ids in in_time_order(pieces()):
            for seg in [seg_index] if seg_index is not None else np.unique(ids).tolist():
                seg_spikes = spikes[ids == seg]
                if len(seg_spikes) == 0:
                    continue
                piece = {}
                keys = np.stack((seg_spikes["packet_id"], seg_spikes["unit_class_nb"]), axis=1)
                for channel_id, unit_id in np.unique(keys, axis=0):
                    unit_index = unit_index_of.get((int(channel_id), int(unit_id)))
                    if unit_index is None:
                        continue
                    unit_spikes = seg_spikes[(keys[:, 0] == channel_id) & (keys[:, 1] == unit_id)]
                    piece[unit_index] = (
                        unit_spikes["timestamp"],
                        self._spike_waveforms(unit_spikes, int(channel_id)),
                    )
                if piece:
                    yield int(seg), piece

    def _count_unmatched_packets(self, raw_data, masks):
        """Add the packets of `raw_data` that match no packet type to nev_unmatched_packets."""
        matched = np.any([mask for mask in masks.values()], axis=0) if masks else np.zeros(len(raw_data), bool)
        ids, counts = np.unique(raw_data["packet_id"][~matched], return_counts=True)
        for packet_id, count in zip(ids.tolist(), counts.tolist()):
            self.nev_unmatched_packets[packet_id] = self.nev_unmatched_packets.get(packet_id, 0) + count

    def _warn_unmatched_packets(self):
        if self.nev_unmatched_packets:
            total = sum(self.nev_unmatched_packets.values())
            ids = sorted(self.nev_unmatched_packets)
            shown = ", ".join(str(i) for i in ids[:10]) + (", ..." if len(ids) > 10 else "")
            warnings.warn(
                f"{total} NEV data packets match no packet type of FileSpec {self._nev_spec} and "
                f"are not read (packet ids {shown}); see nev_unmatched_packets."
            )

    def _warn_reordered_nev_packets(self, state):
        self.nev_reordered_packets = state["reordered"]
        if self.nev_reordered_packets:
            warnings.warn(
                f"{self.nev_reordered_packets} NEV packets were written after packets of their type stamped up "
                f"to {REORDER_WINDOW_S * 1000:g} ms later. They are not clock resets; spikes are returned in "
                "timestamp order, events in file order. See nev_reordered_packets."
            )

    def _get_reset_event_mask(self, raw_event_data, masks, spec):
        """
        Extract mask for reset comment events in 2.3 .nev file
        """
        if "Comments" not in masks:
            return np.zeros(len(raw_event_data), dtype=bool)

        restart_mask = np.logical_and(
            masks["Comments"],
            raw_event_data["value"] == b"\x00\x00\x00\x00\x00\x00critical load restart",
        )
        # TODO: Fix hardcoded number of bytes
        return restart_mask

    def _get_event_segment_ids(self, raw_event_data, masks, spec):
        """
        Construct array of corresponding segment ids for each event for nev version 2.3
        """

        if spec in ["2.1", "2.2"]:
            # No pause or reset mechanism present for file version 2.1 and 2.2
            return np.zeros(len(raw_event_data), dtype=int)

        elif spec in ["2.3", "3.0"]:
            state = self._new_nev_segment_state()
            event_segment_ids = self._nev_piece_segment_ids(raw_event_data, masks, spec, 0, state)
            if state["undocumented"]:
                additional_ids = np.array(state["undocumented"])
                warnings.warn(
                    f"Detected {len(additional_ids)} undocumented segments within "
                    f"nev data after timestamps {additional_ids}."
                    ""
                )
            self._nb_segment_nev = state["boundaries"] + 1
            self._warn_reordered_nev_packets(state)
            return event_segment_ids

        else:
            raise ValueError(f"Unknown File Spec {spec}")

    def _match_nsx_and_nev_segment_ids(self, nsx_nb):
        """
        Ensure matching ids of segments detected in nsx and nev file for version 2.3
        """

        # NSX required for matching, if not available, warn the user
        if not self._avail_nsx:
            warnings.warn(
                "No nsX available so it cannot be checked whether "
                "the segments in nev are all correct. Most importantly, "
                "recording pauses will not be detected",
                UserWarning,
            )
            return

        # Only needs to be done for nev version 2.3
        if self._nev_spec == "2.3":
            nsx_offset = self._nsx_data_header[nsx_nb][0]["timestamp"]
            # Multiples of 1/30.000s that pass between two nsX samples
            nsx_period = self._nsx_basic_header[nsx_nb]["period"]
            # Counts how many segments CAN be created from nev
            nb_possible_nev_segments = self._nb_segment_nev

            # Nonempty segments are those containing at least 2 samples
            # These have to be able to be mapped to nev
            nonempty_nsx_segment_indices = [
                seg_index for seg_index, seg in enumerate(self._nsx_data_header[nsx_nb]) if seg["nb_data_points"] > 1
            ]
            nonempty_nsx_segments = [
                self._nsx_data_header[nsx_nb][seg_index] for seg_index in nonempty_nsx_segment_indices
            ]

            # Account for paused segments
            # This increases nev event segment ids if from the nsx an additional segment is found
            # If one new segment, i.e. that could not be determined from the nev was found,
            # all following ids need to be increased to account for the additional segment before
            for k, (data, ev_ids) in self.nev_data.items():

                # Check all nonempty nsX segments
                for nonempty_segment_index, seg in enumerate(nonempty_nsx_segments[:]):

                    # Last timestamp in this nsX segment
                    # Not subtracting nsX offset from end because spike extraction might continue
                    end_of_current_nsx_seg = (
                        seg["timestamp"] + seg["nb_data_points"] * self._nsx_basic_header[nsx_nb]["period"]
                    )

                    mask_after_seg = (ev_ids == nonempty_segment_index) & (
                        data["timestamp"] > end_of_current_nsx_seg + nsx_period
                    )

                    # Show warning if spikes do not fit any segment (+- 1 sampling 'tick')
                    # Spike should belong to segment before
                    mask_outside = (ev_ids == nonempty_segment_index) & (
                        data["timestamp"] < int(seg["timestamp"]) - int(nsx_offset) - int(nsx_period)
                    )

                    if len(data[mask_outside]) > 0:
                        warnings.warn(f"Spikes outside any segment. Detected on segment #{nonempty_segment_index}")
                        ev_ids[mask_outside] -= 1

                    # If some nev data are outside of this nsX segment, increase their segment ids
                    # and the ids of all following segments. They are checked for the next nsX
                    # segment then. If they do not fit any of them,
                    # a warning will be shown, indicating how far outside the segment spikes are
                    # If they fit the next segment, more segments are possible in nev,
                    # because a new one has been discovered
                    if len(data[mask_after_seg]) > 0:
                        # Warning if spikes are after last segment
                        if nonempty_segment_index == len(nonempty_nsx_segments) - 1:
                            # Get timestamp resolution from header (available for v2.2+)
                            timestamp_resolution = self._nsx_basic_header[nsx_nb]["timestamp_resolution"]
                            time_after_seg = (
                                data[mask_after_seg]["timestamp"][-1] - end_of_current_nsx_seg
                            ) / timestamp_resolution
                            warnings.warn(f"Spikes {time_after_seg}s after last segment.")
                            # Break out of loop because it's the last iteration
                            # and the spikes should stay connected to last segment
                            break

                        # If reset and no segment detected in nev, then these segments cannot be
                        # distinguished in nev, which is a big problem
                        # XXX 96 is an arbitrary number based on observations in available files
                        elif nonempty_nsx_segments[nonempty_segment_index + 1]["timestamp"] - nsx_offset <= 96:
                            # If not all definitely belong to the next segment,
                            # then it cannot be distinguished where some belong
                            if len(data[ev_ids == nonempty_segment_index]) != len(data[mask_after_seg]):
                                raise ValueError("Some segments in nsX cannot be detected in nev")

                        # Actual processing if no problem has occurred
                        nb_possible_nev_segments += 1
                        ev_ids[ev_ids > nonempty_segment_index] += 1
                        ev_ids[mask_after_seg] += 1

            # consistency check: same number of segments for nsx and nev data
            if nb_possible_nev_segments != len(nonempty_nsx_segments):
                raise NeoReadWriteError(
                    f"Inconsistent ns{nsx_nb} and nev file. {nb_possible_nev_segments} "
                    f"segments present in .nev file, but {len(nonempty_nsx_segments)} in "
                    f"ns{nsx_nb} file."
                )

            new_nev_segment_id_mapping = dict(zip(range(nb_possible_nev_segments), nonempty_nsx_segment_indices))

            # replacing event ids by matched event ids in place
            for k, (data, ev_ids) in self.nev_data.items():
                if len(ev_ids):
                    ev_ids[:] = np.vectorize(new_nev_segment_id_mapping.__getitem__)(ev_ids)

    def _nev_params(self, param_name):
        """
        Returns wanted nev parameter.
        """
        nev_parameters = {
            "bytes_in_data_packets": self._nev_basic_header["bytes_in_data_packets"],
            "rec_datetime": datetime.datetime(
                year=self._nev_basic_header["year"],
                month=self._nev_basic_header["month"],
                day=self._nev_basic_header["day"],
                hour=self._nev_basic_header["hour"],
                minute=self._nev_basic_header["minute"],
                second=self._nev_basic_header["second"],
                microsecond=int(self._nev_basic_header["millisecond"]) * 1000,
            ),
            "max_res": self._nev_basic_header["timestamp_resolution"],
            "channel_ids": self._nev_ext_header[b"NEUEVWAV"]["electrode_id"],
            "channel_labels": self._channel_labels[self._nev_spec](),
            "event_unit": pq.CompoundUnit(f"1.0/{self._nev_basic_header['timestamp_resolution']} * s"),
            "nb_units": dict(
                zip(
                    self._nev_ext_header[b"NEUEVWAV"]["electrode_id"],
                    self._nev_ext_header[b"NEUEVWAV"]["nb_sorted_units"],
                )
            ),
            "digitization_factor": dict(
                zip(
                    self._nev_ext_header[b"NEUEVWAV"]["electrode_id"],
                    self._nev_ext_header[b"NEUEVWAV"]["digitization_factor"],
                )
            ),
            "data_size": self._nev_basic_header["bytes_in_data_packets"],
            "waveform_size": self._waveform_size[self._nev_spec](),
            "waveform_dtypes": self._get_waveforms_dtype(),
            "waveform_sampling_rate": self._nev_basic_header["sample_resolution"] * pq.Hz,
            "waveform_time_unit": pq.CompoundUnit(f"1.0/{self._nev_basic_header['sample_resolution']} * s"),
            "waveform_unit": pq.uV,
        }

        return nev_parameters[param_name]

    def _get_file_size(self, filename: str) -> int:
        """
        Returns the file size in bytes for the given file.
        """
        if self._fs is None:
            filebuf = open(filename, "rb")
            filebuf.seek(0, os.SEEK_END)
            file_size = int(filebuf.tell())
            filebuf.close()
        else:
            file_size = self._fs.size(filename)

        return file_size

    def _get_min_time(self):
        """
        Returns the smallest time that can be determined from the recording for
        use as the lower bound n in an interval [n,m).
        """
        tp = []
        if self._avail_files["nev"]:
            tp.extend(self._get_nev_rec_times()[0])
        for nsx_i in self._avail_nsx:
            tp.extend(self._nsx_rec_times[self._nsx_spec[nsx_i]](nsx_i)[0])

        return min(tp)

    def _get_max_time(self):
        """
        Returns the largest time that can be determined from the recording for
        use as the upper bound m in an interval [n,m).
        """
        tp = []
        if self._avail_files["nev"]:
            tp.extend(self._get_nev_rec_times()[1])
        for nsx_i in self._avail_nsx:
            tp.extend(self._nsx_rec_times[self._nsx_spec[nsx_i]](nsx_i)[1])

        return max(tp)

    def _get_nev_rec_times(self):
        """
        Extracts minimum and maximum time points from a nev file.
        """
        filename = ".".join([self._filenames["nev"], "nev"])

        dt = [("timestamp", "uint32")]
        offset = self._get_file_size(filename) - int(self._nev_params("bytes_in_data_packets"))
        last_data_packet = np.memmap(filename, offset=offset, dtype=dt, mode="r")[0]

        n_starts = [0 * self._nev_params("event_unit")]
        n_stops = [last_data_packet["timestamp"] * self._nev_params("event_unit")]

        return n_starts, n_stops

    def _get_waveforms_dtype(self):
        """
        Extracts the actual waveform dtype set for each channel.
        """
        # Blackrock code giving the appropriate dtype
        conv = {0: "int8", 1: "int8", 2: "int16", 4: "int32"}

        # get all electrode ids from nev ext header
        all_el_ids = self._nev_ext_header[b"NEUEVWAV"]["electrode_id"]

        # get the dtype of waveform (this is stupidly complicated)
        if self._is_set(np.array(self._nev_basic_header["additionnal_flags"]), 0):
            dtype_waveforms = {k: "int16" for k in all_el_ids}
        else:
            # extract bytes per waveform
            waveform_bytes = self._nev_ext_header[b"NEUEVWAV"]["bytes_per_waveform"]
            # extract dtype for waveforms fro each electrode
            dtype_waveforms = dict(zip(all_el_ids, conv[waveform_bytes]))

        return dtype_waveforms

    def _get_channel_labels_spec_v21(self):
        """
        Returns labels for all channels for file spec 2.1
        """
        elids = self._nev_ext_header[b"NEUEVWAV"]["electrode_id"]
        labels = []

        for elid in elids:
            if elid < 129:
                labels.append(f"chan{elid}")
            else:
                labels.append(f"ainp{(elid - 12 +1 )}")

        return dict(zip(elids, labels))

    def _get_channel_labels_spec_v22_30(self):
        """
        Returns labels for all channels for file spec 2.2 and 2.3
        """
        elids = self._nev_ext_header[b"NEUEVWAV"]["electrode_id"]
        labels = self._nev_ext_header[b"NEUEVLBL"]["label"]

        return dict(zip(elids, labels)) if len(labels) > 0 else None

    def _get_waveform_size_spec_v21(self):
        """
        Returns waveform sizes for all channels for file spec 2.1 and 2.2
        """
        wf_dtypes = self._get_waveforms_dtype()
        nb_bytes_wf = self._nev_basic_header["bytes_in_data_packets"] - 8

        wf_sizes = {ch: int(nb_bytes_wf / np.dtype(dt).itemsize) for ch, dt in wf_dtypes.items()}

        return wf_sizes

    def _get_waveform_size_spec_v22_30(self):
        """
        Returns waveform sizes for all channels for file spec 2.3
        """
        elids = self._nev_ext_header[b"NEUEVWAV"]["electrode_id"]
        spike_widths = self._nev_ext_header[b"NEUEVWAV"]["spike_width"]

        return dict(zip(elids, spike_widths))

    def _get_left_sweep_waveforms(self):
        """
        Returns left sweep of waveforms for each channel. Left sweep is defined
        as the time from the beginning of the waveform to the trigger time of
        the corresponding spike.
        """
        # TODO: Double check if this is the actual setting for Blackrock
        wf_t_unit = self._nev_params("waveform_time_unit")
        all_ch = self._nev_params("channel_ids")

        # TODO: Double check if this is the correct assumption (10 samples)
        # default value: threshold crossing after 10 samples of waveform
        wf_left_sweep = {ch: 10 * wf_t_unit for ch in all_ch}

        # non-default: threshold crossing at center of waveform
        # wf_size = self._nev_params('waveform_size')
        # wf_left_sweep = dict(
        #     [(ch, (wf_size[ch] / 2) * wf_t_unit) for ch in all_ch])

        return wf_left_sweep

    def _get_nonneural_evdicts_spec_v21_22(self, data):
        """
        Defines event types and the necessary parameters to extract them from
        a 2.1 and 2.2 nev file.
        """
        # TODO: add annotations of nev ext header (NSASEXEX) to event types

        # digital events
        event_types = {
            "digital_input_port": {
                "name": "digital_input_port",
                "field": "digital_input",
                "mask": data["packet_insertion_reason"] == 1,
                "desc": "Events of the digital input port",
            },
            "serial_input_port": {
                "name": "serial_input_port",
                "field": "digital_input",
                "mask": data["packet_insertion_reason"] == 129,
                "desc": "Events of the serial input port",
            },
        }

        # analog input events via threshold crossings
        for ch in range(5):
            event_types.update(
                {
                    f"analog_input_channel_{ch + 1}": {
                        "name": f"analog_input_channel_{ch + 1}",
                        "field": f"analog_input_channel_{ch + 1}",
                        "mask": self._is_set(data["packet_insertion_reason"], ch + 1),
                        "desc": f"Values of analog input channel {ch + 1} in mV " "(+/- 5000)",
                    }
                }
            )

        # TODO: define field and desc
        event_types.update(
            {
                "periodic_sampling_events": {
                    "name": "periodic_sampling_events",
                    "field": "digital_input",
                    "mask": self._is_set(data["packet_insertion_reason"], 6),
                    "desc": "Periodic sampling event of a certain frequency",
                }
            }
        )

        return event_types

    def _delete_empty_segments(self):
        """
        If there are empty segments (e.g. due to a reset or clock synchronization across
        two systems), these can be discarded.
        If this is done, all the data and data_headers need to be remapped to become a range
        starting from 0 again. Nev data are mapped accordingly to stay with their corresponding
        segment in the nsX data.
        """

        # Find empty segments (fewer than 2 samples across all nsx files)
        empty_indices = []
        for seg_index in range(self._nb_segment):
            keep_seg = True
            for nsx_nb in self.nsx_to_load:
                length = self._nsx_data_header[nsx_nb][seg_index]["nb_data_points"]
                keep_seg = keep_seg and (length >= 2)
            if not keep_seg:
                empty_indices.append(seg_index)

        # Remove empty segments in reverse order to preserve indices
        for seg_index in reversed(empty_indices):
            for nsx_nb in self.nsx_to_load:
                del self._nsx_data_header[nsx_nb][seg_index]

            # Remap nev segment ids: shift down all ids above the removed segment
            if self._avail_files["nev"]:
                for _data, ev_ids in self.nev_data.values():
                    ev_ids[ev_ids > seg_index] -= 1
                if self._nev_spikes is not None:
                    self._nev_spikes["deleted_segments"].append(seg_index)

            self._nb_segment -= 1

    def _get_nonneural_evdicts_spec_v23(self, data):
        """
        Defines event types and the necessary parameters to extract them from
        a 2.3 nev file.
        """
        # digital events
        if not np.all(np.isin(data["packet_insertion_reason"], [1, 129])):
            # Blackrock spec gives reason==64 means PERIODIC, but never seen this live
            warnings.warn("Unknown event codes found", RuntimeWarning)
        event_types = {
            "digital_input_port": {
                "name": "digital_input_port",
                "field": "digital_input",
                "mask": self._is_set(data["packet_insertion_reason"], 0)
                & ~self._is_set(data["packet_insertion_reason"], 7),
                "desc": "Events of the digital input port",
            },
            "serial_input_port": {
                "name": "serial_input_port",
                "field": "digital_input",
                "mask": self._is_set(data["packet_insertion_reason"], 0)
                & self._is_set(data["packet_insertion_reason"], 7),
                "desc": "Events of the serial input port",
            },
        }

        return event_types

    def _get_comment_evdict_spec_v21_22(self, data):
        return {
            "comments": {"name": "comments", "field": "comment", "mask": data["packet_id"] == 65535, "desc": "Comments"}
        }

    def _is_set(self, flag, pos):
        """
        Checks if bit is set at the given position for flag. If flag is an
        array, an array will be returned.
        """
        return flag & (1 << pos) > 0


# Extended header types for different NEV file specifications
# Structure: {spec: {packet_id: data_type_definition}}
NEV_EXT_HEADER_TYPES_BY_SPEC = {
    "2.1": {
        b"NEUEVWAV": [
            ("packet_id", "S8"),
            ("electrode_id", "uint16"),
            ("physical_connector", "uint8"),
            ("connector_pin", "uint8"),
            ("digitization_factor", "uint16"),
            ("energy_threshold", "uint16"),
            ("hi_threshold", "int16"),
            ("lo_threshold", "int16"),
            ("nb_sorted_units", "uint8"),
            ("bytes_per_waveform", "uint8"),
            ("unused", "S10"),
        ],
        b"ARRAYNME": [("packet_id", "S8"), ("electrode_array_name", "S24")],
        b"ECOMMENT": [("packet_id", "S8"), ("extra_comment", "S24")],
        b"CCOMMENT": [("packet_id", "S8"), ("continued_comment", "S24")],
        b"MAPFILE": [("packet_id", "S8"), ("mapFile", "S24")],
        b"NSASEXEV": [
            ("packet_id", "S8"),
            ("frequency", "uint16"),
            ("digital_input_config", "uint8"),
            ("analog_channel_1_config", "uint8"),
            ("analog_channel_1_edge_detec_val", "uint16"),
            ("analog_channel_2_config", "uint8"),
            ("analog_channel_2_edge_detec_val", "uint16"),
            ("analog_channel_3_config", "uint8"),
            ("analog_channel_3_edge_detec_val", "uint16"),
            ("analog_channel_4_config", "uint8"),
            ("analog_channel_4_edge_detec_val", "uint16"),
            ("analog_channel_5_config", "uint8"),
            ("analog_channel_5_edge_detec_val", "uint16"),
            ("unused", "S6"),
        ],
    },
    "2.2": {
        b"NEUEVWAV": [
            ("packet_id", "S8"),
            ("electrode_id", "uint16"),
            ("physical_connector", "uint8"),
            ("connector_pin", "uint8"),
            ("digitization_factor", "uint16"),
            ("energy_threshold", "uint16"),
            ("hi_threshold", "int16"),
            ("lo_threshold", "int16"),
            ("nb_sorted_units", "uint8"),
            ("bytes_per_waveform", "uint8"),
            ("spike_width", "uint16"),
            ("unused", "S8"),
        ],
        b"ARRAYNME": [("packet_id", "S8"), ("electrode_array_name", "S24")],
        b"ECOMMENT": [("packet_id", "S8"), ("extra_comment", "S24")],
        b"CCOMMENT": [("packet_id", "S8"), ("continued_comment", "S24")],
        b"MAPFILE": [("packet_id", "S8"), ("mapFile", "S24")],
        b"NEUEVLBL": [
            ("packet_id", "S8"),
            ("electrode_id", "uint16"),
            ("label", "S16"),
            ("unused", "S6"),
        ],
        b"NEUEVFLT": [
            ("packet_id", "S8"),
            ("electrode_id", "uint16"),
            ("hi_freq_corner", "uint32"),
            ("hi_freq_order", "uint32"),
            ("hi_freq_type", "uint16"),
            ("lo_freq_corner", "uint32"),
            ("lo_freq_order", "uint32"),
            ("lo_freq_type", "uint16"),
            ("unused", "S2"),
        ],
        b"DIGLABEL": [
            ("packet_id", "S8"),
            ("label", "S16"),
            ("mode", "uint8"),
            ("unused", "S7"),
        ],
        b"NSASEXEV": [
            ("packet_id", "S8"),
            ("frequency", "uint16"),
            ("digital_input_config", "uint8"),
            ("analog_channel_1_config", "uint8"),
            ("analog_channel_1_edge_detec_val", "uint16"),
            ("analog_channel_2_config", "uint8"),
            ("analog_channel_2_edge_detec_val", "uint16"),
            ("analog_channel_3_config", "uint8"),
            ("analog_channel_3_edge_detec_val", "uint16"),
            ("analog_channel_4_config", "uint8"),
            ("analog_channel_4_edge_detec_val", "uint16"),
            ("analog_channel_5_config", "uint8"),
            ("analog_channel_5_edge_detec_val", "uint16"),
            ("unused", "S6"),
        ],
    },
    "2.3": {
        b"NEUEVWAV": [
            ("packet_id", "S8"),
            ("electrode_id", "uint16"),
            ("physical_connector", "uint8"),
            ("connector_pin", "uint8"),
            ("digitization_factor", "uint16"),
            ("energy_threshold", "uint16"),
            ("hi_threshold", "int16"),
            ("lo_threshold", "int16"),
            ("nb_sorted_units", "uint8"),
            ("bytes_per_waveform", "uint8"),
            ("spike_width", "uint16"),
            ("unused", "S8"),
        ],
        b"ARRAYNME": [("packet_id", "S8"), ("electrode_array_name", "S24")],
        b"ECOMMENT": [("packet_id", "S8"), ("extra_comment", "S24")],
        b"CCOMMENT": [("packet_id", "S8"), ("continued_comment", "S24")],
        b"MAPFILE": [("packet_id", "S8"), ("mapFile", "S24")],
        b"NEUEVLBL": [
            ("packet_id", "S8"),
            ("electrode_id", "uint16"),
            ("label", "S16"),
            ("unused", "S6"),
        ],
        b"NEUEVFLT": [
            ("packet_id", "S8"),
            ("electrode_id", "uint16"),
            ("hi_freq_corner", "uint32"),
            ("hi_freq_order", "uint32"),
            ("hi_freq_type", "uint16"),
            ("lo_freq_corner", "uint32"),
            ("lo_freq_order", "uint32"),
            ("lo_freq_type", "uint16"),
            ("unused", "S2"),
        ],
        b"DIGLABEL": [
            ("packet_id", "S8"),
            ("label", "S16"),
            ("mode", "uint8"),
            ("unused", "S7"),
        ],
        b"VIDEOSYN": [
            ("packet_id", "S8"),
            ("video_source_id", "uint16"),
            ("video_source", "S16"),
            ("frame_rate", "float32"),
            ("unused", "S2"),
        ],
        b"TRACKOBJ": [
            ("packet_id", "S8"),
            ("trackable_type", "uint16"),
            ("trackable_id", "uint16"),
            ("point_count", "uint16"),
            ("video_source", "S16"),
            ("unused", "S2"),
        ],
        b"NSASEXEV": [
            ("packet_id", "S8"),
            ("frequency", "uint16"),
            ("digital_input_config", "uint8"),
            ("analog_channel_1_config", "uint8"),
            ("analog_channel_1_edge_detec_val", "uint16"),
            ("analog_channel_2_config", "uint8"),
            ("analog_channel_2_edge_detec_val", "uint16"),
            ("analog_channel_3_config", "uint8"),
            ("analog_channel_3_edge_detec_val", "uint16"),
            ("analog_channel_4_config", "uint8"),
            ("analog_channel_4_edge_detec_val", "uint16"),
            ("analog_channel_5_config", "uint8"),
            ("analog_channel_5_edge_detec_val", "uint16"),
            ("unused", "S6"),
        ],
    },
    "3.0": {
        # Version 3.0 uses the same structure as 2.3
        b"NEUEVWAV": [
            ("packet_id", "S8"),
            ("electrode_id", "uint16"),
            ("physical_connector", "uint8"),
            ("connector_pin", "uint8"),
            ("digitization_factor", "uint16"),
            ("energy_threshold", "uint16"),
            ("hi_threshold", "int16"),
            ("lo_threshold", "int16"),
            ("nb_sorted_units", "uint8"),
            ("bytes_per_waveform", "uint8"),
            ("spike_width", "uint16"),
            ("unused", "S8"),
        ],
        b"ARRAYNME": [("packet_id", "S8"), ("electrode_array_name", "S24")],
        b"ECOMMENT": [("packet_id", "S8"), ("extra_comment", "S24")],
        b"CCOMMENT": [("packet_id", "S8"), ("continued_comment", "S24")],
        b"MAPFILE": [("packet_id", "S8"), ("mapFile", "S24")],
        b"NEUEVLBL": [
            ("packet_id", "S8"),
            ("electrode_id", "uint16"),
            ("label", "S16"),
            ("unused", "S6"),
        ],
        b"NEUEVFLT": [
            ("packet_id", "S8"),
            ("electrode_id", "uint16"),
            ("hi_freq_corner", "uint32"),
            ("hi_freq_order", "uint32"),
            ("hi_freq_type", "uint16"),
            ("lo_freq_corner", "uint32"),
            ("lo_freq_order", "uint32"),
            ("lo_freq_type", "uint16"),
            ("unused", "S2"),
        ],
        b"DIGLABEL": [
            ("packet_id", "S8"),
            ("label", "S16"),
            ("mode", "uint8"),
            ("unused", "S7"),
        ],
        b"VIDEOSYN": [
            ("packet_id", "S8"),
            ("video_source_id", "uint16"),
            ("video_source", "S16"),
            ("frame_rate", "float32"),
            ("unused", "S2"),
        ],
        b"TRACKOBJ": [
            ("packet_id", "S8"),
            ("trackable_type", "uint16"),
            ("trackable_id", "uint16"),
            ("point_count", "uint16"),
            ("video_source", "S16"),
            ("unused", "S2"),
        ],
        b"NSASEXEV": [
            ("packet_id", "S8"),
            ("frequency", "uint16"),
            ("digital_input_config", "uint8"),
            ("analog_channel_1_config", "uint8"),
            ("analog_channel_1_edge_detec_val", "uint16"),
            ("analog_channel_2_config", "uint8"),
            ("analog_channel_2_edge_detec_val", "uint16"),
            ("analog_channel_3_config", "uint8"),
            ("analog_channel_3_edge_detec_val", "uint16"),
            ("analog_channel_4_config", "uint8"),
            ("analog_channel_4_edge_detec_val", "uint16"),
            ("analog_channel_5_config", "uint8"),
            ("analog_channel_5_edge_detec_val", "uint16"),
            ("unused", "S6"),
        ],
    },
}


# Packet identifiers for different NEV file specifications
# Used to create masks that filter raw data packets by their packet ID field.
# Single values indicate equality check, tuples (min, max) indicate range check.
# According to NEV spec: packet IDs < 32768 identify channels, IDs >= 32768 are system events.
# Packets whose ID is not listed for a spec are not read. Known cases, ignored on purpose:
# - 0x8000 + channel (e.g. 32769): preview packets of the live protocol (cbproto's "Preview Data Packet
#   Definitions"), found in 3.0 files written by Central. They carry no neural data.
# - 0xFFF9: recording events of 3.0 files written by Central (header type 0 start, 1 stop, 2 pause,
#   3 resume; Central's CNevFile30::AddRecordingEvent). Not used to split segments, which come from
#   resets and gaps.
NEV_PACKET_IDENTIFIERS_BY_SPEC = {
    "2.1": {
        "NonNeural": 0,
        "Spikes": (1, 255),  # Packet IDs in this range identify spike events on electrodes
    },
    "2.2": {
        "NonNeural": 0,
        # Packet IDs in this range identify spike events on electrodes. Wider than 2.1's range:
        # Ripple Trellis writes 2.2 files with electrode IDs above 255 (its front-end ports are
        # numbered 1-128, 129-256, 257-384...), and every spike on them was dropped without a
        # warning (NeuralEnsemble/python-neo#1918). 2048 is the range of 2.3.
        "Spikes": (1, 2048),
    },
    "2.3": {
        "NonNeural": 0,
        "Spikes": (1, 2048),  # Packet IDs in this range identify spike events on electrodes
        "Comments": 0xFFFF,
        "VideoSync": 0xFFFE,
        "TrackingEvents": 0xFFFD,
        "ButtonTrigger": 0xFFFC,
        "ConfigEvent": 0xFFFB,
    },
    "3.0": {
        "NonNeural": 0,
        "Spikes": (1, 2048),  # Packet IDs in this range identify spike events on electrodes
        "Comments": 0xFFFF,
        "VideoSync": 0xFFFE,
        "TrackingEvents": 0xFFFD,
        "ButtonTrigger": 0xFFFC,
        "ConfigEvent": 0xFFFB,
    },
}


# Data types for different NEV file specifications
# Structure: {spec: {data_type: lambda function that returns dtype definition}}
NEV_PACKET_DATA_TYPES_BY_SPEC = {
    "2.1": {
        "NonNeural": lambda packet_size_bytes: [
            ("timestamp", "uint32"),
            ("packet_id", "uint16"),
            ("packet_insertion_reason", "uint8"),
            ("reserved", "uint8"),
            ("digital_input", "uint16"),
            ("analog_input_channel_1", "int16"),
            ("analog_input_channel_2", "int16"),
            ("analog_input_channel_3", "int16"),
            ("analog_input_channel_4", "int16"),
            ("analog_input_channel_5", "int16"),
            ("unused", f"S{packet_size_bytes - 20}"),
        ],
        "Spikes": lambda packet_size_bytes: [
            ("timestamp", "uint32"),
            ("packet_id", "uint16"),
            ("unit_class_nb", "uint8"),
            ("reserved", "uint8"),
            ("waveform", f"S{packet_size_bytes - 8}"),
        ],
    },
    "2.2": {
        "NonNeural": lambda packet_size_bytes: [
            ("timestamp", "uint32"),
            ("packet_id", "uint16"),
            ("packet_insertion_reason", "uint8"),
            ("reserved", "uint8"),
            ("digital_input", "uint16"),
            ("analog_input_channel_1", "int16"),
            ("analog_input_channel_2", "int16"),
            ("analog_input_channel_3", "int16"),
            ("analog_input_channel_4", "int16"),
            ("analog_input_channel_5", "int16"),
            ("unused", f"S{packet_size_bytes - 20}"),
        ],
        "Spikes": lambda packet_size_bytes: [
            ("timestamp", "uint32"),
            ("packet_id", "uint16"),
            ("unit_class_nb", "uint8"),
            ("reserved", "uint8"),
            ("waveform", f"S{packet_size_bytes - 8}"),
        ],
    },
    "2.3": {
        "NonNeural": lambda packet_size_bytes: [
            ("timestamp", "uint32"),
            ("packet_id", "uint16"),
            ("packet_insertion_reason", "uint8"),
            ("reserved", "uint8"),
            ("digital_input", "uint16"),
            ("unused", f"S{packet_size_bytes - 10}"),
        ],
        "Spikes": lambda packet_size_bytes: [
            ("timestamp", "uint32"),
            ("packet_id", "uint16"),
            ("unit_class_nb", "uint8"),
            ("reserved", "uint8"),
            ("waveform", f"S{packet_size_bytes - 8}"),
        ],
        "Comments": lambda packet_size_bytes: [
            ("timestamp", "uint32"),
            ("packet_id", "uint16"),
            ("char_set", "uint8"),
            ("flag", "uint8"),
            ("color", "uint32"),
            ("comment", f"S{packet_size_bytes - 12}"),
        ],
        "VideoSync": lambda packet_size_bytes: [
            ("timestamp", "uint32"),
            ("packet_id", "uint16"),
            ("video_file_nb", "uint16"),
            ("video_frame_nb", "uint32"),
            ("video_elapsed_time", "uint32"),
            ("video_source_id", "uint32"),
            ("unused", "int8", (packet_size_bytes - 20,)),
        ],
        "TrackingEvents": lambda packet_size_bytes: [
            ("timestamp", "uint32"),
            ("packet_id", "uint16"),
            ("parent_id", "uint16"),
            ("node_id", "uint16"),
            ("node_count", "uint16"),
            ("point_count", "uint16"),
            ("tracking_points", "uint16", ((packet_size_bytes - 14) // 2,)),
        ],
        "ButtonTrigger": lambda packet_size_bytes: [
            ("timestamp", "uint32"),
            ("packet_id", "uint16"),
            ("trigger_type", "uint16"),
            ("unused", "int8", (packet_size_bytes - 8,)),
        ],
        "ConfigEvent": lambda packet_size_bytes: [
            ("timestamp", "uint32"),
            ("packet_id", "uint16"),
            ("config_change_type", "uint16"),
            ("config_changed", f"S{packet_size_bytes - 8}"),
        ],
    },
    "3.0": {
        "NonNeural": lambda packet_size_bytes: [
            ("timestamp", "uint64"),
            ("packet_id", "uint16"),
            ("packet_insertion_reason", "uint8"),
            ("dlen", "uint8"),
            ("digital_input", "uint16"),
            ("unused", f"S{packet_size_bytes - 14}"),
        ],
        "Spikes": lambda packet_size_bytes: [
            ("timestamp", "uint64"),
            ("packet_id", "uint16"),
            ("unit_class_nb", "uint8"),
            ("dlen", "uint8"),
            ("waveform", f"S{packet_size_bytes - 12}"),
        ],
        "Comments": lambda packet_size_bytes: [
            ("timestamp", "uint64"),
            ("packet_id", "uint16"),
            ("char_set", "uint8"),
            ("flag", "uint8"),
            ("color", "uint32"),
            ("comment", f"S{packet_size_bytes - 16}"),
        ],
        "VideoSync": lambda packet_size_bytes: [
            ("timestamp", "uint64"),
            ("packet_id", "uint16"),
            ("video_file_nb", "uint16"),
            ("video_frame_nb", "uint32"),
            ("video_elapsed_time", "uint32"),
            ("video_source_id", "uint32"),
            ("unused", "int8", (packet_size_bytes - 24,)),
        ],
        "TrackingEvents": lambda packet_size_bytes: [
            ("timestamp", "uint64"),
            ("packet_id", "uint16"),
            ("parent_id", "uint16"),
            ("node_id", "uint16"),
            ("node_count", "uint16"),
            ("point_count", "uint16"),
            ("tracking_points", "uint16", ((packet_size_bytes - 18) // 2,)),
        ],
        "ButtonTrigger": lambda packet_size_bytes: [
            ("timestamp", "uint64"),
            ("packet_id", "uint16"),
            ("trigger_type", "uint16"),
            ("unused", "int8", (packet_size_bytes - 12,)),
        ],
        "ConfigEvent": lambda packet_size_bytes: [
            ("timestamp", "uint64"),
            ("packet_id", "uint16"),
            ("config_change_type", "uint16"),
            ("config_changed", f"S{packet_size_bytes - 12}"),
        ],
    },
}


# Basic header types for different NSX file specifications
NSX_BASIC_HEADER_TYPES = {
    "2.1": [
        ("file_id", "S8"),
        ("label", "S16"),
        ("period", "uint32"),
        ("channel_count", "uint32"),
    ],
    "2.2": [
        ("file_id", "S8"),
        ("ver_major", "uint8"),
        ("ver_minor", "uint8"),
        ("bytes_in_headers", "uint32"),
        ("label", "S16"),
        ("comment", "S256"),
        ("period", "uint32"),
        ("timestamp_resolution", "uint32"),
        ("year", "uint16"),
        ("month", "uint16"),
        ("weekday", "uint16"),
        ("day", "uint16"),
        ("hour", "uint16"),
        ("minute", "uint16"),
        ("second", "uint16"),
        ("millisecond", "uint16"),
        ("channel_count", "uint32"),
    ],
    "2.3": [
        ("file_id", "S8"),
        ("ver_major", "uint8"),
        ("ver_minor", "uint8"),
        ("bytes_in_headers", "uint32"),
        ("label", "S16"),
        ("comment", "S256"),
        ("period", "uint32"),
        ("timestamp_resolution", "uint32"),
        ("year", "uint16"),
        ("month", "uint16"),
        ("weekday", "uint16"),
        ("day", "uint16"),
        ("hour", "uint16"),
        ("minute", "uint16"),
        ("second", "uint16"),
        ("millisecond", "uint16"),
        ("channel_count", "uint32"),
    ],
    "3.0": [
        ("file_id", "S8"),
        ("ver_major", "uint8"),
        ("ver_minor", "uint8"),
        ("bytes_in_headers", "uint32"),
        ("label", "S16"),
        ("comment", "S256"),
        ("period", "uint32"),
        ("timestamp_resolution", "uint32"),
        ("year", "uint16"),
        ("month", "uint16"),
        ("weekday", "uint16"),
        ("day", "uint16"),
        ("hour", "uint16"),
        ("minute", "uint16"),
        ("second", "uint16"),
        ("millisecond", "uint16"),
        ("channel_count", "uint32"),
    ],
}


# Extended header types for different NSX file specifications
NSX_EXT_HEADER_TYPES = {
    "2.1": [
        ("electrode_id", "uint32"),
    ],
    "2.2": [
        ("type", "S2"),
        ("electrode_id", "uint16"),
        ("electrode_label", "S16"),
        ("physical_connector", "uint8"),
        ("connector_pin", "uint8"),
        ("min_digital_val", "int16"),
        ("max_digital_val", "int16"),
        ("min_analog_val", "int16"),
        ("max_analog_val", "int16"),
        ("units", "S16"),
        ("hi_freq_corner", "uint32"),
        ("hi_freq_order", "uint32"),
        ("hi_freq_type", "uint16"),
        ("lo_freq_corner", "uint32"),
        ("lo_freq_order", "uint32"),
        ("lo_freq_type", "uint16"),
    ],
    "2.3": [
        ("type", "S2"),
        ("electrode_id", "uint16"),
        ("electrode_label", "S16"),
        ("physical_connector", "uint8"),
        ("connector_pin", "uint8"),
        ("min_digital_val", "int16"),
        ("max_digital_val", "int16"),
        ("min_analog_val", "int16"),
        ("max_analog_val", "int16"),
        ("units", "S16"),
        ("hi_freq_corner", "uint32"),
        ("hi_freq_order", "uint32"),
        ("hi_freq_type", "uint16"),
        ("lo_freq_corner", "uint32"),
        ("lo_freq_order", "uint32"),
        ("lo_freq_type", "uint16"),
    ],
    "3.0": [
        ("type", "S2"),
        ("electrode_id", "uint16"),
        ("electrode_label", "S16"),
        ("physical_connector", "uint8"),
        ("connector_pin", "uint8"),
        ("min_digital_val", "int16"),
        ("max_digital_val", "int16"),
        ("min_analog_val", "int16"),
        ("max_analog_val", "int16"),
        ("units", "S16"),
        ("hi_freq_corner", "uint32"),
        ("hi_freq_order", "uint32"),
        ("hi_freq_type", "uint16"),
        ("lo_freq_corner", "uint32"),
        ("lo_freq_order", "uint32"),
        ("lo_freq_type", "uint16"),
    ],
}

# NSX Data Header Types by specification version
# These define the structure of data block headers within NSX files
NSX_DATA_HEADER_TYPES = {
    # Version 2.1 has no data headers - data is stored continuously after the main header
    "2.1": None,
    # Versions 2.2+ use data block headers with timestamp size based on major version
    "2.2": [("header_flag", "uint8"), ("timestamp", "uint32"), ("nb_data_points", "uint32")],
    "2.3": [("header_flag", "uint8"), ("timestamp", "uint32"), ("nb_data_points", "uint32")],
    "3.0": [("header_flag", "uint8"), ("timestamp", "uint64"), ("nb_data_points", "uint32")],
    # PTP variant has a completely different structure with samples embedded
    "3.0-ptp": lambda channel_count: [
        ("reserved", "uint8"),
        ("timestamps", "uint64"),
        ("num_data_points", "uint32"),
        ("samples", "int16", (channel_count,)),
    ],
}
