"""Guarded decoding of the TUM MATLAB motion-capture object layout."""
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import numpy as np
from scipy.io import loadmat


@dataclass
class Motion:
    points: np.ndarray  # frames, markers, xyz; millimetres
    time: np.ndarray  # seconds
    names: list
    source: str

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=float)
        self.time = np.asarray(self.time, dtype=float).reshape(-1)
        self.names = [str(n) for n in self.names]
        if self.points.ndim != 3 or self.points.shape[2] != 3:
            raise ValueError('Expected coordinates shaped (frames, markers, 3).')
        if not len(self.time) or self.points.shape[:2] != (len(self.time), len(self.names)):
            raise ValueError('Coordinate, time and marker-name dimensions do not match.')
        if not np.isfinite(self.time).all() or np.any(np.diff(self.time) <= 0):
            raise ValueError('Time must be finite and strictly increasing.')
        self.time = self.time - self.time[0]
        if not np.isfinite(self.points).all(axis=2).any():
            raise ValueError('No complete XYZ position is available in this file.')
        # A partial XYZ triplet cannot be plotted as a 3-D position.
        self.points[~np.isfinite(self.points).all(axis=2)] = np.nan




def _tum_workspace(payload):
    """Read the embedded MAT stream; private SciPy API is isolated here."""
    from scipy.io.matlab._mio5 import MatFile5Reader
    raw = np.asarray(payload, dtype=np.uint8).tobytes()
    if len(raw) < 16 or raw[:4] != b'\x00\x01IM':
        raise ValueError('Unsupported TUM MCOS byte order or workspace version.')
    stream = BytesIO(raw)
    reader = MatFile5Reader(stream, byte_order='<', struct_as_record=False, squeeze_me=True)
    reader.initialize_read()
    stream.seek(8)
    header, _ = reader.read_var_header()
    wrapper = reader.read_var_array(header, process=False).flat[0].MCOS
    if wrapper['s2'][0] != b'FileWrapper__':
        raise ValueError('Expected a TUM MCOS FileWrapper object.')
    slots = wrapper['arr'][0].ravel()
    if np.asarray(slots[0], dtype=np.uint8).tobytes()[:4] != b'\x02\x00\x00\x00':
        raise ValueError('Unsupported TUM MCOS property-table version.')
    return slots


def read_tum(path):
    """Decode the guarded 19-slot timeseries layout used by local TUM files.

    This is deliberately not a general MATLAB MCOS decoder. No shape-based
    guessing is used: object IDs, class IDs, property links, lengths and time
    arrays must all match the supported layout before any motion is returned.
    """
    mat = loadmat(path, simplify_cells=True)
    markers = mat.get('marker')
    if markers is None:
        raise ValueError('MAT file has no marker variable; select a TUM MoCap file.')
    if isinstance(markers, dict):
        markers = [markers]
    elif isinstance(markers, np.ndarray):
        markers = list(markers.ravel())
    if not isinstance(markers, list) or not markers:
        raise ValueError('Unsupported marker structure in MAT file.')
    names, arrays, times = [], [], []
    opaque = [type(m.get('ts')).__name__ == 'MatlabOpaque' for m in markers]
    if any(opaque) and not all(opaque):
        raise ValueError('Mixed opaque and numeric marker structures are unsupported.')
    try:
        slots = _tum_workspace(mat['__function_workspace__']) if all(opaque) else None
        if slots is not None and len(slots) != 19 * len(markers) + 3:
            raise ValueError('Unexpected number of MCOS property slots.')
        for index, marker in enumerate(markers):
            ts = marker['ts']
            names.append(str(marker['name']))
            if slots is None:
                data, time = ts['Data'], ts['Time']
            else:
                handle = ts.ravel()[0]
                expected = [3707764736, 2, 1, 1, 5 * index + 1, 5]
                if handle['s2'] != b'timeseries' or not np.array_equal(handle['arr'].ravel(), expected):
                    raise ValueError(f'Unexpected timeseries handle for {names[-1]}.')
                b = 19 * index
                for offset, obj, cls in [(5, 3, 1), (7, 2, 2), (15, 4, 3), (19, 5, 4)]:
                    expected_ref = [3707764736, 2, 1, 1, 5 * index + obj, cls]
                    if not np.array_equal(np.asarray(slots[b + offset]).ravel(), expected_ref):
                        raise ValueError(f'Unexpected MCOS property link for {names[-1]}.')
                data, time = slots[b + 20], slots[b + 11]
                if int(slots[b + 9]) != len(np.atleast_1d(time)):
                    raise ValueError(f'Incorrect timeseries length for {names[-1]}.')
            data = np.asarray(data, dtype=float)
            time = np.asarray(time, dtype=float).reshape(-1)
            if data.ndim != 2 or data.shape != (len(time), 3):
                raise ValueError(f'{names[-1]} does not contain an N-by-3 coordinate array.')
            arrays.append(data)
            times.append(time)
        for time in times[1:]:
            if time.shape != times[0].shape or not np.allclose(time, times[0], rtol=0, atol=1e-9):
                raise ValueError('Markers have different time axes; synchronize before plotting.')
    except (KeyError, AttributeError, IndexError, TypeError) as exc:
        raise ValueError('Unsupported TUM timeseries layout. Export marker.ts.Data and '
                         'marker.ts.Time as numeric arrays in MATLAB, or use NPZ.') from exc
    return Motion(np.stack(arrays, axis=1), times[0], names, str(path))
