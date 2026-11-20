"""Model-conditioned coordinate conversion, separate from both reconstruction solvers.

Forward: legacy twelve angles -> legacy predicted markers -> native technical
frames -> native seven-coordinate projection. Missing legacy marker bindings
stay missing. This is a common-marker model projection, not anatomical truth or
an exact conversion between the independently estimated motion trajectories.
"""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from ..legacy_g99.model import Model, Q_NAMES as LEGACY_NAMES
from ..legacy_g99.kinematics import forward as legacy_forward
from ..tum_native.observations import track
from ..tum_native.model import decompose
from ..tum_native.schema import SEGMENTS, Q_NAMES


class AngleConversion:
    def __init__(self, legacy_model, native_model):
        self.legacy = legacy_model
        self.native = native_model
        self.names = list(dict.fromkeys(name for segment in SEGMENTS
            for name in native_model['segments'][segment]['fit_names']))
        self.predicted_names = list(legacy_model.bindings)
        self.common = [name for name in self.names if name in legacy_model.bindings]
        self.common_output_ids = [self.predicted_names.index(name) for name in self.common]
        self.common_input_ids = [self.names.index(name) for name in self.common]
        self.support = {}
        for segment in SEGMENTS:
            names = native_model['segments'][segment]['fit_names']
            shared = [name for name in names if name in legacy_model.bindings]
            if len(shared) < 3:
                raise ValueError(f'{segment}: fewer than three shared markers; conversion unsupported')
            self.support[segment] = dict(shared=shared, missing=[n for n in names if n not in shared],
                legacy_segments={n: legacy_model.bindings[n]['segment'] for n in shared},
                native_marker_count=len(names))

    @classmethod
    def load(cls, legacy_path, native_path):
        return cls(Model.load(legacy_path, allow_estimated=True),
                   json.loads(Path(native_path).read_text(encoding='utf-8')))

    def frames(self, old_q):
        q = np.atleast_2d(np.asarray(old_q, float))
        if q.shape[1] != 12 or not np.isfinite(q).all():
            raise ValueError('Expected finite legacy q[N,12] in radians')
        predicted = legacy_forward(self.legacy, q)[0]
        points = np.full((len(q), len(self.names), 3), np.nan)
        points[:, self.common_input_ids] = predicted[:, self.common_output_ids]
        mask = np.isfinite(points).all(2)
        observation = dict(t=np.arange(len(q), dtype=float), p=points, names=self.names,
            edge=np.zeros(mask.shape, bool), interpolated=np.zeros(mask.shape, bool))
        return track(observation, self.native)

    def forward(self, old_q, diagnostics=False):
        poses = self.frames(old_q)
        q, displacement, error = decompose(poses, self.native)
        if diagnostics:
            return q, dict(segment_R=poses['R'], cluster_rms_m=poses['rms_m'],
                local_projection_error_rad=error, shoulder_displacement_T_m=displacement,
                marker_counts=poses['counts'], degenerate=poses['degenerate'])
        return q

    def linearize(self, reference, step=1e-4):
        """Local d(native q)/d(legacy q); all columns have units rad/rad."""
        reference = np.asarray(reference, float)
        values = [reference]
        for k in range(12):
            offset = np.eye(12)[k] * step
            values.extend([reference+offset, reference-offset])
        mapped = self.forward(np.array(values))
        jacobian = np.column_stack([(mapped[1+2*k]-mapped[2+2*k])/(2*step) for k in range(12)])
        if not np.isfinite(jacobian).all():
            raise ValueError('Degenerate common-marker geometry at reference')
        return mapped[0], jacobian

    def inverse_conditional(self, native_q, upstream, initial, max_nfev=150):
        """Local inverse after fixing legacy q1..q5; does not recover those angles.

        Success means coordinate projection is matched on this branch, not that
        full segment orientations, joint positions or marker geometry match.
        """
        target, upstream, initial = [np.asarray(v, float) for v in [native_q, upstream, initial]]
        if target.shape != (7,) or upstream.shape != (5,) or initial.shape != (7,):
            raise ValueError('Expected native[7], fixed upstream[5], initial legacy last-seven[7]')
        if not all(np.isfinite(v).all() for v in [target, upstream, initial]):
            raise ValueError('Inverse arguments must be finite')
        target_rotation = Rotation.from_rotvec(target[:3]).as_matrix()
        def residual(x):
            value = self.forward(np.r_[upstream, x])[0]
            if not np.isfinite(value).all():
                raise ValueError('Degenerate common-marker geometry during inverse')
            orientation = Rotation.from_matrix(Rotation.from_rotvec(value[:3]).as_matrix().T @ target_rotation).as_rotvec()
            scalar = np.arctan2(np.sin(value[3:]-target[3:]), np.cos(value[3:]-target[3:]))
            return np.r_[orientation, scalar]
        def jacobian(x):
            # The inner two-axis projection has a finite numerical tolerance;
            # scipy's default sqrt(eps) differences can report a spurious rank loss.
            step = 1e-4
            return np.column_stack([(residual(x+np.eye(7)[k]*step)-residual(x-np.eye(7)[k]*step))/(2*step)
                                    for k in range(7)])
        lower, upper = self.legacy.lower[5:], self.legacy.upper[5:]
        fit = least_squares(residual, np.clip(initial, lower+1e-10, upper-1e-10), bounds=(lower, upper),
                            jac=jacobian, max_nfev=max_nfev, xtol=1e-10, ftol=1e-10, gtol=1e-10)
        singular = np.linalg.svd(jacobian(fit.x), compute_uv=False)
        residual_deg = np.rad2deg(residual(fit.x))
        return fit.x, dict(converged=bool(fit.success), nfev=int(fit.nfev),
            maximum_coordinate_residual_deg=float(np.max(abs(residual_deg))),
            coordinate_match=bool(np.max(abs(residual_deg)) < .01),
            local_rank=int(np.sum(singular > singular[0]*1e-6)),
            local_condition_number=float(singular[0]/max(singular[-1], 1e-15)),
            active_bounds=fit.active_mask.tolist(), upstream_fixed=True,
            global_uniqueness_established=False, full_geometry_equivalence_established=False)
