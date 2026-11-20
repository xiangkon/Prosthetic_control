"""Numerical checks for the two-entry-point reconstruction workflow."""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from .calibration import functional_elbow
from .data_io import MotionData, load_motion, kabsch
from ..common.emg_sync import infer_pulse_alignment, resample_estimates
from .filtering import trajectory_ekf
from .kinematics import forward, contact, solve_trajectory
from .model import Model, SEGMENTS, finite_difference
from .quality import diagnostics
from .reference_geometry import compile_reference, audit_reference


def run_checks(data_root):
    reference = compile_reference()
    reference_model = Model(reference, allow_reference=True, require_bindings=False)
    assert audit_reference(reference_model)['passed']
    configuration = dict(reference)
    configuration.update(status='synthetic_demo', lower_rad=[-3.14]*12, upper_rad=[3.14]*12,
                         initial_q_rad=[0.0]*12, scapulothoracic=None,
                         calibration={'prior_dominated_angles': [True]*8+[False]*4})
    offsets = [[.04,0,0],[0,.05,0],[0,0,-.07],[-.03,.02,-.04]]
    configuration['markers'] = {
        f'{segment}_{index}': {'segment':segment,'local_m':point,'sigma_m':.004}
        for segment in SEGMENTS for index,point in enumerate(offsets)
    }
    model = Model(configuration, allow_demo=True)
    t = np.arange(8)*.02
    q = .1 + .03*np.sin(t[:,None]*4+np.arange(12)[None])
    poses = np.repeat(np.eye(4)[None],len(t),axis=0)
    prediction, jacobian, _, _, _ = forward(model,q,poses)
    names = list(model.bindings)
    for index in range(len(t)):
        np.testing.assert_allclose(prediction[index],model.predict(q[index],names,poses[index]),atol=1e-10)
    numerical = finite_difference(lambda angles:model.predict(angles,names,poses[0]),q[0])
    np.testing.assert_allclose(jacobian[0].reshape(-1,12),numerical,atol=2e-8)
    _, _, frames, axes, centers = forward(reference_model,q[:1])
    residual, derivative = contact(reference_model,frames,axes,centers)
    np.testing.assert_allclose(residual[0],reference_model.constraints(q[0]),atol=1e-12)
    np.testing.assert_allclose(derivative[0],finite_difference(reference_model.constraints,q[0]),atol=2e-8)
    motion = MotionData(t,prediction,names)
    recovered,_ = solve_trajectory(model,prediction,motion.valid,poses,q+.04,iterations=60)
    assert np.sqrt(np.mean((recovered-q)**2)) < .003
    filtered,sd,clipped = trajectory_ekf(motion,model,poses,q)
    assert np.isfinite(sd).all() and np.sqrt(np.mean((filtered-q)**2)) < .02
    flags = diagnostics(motion,model,poses,filtered,clipped=clipped)['angle_quality_flag']
    assert np.all(flags[:,:8]&64) and not np.any(flags[:,8:]&64)
    transform = kabsch(np.array(offsets),np.array(offsets)+[.2,.3,.4])
    np.testing.assert_allclose(transform[:3,3],[.2,.3,.4],atol=1e-12)

    theta = np.linspace(-.8,1.1,40)
    upper = np.repeat(np.eye(4)[None],len(theta),axis=0)
    forearm = upper.copy()
    forearm[:,:3,:3] = Rotation.from_rotvec(theta[:,None]*[1,0,0]).as_matrix()
    expected_center = np.tile([.02,.01,-.25],(len(theta),1))
    forearm[:,:3,3] = expected_center - np.einsum('nij,j->ni',forearm[:,:3,:3],[-.01,.02,.03])
    estimated,_,report = functional_elbow(upper,forearm,expected_center+[.025,0,0],upper)
    np.testing.assert_allclose(estimated,expected_center,atol=1e-8)
    assert report['sufficient_excitation']

    clock = np.arange(10000)*.0005
    pulse = np.full(len(clock),255.)
    pulse[2000:5999] = 254
    alignment = infer_pulse_alignment(clock,pulse,np.arange(1000)*.002)
    assert abs(alignment['offset_s']-1) < 1e-10
    values,quality = resample_estimates(np.array([0.,.01,.02]),np.ones((3,4)),
        np.array([[1]*4,[0]*4,[64]*4],np.uint16),np.array([-.01,.005,.01,.015,.03]),.02)
    np.testing.assert_array_equal(quality[:,0],[1,1,0,64,1])
    assert np.isnan(values[0]).all() and np.isnan(values[-1]).all()
    sample = data_root/'subj_001/MoCap/trial_001.mat'
    source_checked = sample.exists()
    if source_checked:
        raw = load_motion(sample)
        assert len(raw.names)==38 and np.isclose(np.median(np.diff(raw.t)),.002)
    print(json.dumps({'numerical_checks':'passed','real_mat_checked':source_checked}))
