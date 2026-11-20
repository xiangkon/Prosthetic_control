"""One-command estimated 12-angle pipeline from original TUM files.

Automatic calibration does not establish external angle truth. Prior-dependent
coordinates, weak information, failed fits and empirical synchronization remain
explicit in every export. No files in datasets/ are changed.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.io import loadmat
from .calibration import estimate_subject
from .data_io import load_motion,thorax_poses,MotionData
from .model import Model,Q_NAMES
from .filtering import trajectory_ik,trajectory_ekf
from .quality import diagnostics
from .kinematics import solve_trajectory
from ..common.emg_sync import read_emg_clock,infer_pulse_alignment,resample_estimates
from .marker_analysis import write_json,quantile


def jobs(subject_dir):
    rows=loadmat(Path(subject_dir)/'trial_info.mat',simplify_cells=True)['trial_info']
    out=[]
    for task,row in enumerate(rows):
        emg={int(v) for v in np.atleast_1d(row['EMG']).ravel()}
        for trial in np.atleast_1d(row['Kinematics']).ravel():
            out.append(dict(task=int(task),trial=int(trial),description=str(row['Description']),
                            paired_emg=int(trial) in emg))
    return out


def code_hash():
    digest=hashlib.sha256()
    for name in ['pipeline.py','runner.py','calibration.py','kinematics.py','filtering.py',
                 'quality.py','model.py','data_io.py',
                 'reference_geometry.py','marker_analysis.py']:
        digest.update((Path(__file__).parent/name).read_bytes())
    for name in ['motion_io.py', '_matlab.py', 'emg_sync.py']:
        digest.update((Path(__file__).parents[1] / 'common' / name).read_bytes())
    digest.update((Path(__file__).parents[1] / 'paths.py').read_bytes())
    from ..paths import REFERENCE_TABLES
    digest.update(REFERENCE_TABLES.read_bytes())
    return digest.hexdigest()


def validate_subject(subject_dir,model):
    selected=[j for j in jobs(subject_dir) if j['task'] in [0,2,3,4,5,7,8,12,20,28]]
    first={}
    for j in jobs(subject_dir): first.setdefault(j['task'],j['trial'])
    selected=[j for j in selected if j['trial']!=first[j['task']]]
    ps=[]; ts=[]; records=[]; names=None
    for j in selected:
        d=load_motion(Path(subject_dir)/'MoCap'/f'trial_{j["trial"]:03d}.mat')
        poses,_=thorax_poses(d,model)
        for k in np.unique(np.linspace(0,len(d.t)-1,4).astype(int)):
            ps.append(d.p[k]); ts.append(poses[k]); records.append(dict(**j,frame=int(k),time=float(d.t[k])))
        names=d.names
    data=MotionData(np.arange(len(ps),dtype=float),np.array(ps),names)
    poses=np.array(ts)
    q,details=trajectory_ik(data,model,poses,iterations=45)
    diag=diagnostics(data,model,poses,q)
    p90=quantile(diag['marker_rms_m'],.9)
    # Conditional multi-start check on six poses; calibration gauges are separately flagged.
    idx=np.unique(np.linspace(0,len(q)-1,min(6,len(q))).astype(int))
    tiny=MotionData(np.arange(len(idx),dtype=float),data.p[idx],data.names)
    fits=[]; errors=[]
    rng=np.random.default_rng(81)
    for k in range(3):
        initial=np.clip(q[idx]+rng.normal(0,.25,(len(idx),12)),model.lower,model.upper)
        fit,_=trajectory_ik(tiny,model,poses[idx],initial,iterations=50)
        fit_diag=diagnostics(tiny,model,poses[idx],fit)
        fits.append(fit); errors.append(fit_diag['marker_rms_m'])
    fits=np.array(fits); errors=np.array(errors)
    stable_range=np.full(12,np.nan)
    variations=[]
    for i in range(len(idx)):
        good=errors[:,i]<=np.nanmin(errors[:,i])+.002
        if good.sum()>=2: variations.append(np.ptp(fits[good,i],axis=0))
    if variations: stable_range=np.nanmedian(variations,axis=0)
    prior=np.array(model.config['calibration']['prior_dominated_angles'])|(~np.isfinite(stable_range))|(stable_range>np.deg2rad(5))
    model.config['calibration']['prior_dominated_angles']=prior.tolist()
    loo=[]
    from .filtering import prepare_observations
    measured,valid=prepare_observations(tiny,model)
    for k,name in enumerate(model.bindings):
        if k%4: continue
        masked=valid.copy(); masked[:,k]=False
        qq,_=solve_trajectory(model,measured,masked,poses[idx],q[idx],iterations=30)
        from .kinematics import forward
        pred,_,_,_,_=forward(model,qq,poses[idx])
        e=np.linalg.norm(pred[:,k]-measured[:,k],axis=1)
        loo.append(dict(marker=name,median_prediction_error_m=quantile(e,.5),p90_prediction_error_m=quantile(e,.9)))
    report=dict(validation_scope='geometry-fit heldout repetitions; rigidity screening previously saw these repeats; not external angle truth',
        sampled_frames=len(q),records=records,marker_rms_median_m=quantile(diag['marker_rms_m'],.5),
        marker_rms_p90_m=p90,contact_residual_abs_p90=quantile(abs(diag['contact_residual']),.9),
        ik_converged_fraction=float(details['converged'].mean()),
        conditional_multistart_median_range_deg=np.rad2deg(stable_range).tolist(),
        marker_leaveout=loo,prior_dominated_angles=prior.tolist(),
        geometry_failed=bool(not np.isfinite(p90) or p90>.02 or not model.config['calibration']['optimizer_success']))
    model.config['calibration_validation']=report
    return report


def summary_metrics(q,diag):
    return dict(frames=len(q),finite_frames=int(np.isfinite(q).all(1).sum()),
        marker_rms_median_mm=quantile(diag['marker_rms_m']*1000,.5),
        marker_rms_p90_mm=quantile(diag['marker_rms_m']*1000,.9),
        contact_abs_p90=quantile(abs(diag['contact_residual']),.9),
        per_angle_clean_fraction=np.mean(diag['angle_quality_flag']==0,axis=0).tolist(),
        per_angle_weak_fraction=np.mean((diag['angle_quality_flag']&8)!=0,axis=0).tolist(),
        joint_range_deg=np.rad2deg(np.stack([np.nanmin(q,axis=0),np.nanmax(q,axis=0)],axis=1)).tolist())


def save_method(out,method,data,poses,torso_rms,q,diag,sd=None):
    payload=dict(t=data.t,q=q.astype(np.float32),q_distal=q[:,8:].astype(np.float32),
        q_names=np.array(Q_NAMES),angle_unit=np.array('rad'),time_unit=np.array('s'),
        T_world_thorax=poses.astype(np.float32),torso_rms_m=torso_rms.astype(np.float32),
        angle_quality_flag=diag['angle_quality_flag'],quality_flag=diag['quality_flag'],
        marker_rms_m=diag['marker_rms_m'].astype(np.float32),
        contact_residual=diag['contact_residual'].astype(np.float32),marker_rank=diag['marker_rank'].astype(np.int8),
        marker_singular_values=diag['marker_singular_values'].astype(np.float32),
        measurement_only_sd_rad=diag['measurement_only_sd_rad'].astype(np.float32),
        prior_dominated=diag['prior_dominated'])
    if sd is not None: payload['conditional_filter_sd_rad']=sd.astype(np.float32)
    np.savez_compressed(out/f'{method}_angles.npz',**payload)


def run_trial(subject_dir,job,model,out,iterations=35,rate_hz=500):
    source=Path(subject_dir)/'MoCap'/f'trial_{job["trial"]:03d}.mat'
    original=load_motion(source)
    raw_rate=1/np.median(np.diff(original.t)); stride=max(1,int(round(raw_rate/rate_hz)))
    data=original.select(stride); poses,torso_rms=thorax_poses(data,model)
    q,details=trajectory_ik(data,model,poses,iterations=iterations)
    if not np.isfinite(q).all(1).any(): raise RuntimeError('No finite angles: torso tracking unavailable')
    di=diagnostics(data,model,poses,q,details['converged'])
    filtered,sd,clipped=trajectory_ekf(data,model,poses,q)
    de=diagnostics(data,model,poses,filtered,clipped=clipped)
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    save_method(out,'ik',data,poses,torso_rms,q,di)
    save_method(out,'ekf',data,poses,torso_rms,filtered,de,sd)
    sync=dict(status='not_available')
    if job['paired_emg']:
        emg=Path(subject_dir)/'EMG'/f'trial_{job["trial"]:03d}.mat'
        try:
            emg_t,pulse,clock=read_emg_clock(emg)
            sync=infer_pulse_alignment(emg_t,pulse,original.t)
            mapped=emg_t-sync['offset_s']
            estimates,flags=resample_estimates(data.t,filtered[:,8:],de['angle_quality_flag'][:,8:],mapped,
                                                max_gap_s=max(.01,2.1*stride/raw_rate))
            np.savez_compressed(out/'emg_aligned_distal.npz',emg_time=emg_t,
                q_distal_estimated=estimates,angle_quality_flag=flags,angle_unit=np.array('rad'),
                strict_valid=flags==0,sync_metadata=np.array(json.dumps(sync)),
                source_emg=np.array(str(emg)),unused_bipolar_channels_1based=np.array([19,25,26]))
            sync['clock_metadata']=clock
        except (ValueError,OSError,KeyError,TypeError) as exc:
            sync=dict(status='unresolved',error=str(exc))
    report=dict(**job,source=str(source),source_frames=len(original.t),output_frames=len(data.t),
        source_rate_hz=float(raw_rate),output_rate_hz=float(raw_rate/stride),requested_rate_hz=rate_hz,solver_iterations=iterations,
        ik=summary_metrics(q,di),ekf=summary_metrics(filtered,de),sync=sync,
        uncertainty='Model-dependent estimates; conditional filter SD excludes calibration and torso uncertainty',
        quality_bits={'1':'missing','2':'IK not converged','4':'position RMS >20mm','8':'weak marker information',
            '16':'contact residual','32':'bound/clipping','64':'prior-dependent absolute coordinate','128':'calibration validation failed'},
        model_file='../../estimated_model.json',model_sha256=hashlib.sha256(json.dumps(model.config,sort_keys=True).encode()).hexdigest(),
        code_sha256=code_hash(),status='computed_estimates')
    write_json(out/'report.json',report)
    return report
