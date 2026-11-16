"""Per-angle quality and conditional observability diagnostics."""
import numpy as np
from .kinematics import residual_jacobian
from .filtering import prepare_observations


def diagnostics(data,model,poses,q,converged=None,clipped=None):
    p,valid=prepare_observations(data,model)
    finite=np.isfinite(q).all(1)&np.isfinite(poses).all((1,2))
    safe_pose=poses.copy(); safe_pose[~finite]=np.eye(4)
    safe_q=q.copy(); safe_q[~finite]=model.q0
    r,J,pred=residual_jacobian(model,safe_q,safe_pose,p,valid,.05)
    M=len(model.bindings); marker_jac=J[:,:M*3]
    _,singular,V=np.linalg.svd(marker_jac,full_matrices=False)
    rank=(singular>np.maximum(singular[:,:1]*1e-5,1e-8)).sum(1)
    inv=np.where(singular>1e-8,1/np.maximum(singular,1e-8)**2,1e8)
    conditional_sd=np.sqrt(np.einsum('nki,nk->ni',V**2,inv))
    weak=conditional_sd>.35
    prior=np.array(model.config['calibration']['prior_dominated_angles'],bool)
    flags=np.zeros((len(q),12),np.uint16)
    flags[weak]|=8; flags[:,prior]|=64
    missing=(valid.sum(1)<4)|~finite; flags[missing]|=1
    count=valid.sum(1)
    error=np.where(valid,np.linalg.norm(pred-np.nan_to_num(p),axis=2),np.nan)
    rms=np.sqrt(np.nansum(error**2,axis=1)/np.maximum(count,1))
    flags[rms>.02]|=4
    contact_values=r[:,M*3:]*.05
    if contact_values.shape[1]: flags[np.max(abs(contact_values),axis=1)>.05]|=16
    boundary=np.any((safe_q-model.lower<1e-4)|(model.upper-safe_q<1e-4),axis=1)
    if clipped is not None: boundary|=clipped
    flags[boundary]|=32
    if converged is not None: flags[~converged]|=2
    if model.config.get('calibration_validation',{}).get('geometry_failed',False): flags|=128
    return dict(angle_quality_flag=flags,quality_flag=np.bitwise_or.reduce(flags,axis=1),
        marker_rms_m=rms,contact_residual=contact_values,marker_rank=rank,
        marker_singular_values=singular,measurement_only_sd_rad=conditional_sd,
        marker_error_m=error,prior_dominated=prior,missing=missing)
