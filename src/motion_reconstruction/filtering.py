"""Fast full-rate IK, IK-linearized raw-marker EKF, and per-angle diagnostics."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from .kinematics import forward,residual_jacobian,solve_trajectory


def prepare_observations(data,model):
    names=list(model.bindings)
    p=np.full((len(data.t),len(names),3),np.nan)
    valid=np.zeros((len(data.t),len(names)),bool)
    for k,name in enumerate(names):
        if name in data.names:
            i=data.names.index(name); p[:,k]=data.p[:,i]; valid[:,k]=data.valid[:,i]
    return p,valid


def trajectory_ik(data,model,poses,initial=None,iterations=35):
    p,valid=prepare_observations(data,model)
    pose_ok=np.isfinite(poses).all((1,2))
    safe=poses.copy(); safe[~pose_ok]=np.eye(4); valid[~pose_ok]=False
    initial=np.repeat(model.q0[None],len(data.t),axis=0) if initial is None else initial
    q,details=solve_trajectory(model,p,valid,safe,initial,iterations=iterations,prior_sigma=2)
    q[~pose_ok]=np.nan
    return q,details


def trajectory_ekf(data,model,poses,ik):
    """One linearization per sample at independent IK, then a 24D information update.

    Observations remain raw marker positions and soft ST residuals, not IK-angle
    pseudo-observations. Fixed linearization is recorded as an approximation.
    """
    measured,valid=prepare_observations(data,model); n=len(data.t)
    pose_ok=np.isfinite(poses).all((1,2))
    safe=poses.copy(); safe[~pose_ok]=np.eye(4); valid[~pose_ok]=False
    linear=ik.copy(); linear[~np.isfinite(linear).all(1)]=model.q0
    r,J,_=residual_jacobian(model,linear,safe,measured,valid,.05)
    output=np.full((n,12),np.nan); sd=np.full((n,12),np.nan); clipped=np.zeros(n,bool)
    x=None; P=None
    for i in range(n):
        observed=pose_ok[i] and valid[i].sum()>=4 and np.isfinite(ik[i]).all()
        if x is None:
            if not observed: continue
            x=np.r_[ik[i],np.zeros(12)]; P=np.diag(np.r_[np.full(12,.1**2),np.ones(12)])
        else:
            dt=data.t[i]-data.t[i-1]
            F=np.eye(24); F[:12,12:]=np.eye(12)*dt
            Q=2.*np.kron(np.array([[dt**3/3,dt**2/2],[dt**2/2,dt]]),np.eye(12))
            x=F@x; P=F@P@F.T+Q
            if observed:
                # Model residual is h(q_ik)-z. Linearize it at the solved q_ik.
                residual=-r[i]-J[i]@(x[:12]-linear[i])
                weights=1/np.sqrt(1+residual**2)
                A=J[i].T@(J[i]*weights[:,None])
                b=J[i].T@(weights*residual)
                info=cho_solve(cho_factor(P,lower=True),np.eye(24))
                info[:12,:12]+=A
                factor=cho_factor(info,lower=True)
                x+=cho_solve(factor,np.r_[b,np.zeros(12)])
                P=cho_solve(factor,np.eye(24)); P=(P+P.T)/2
        clipped[i]=np.any((x[:12]<model.lower)|(x[:12]>model.upper))
        x[:12]=np.clip(x[:12],model.lower,model.upper)
        output[i]=x[:12]; sd[i]=np.sqrt(np.maximum(np.diag(P)[:12],0))
    return output,sd,clipped
