"""Vectorized FK and exact first derivatives for independent trajectory frames."""
import numpy as np
from scipy.spatial.transform import Rotation
from .model import SEGMENTS


def forward(model,q,poses=None,scales=None,local=None):
    q=np.asarray(q,float)
    if q.ndim==1: q=q[None]
    n=len(q)
    if poses is None: poses=np.repeat(np.eye(4)[None],n,axis=0)
    scale=np.ones(8) if scales is None else np.asarray(scales)
    frames={'thorax':poses.copy()}
    axes=np.zeros((n,12,3)); centers=np.zeros((n,12,3))
    for joint in model.joints:
        parent=frames[joint['parent']]
        cp=np.array(joint['center_parent_m'])*scale[SEGMENTS.index(joint['parent'])]
        cc=np.array(joint['center_child_m'])*scale[SEGMENTS.index(joint['child'])]
        center=np.einsum('nij,j->ni',parent[:,:3,:3],cp)+parent[:,:3,3]
        R=np.repeat(np.eye(3)[None],n,axis=0)
        for entry in joint['axes']:
            a=np.array(entry['direction']); index=entry['q_index']
            if index is not None:
                axes[:,index]=np.einsum('nij,njk,k->ni',parent[:,:3,:3],R,a)
                centers[:,index]=center; angle=q[:,index]
            else: angle=np.full(n,entry['fixed_rad'])
            R=R@Rotation.from_rotvec(angle[:,None]*a).as_matrix()
        R=R@np.array(joint['R_parent_child_zero'])
        T=np.repeat(np.eye(4)[None],n,axis=0); T[:,:3,:3]=R
        T[:,:3,3]=cp-np.einsum('nij,j->ni',R,cc)
        frames[joint['child']]=parent@T
    names=list(model.bindings)
    prediction=np.empty((n,len(names),3)); jac=np.zeros((n,len(names),3,12))
    cutoffs=[0,2,5,8,9,10,11,12]
    for k,name in enumerate(names):
        b=model.bindings[name]; segment=b['segment']; T=frames[segment]
        point=np.array(b['local_m']) if local is None else local[k]
        p=np.einsum('nij,j->ni',T[:,:3,:3],point)+T[:,:3,3]
        prediction[:,k]=p; stop=cutoffs[SEGMENTS.index(segment)]
        jac[:,k,:,:stop]=np.cross(axes[:,:stop],p[:,None]-centers[:,:stop]).transpose(0,2,1)
    return prediction,jac,frames,axes,centers


def contact(model,frames,axes,centers,scales=None):
    n=len(axes); c=model.contact
    if not c: return np.empty((n,0)),np.empty((n,0,12))
    s=np.ones(8) if scales is None else scales
    rib=np.array(c['T_thorax_rib']).copy(); rib[:3,3]*=s[0]
    world_rib=frames['thorax']@rib; world_scapula=frames['scapula']
    points=np.array(c['points_scapula_m'])*s[2]
    radii=np.array(c['radii_m'])*s[0]
    world=np.einsum('nij,kj->nki',world_scapula[:,:3,:3],points)+world_scapula[:,None,:3,3]
    p=np.einsum('nji,nkj->nki',world_rib[:,:3,:3],world-world_rib[:,None,:3,3])
    residual=np.sum((p/radii)**2,axis=2)-1
    derivative=np.zeros((n,2,12))
    for k in range(2):
        velocity=np.cross(axes[:,:5],world[:,k,None]-centers[:,:5])
        gradient=np.einsum('nij,nj->ni',world_rib[:,:3,:3],2*p[:,k]/radii**2)
        derivative[:,k,:5]=np.einsum('ni,nji->nj',gradient,velocity)
    return residual,derivative


def residual_jacobian(model,q,poses,measured,valid,contact_sigma=.05,local=None,scales=None):
    prediction,J,frames,axes,centers=forward(model,q,poses,scales,local)
    sigma=np.array([b['sigma_m'] for b in model.bindings.values()])
    mask=valid & np.isfinite(measured).all(2)
    r=np.where(mask[:,:,None],(prediction-np.nan_to_num(measured))/sigma[None,:,None],0)
    J=J/sigma[None,:,None,None]*mask[:,:,None,None]
    cr,cj=contact(model,frames,axes,centers,scales)
    return np.concatenate([r.reshape(len(q),-1),cr/contact_sigma],axis=1),np.concatenate([
        J.reshape(len(q),-1,12),cj/contact_sigma],axis=1),prediction


def solve_trajectory(model,measured,valid,poses,initial,iterations=30,contact_sigma=.05,prior_sigma=2.0):
    """Independent-frame robust damped Gauss–Newton, with explicit start prior.

    No temporal penalty is hidden in this solver. The EKF is a separate stage.
    An accepted cost decrease is required; convergence is recorded per frame.
    """
    q=np.clip(np.asarray(initial,float).copy(),model.lower,model.upper)
    start=q.copy(); n=len(q); damping=np.ones(n)*.01
    converged=np.zeros(n,bool); last_step=np.full(n,np.inf)
    def evaluate(v):
        r,J,p=residual_jacobian(model,v,poses,measured,valid,contact_sigma)
        r=np.concatenate([r,(v-start)/prior_sigma],axis=1)
        J=np.concatenate([J,np.repeat((np.eye(12)/prior_sigma)[None],n,axis=0)],axis=1)
        return r,J,p
    r,J,p=evaluate(q); cost=np.sum(2*(np.sqrt(1+r*r)-1),axis=1)
    for _ in range(iterations):
        w=1/np.sqrt(1+r*r)
        normal=np.swapaxes(J,1,2)@(J*w[:,:,None])
        gradient=np.einsum('ndi,nd->ni',J,r*w)
        diag=np.maximum(np.diagonal(normal,axis1=1,axis2=2),1)
        step=np.linalg.solve(normal+np.eye(12)[None]*(damping[:,None]*diag)[:,None,:],-gradient[:,:,None])[:,:,0]
        step*=np.minimum(1,.35/np.maximum(np.linalg.norm(step,axis=1),1e-15))[:,None]
        trial=np.clip(q+step,model.lower,model.upper)
        rr,JJ,pp=evaluate(trial); cc=np.sum(2*(np.sqrt(1+rr*rr)-1),axis=1)
        good=cc<cost
        actual_step=np.linalg.norm(trial-q,axis=1)
        small=(actual_step<1e-5)&good
        converged|=small | (np.max(abs(gradient),axis=1)<1e-4)
        q[good]=trial[good]; r[good]=rr[good]; J[good]=JJ[good]; p[good]=pp[good]
        cost[good]=cc[good]; last_step[good]=actual_step[good]
        damping=np.where(good,np.maximum(damping*.4,1e-7),np.minimum(damping*5,1e8))
        if np.all(converged): break
    return q,dict(converged=converged,last_step_rad=last_step,robust_cost=cost,predicted=p)
