"""Data-only constrained calibration. Absolute offsets remain model estimates.

No reference q is supplied. Anatomical torso conventions, a functional elbow
axis, G99 geometry, and bounded marker/scale priors fix otherwise unresolved
gauges. The resulting model is explicitly estimated/unreviewed, not Hu code.
"""
from pathlib import Path
import copy
import json
import numpy as np
from scipy.io import loadmat
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from .data_io import load_motion,kabsch
from .model import Model,SEGMENTS
from .reference_geometry import compile_reference
from .marker_analysis import fit_template,fit_frame,region
from .kinematics import forward,contact


def unit(x):
    x=np.asarray(x,float)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def bone_rotation(z,x):
    z=unit(z); x=unit(x-np.sum(x*z,axis=-1,keepdims=True)*z)
    y=unit(np.cross(z,x))
    return np.stack([np.cross(y,z),y,z],axis=-1)


def anatomical_torso(points,names):
    def p(name): return points[:,names.index(name)]
    jn,c7,xp,t8=[p(n) for n in ['m_jn','m_c7','m_xp','m_t8']]
    z=unit((jn+c7-xp-t8)/2); y=jn-c7; y=unit(y-np.sum(y*z,axis=1)[:,None]*z)
    x=unit(np.cross(y,z)); y=np.cross(z,x)
    T=np.repeat(np.eye(4)[None],len(points),axis=0)
    T[:,:3,:3]=np.stack([x,y,z],axis=-1); T[:,:3,3]=jn
    valid=np.isfinite(T).all(axis=(1,2))
    valid&=np.linalg.det(np.nan_to_num(T[:,:3,:3]))>.9
    T[~valid]=np.nan
    return T


def select_groups(audit_dir):
    rows=json.loads((Path(audit_dir)/'candidate_groups.json').read_text(encoding='utf-8'))
    selected={}
    for segment,allowed in [('thorax','thorax_region'),('humerus','elbow_region'),('hand','hand_region')]:
        candidates=[]
        for r in rows:
            members=r['marker_names']
            if len(members)<3 or not all(region(n)==allowed for n in members): continue
            status=r['rigidity_status']
            if status.startswith('candidate_'):
                score=(0 if 'strict' in status else 5)+r['worst_marker_holdout_loo_p90_mm']-.05*len(members)
            elif status=='three_marker_fit_only_no_leave_one_out_pose':
                score=15+r['holdout_fit_rms_trial_p90_mm']
            else: continue
            candidates.append((score,r))
        if not candidates: raise ValueError(f'No usable {segment} tracking candidate')
        row=min(candidates,key=lambda v:v[0])[1]
        selected[segment]=dict(markers=row['marker_names'],audit_group=row['group'],
            audit_status=row['rigidity_status'],anatomical_binding_reviewed=False)
    selected['scapula']=dict(markers=['m_acr','m_acr_h1','m_acr_h2'],audit_group='acromion_triplet',
        audit_status='three_point_skin_cluster; orientation gauge fixed by reference prior',anatomical_binding_reviewed=False)
    selected['ulna']=dict(markers=['m_rp_h2','m_rp_h3','m_rp_h4','m_up'],
        audit_group='forearm_axis_point_group plus ulnar-head candidate',
        audit_status='functional hypothesis; m_rp helper prefix is not an anatomical assignment',anatomical_binding_reviewed=False)
    selected['radius']=dict(markers=['m_rp','m_rp_h1'],audit_group='off-axis wrist pair',
        audit_status='two points with constrained RU hinge; exact m_rp anatomy unverified',anatomical_binding_reviewed=False)
    return selected


def training_samples(subject_dir,per_trial=4):
    info=loadmat(Path(subject_dir)/'trial_info.mat',simplify_cells=True)['trial_info']
    points=[]; records=[]; names=None
    for task,row in enumerate(info[:31]):
        ids=np.atleast_1d(row['Kinematics']).ravel()
        if not len(ids): continue
        trial=int(ids[0]); d=load_motion(Path(subject_dir)/'MoCap'/f'trial_{trial:03d}.mat')
        names=d.names
        indices=np.unique(np.linspace(0,len(d.t)-1,per_trial).astype(int))
        for k in indices:
            points.append(d.p[k]); records.append(dict(task=task,trial=trial,frame=int(k),time=float(d.t[k])))
    return np.asarray(points),names,records


def template_poses(points,indices):
    template=fit_template(points[:,indices],3)
    if template is None: raise ValueError('Cluster cannot form a rigid template')
    poses=[]; error=[]
    for frame in points[:,indices]:
        T,e=fit_frame(template,frame)
        poses.append(T if T is not None else np.full((4,4),np.nan))
        error.append(float(np.sqrt(np.nanmean(e**2))) if np.isfinite(e).any() else np.inf)
    return template,np.array(poses),np.array(error)


def functional_elbow(A,B,elbow_prior,torso):
    good=np.isfinite(A).all(axis=(1,2))&np.isfinite(B).all(axis=(1,2))&np.isfinite(elbow_prior).all(1)
    if good.sum()<10: raise ValueError('Not enough frames for functional elbow')
    AA,BB=A[good],B[good]
    prior=elbow_prior[good]-.025*torso[good,:3,0]
    ca=np.median(np.einsum('nji,nj->ni',AA[:,:3,:3],prior-AA[:,:3,3]),axis=0)
    cb=np.median(np.einsum('nji,nj->ni',BB[:,:3,:3],prior-BB[:,:3,3]),axis=0)
    design=np.concatenate([AA[:,:3,:3],-BB[:,:3,:3]],axis=2).reshape(-1,6)
    rhs=(BB[:,:3,3]-AA[:,:3,3]).ravel()
    reference=np.r_[ca,cb]
    fit=least_squares(lambda x:np.r_[(design@x-rhs)/.005,(x-reference)/.03],reference,
                      loss='soft_l1',max_nfev=100)
    residual=(design@fit.x-rhs).reshape(-1,3)
    relative=np.swapaxes(AA[:,:3,:3],1,2)@BB[:,:3,:3]
    vectors=Rotation.from_matrix(relative@relative[0].T).as_rotvec()
    _,sv,vt=np.linalg.svd(vectors,full_matrices=False)
    axis=vt[0]
    if np.dot(AA[0,:3,:3]@axis,torso[good][0,:3,0])<0: axis=-axis
    center=(np.einsum('nij,j->ni',A[:,:3,:3],fit.x[:3])+A[:,:3,3]+
            np.einsum('nij,j->ni',B[:,:3,:3],fit.x[3:])+B[:,:3,3])/2
    world_axis=np.einsum('nij,j->ni',A[:,:3,:3],axis)
    rms=float(np.sqrt(np.mean(np.sum(residual**2,axis=1))))
    report=dict(center_residual_rms_m=rms,center_parameters=fit.x.tolist(),
        design_singular_values=np.linalg.svd(design,compute_uv=False).tolist(),
        rotation_axis_singular_values=sv.tolist(),axis_in_upper_cluster=axis.tolist(),
        axial_gauge_prior='lateral elbow marker minus 25mm torso-right; 30mm regularization, not measured center',
        sufficient_excitation=bool(sv[0]/np.sqrt(len(vectors))>.15),success=bool(fit.success))
    if rms>.02 or not report['sufficient_excitation']:
        center=elbow_prior-.025*torso[:,:3,0]
        report['fallback']='anatomical center prior; functional fit unreliable'
    return center,world_axis,report


def scaled_geometry(points,names):
    cfg=compile_reference(); cfg.update(status='estimated_subject',geometry_reviewed=False)
    def dist(a,b): return np.nanmedian(np.linalg.norm(points[:,names.index(a)]-points[:,names.index(b)],axis=1))
    ratios=[dist('m_jn','m_xp')/np.linalg.norm([.00216,.05665,-.20524]),
            dist('m_jn','m_t8')/np.linalg.norm([.00072,-.16735,-.14986])]
    torso=float(np.clip(np.median(ratios),.7,1.4))
    hum=float(np.clip((dist('m_acr','m_elb')-.04384*torso)/np.linalg.norm(cfg['joints'][3]['center_parent_m']),.7,1.4))
    fore=float(np.clip(dist('m_elb','m_up')/.25711,.7,1.4))
    hand=float(np.sqrt(hum*fore))
    scales=np.array([torso,torso,torso,hum,fore,fore,hand,hand])
    for j in cfg['joints']:
        j['center_parent_m']=(np.array(j['center_parent_m'])*scales[SEGMENTS.index(j['parent'])]).tolist()
        j['center_child_m']=(np.array(j['center_child_m'])*scales[SEGMENTS.index(j['child'])]).tolist()
    c=cfg['scapulothoracic']; T=np.array(c['T_thorax_rib']); T[:3,3]*=torso
    c['T_thorax_rib']=T.tolist(); c['radii_m']=(np.array(c['radii_m'])*torso).tolist()
    c['points_scapula_m']=(np.array(c['points_scapula_m'])*torso).tolist()
    cfg['lower_rad']=np.deg2rad([-60,-60,-90,-90,-90,-150,-120,-180,-15,-30,-100,-60]).tolist()
    cfg['upper_rad']=np.deg2rad([60,60,90,90,90,180,180,180,165,210,100,60]).tolist()
    cfg['initial_q_rad']=np.deg2rad([0,0,0,0,0,0,0,0,90,170,0,0]).tolist()
    cfg['joint_limits_source']='Broad engineering bounds, not Hu or G99 published ROM; sensitivity must be checked'
    return cfg,dict(zip(SEGMENTS,scales.tolist()))


def pose_seed(model,points,names,groups,torso):
    index=lambda values:[names.index(n) for n in values]
    clusters={}
    for segment in ['humerus','ulna','scapula','hand']:
        _,T,e=template_poses(points,index(groups[segment]['markers']))
        clusters[segment]=T
    get=lambda n:points[:,names.index(n)]
    elbow,axis,elbow_report=functional_elbow(clusters['humerus'],clusters['ulna'],get('m_elb'),torso)
    good=np.flatnonzero(np.isfinite(clusters['scapula']).all((1,2))&np.isfinite(torso).all((1,2)))
    if not len(good): raise ValueError('No scapula template anchor')
    ref=model.frames(np.zeros(12))
    anchor=good[0]
    # Only shoulder-girdle gauge uses a zero-pose prior, not the elbow/forearm/wrist.
    C=np.linalg.inv(clusters['scapula'][anchor])@torso[anchor]@ref['scapula']
    scap=clusters['scapula']@C
    gh=np.einsum('nij,j->ni',scap[:,:3,:3],model.joints[2]['center_parent_m'])+scap[:,:3,3]
    H=bone_rotation(gh-elbow,axis)
    U=bone_rotation(elbow-get('m_up'),axis)
    head=elbow+np.einsum('nij,j->ni',U,model.joints[4]['center_parent_m'])
    Rad=bone_rotation(head-get('m_rp'),get('m_rp')-get('m_up'))
    # Radius styloid-to-carpal difference is from G99 Table II/III, scaled by radius length.
    radial_scale=np.linalg.norm(model.joints[5]['center_parent_m'])/np.linalg.norm([-.01834,-.00034,-.27811])
    fe=get('m_rp')+np.einsum('nij,j->ni',Rad,np.array([-.01834,-.00034,-.00366])*radial_scale)
    rd=fe+np.einsum('nij,j->ni',Rad,model.joints[6]['center_parent_m'])
    hand_points=points[:,index(groups['hand']['markers'])]
    hand_R=[]
    for p,z,radius in zip(hand_points,rd-get('m_mf'),Rad):
        finite=np.isfinite(p).all(1)
        if finite.sum()<3 or not np.isfinite(z).all(): hand_R.append(np.full((3,3),np.nan)); continue
        _,_,vt=np.linalg.svd(p[finite]-p[finite].mean(0))
        y=vt[-1]
        if np.dot(y,radius[:,1])<0: y=-y
        zz=unit(z); xx=unit(np.cross(y,zz)); yy=np.cross(zz,xx)
        hand_R.append(np.stack([xx,yy,zz],axis=1))
    target={}
    for segment,R,origin in [('scapula',scap[:,:3,:3],scap[:,:3,3]),('humerus',H,gh),
                              ('ulna',U,elbow),('radius',Rad,head),('hand',np.array(hand_R),rd)]:
        T=np.repeat(np.eye(4)[None],len(points),axis=0); T[:,:3,:3]=R; T[:,:3,3]=origin
        target[segment]=T
    return target,elbow_report


def initialize_angles(model,torso,targets):
    n=len(torso); q=np.repeat(model.q0[None],n,axis=0)
    def residual(v):
        _,_,frames,_,_=forward(model,v,torso)
        result=[]
        for name,target in targets.items():
            finite=np.isfinite(target).all((1,2))
            rotation=(frames[name][:,:3,:3]-np.nan_to_num(target[:,:3,:3])).reshape(n,9)/.2
            translation=(frames[name][:,:3,3]-np.nan_to_num(target[:,:3,3]))/.025
            result.extend([rotation*finite[:,None],translation*finite[:,None]])
        return np.concatenate(result,axis=1)
    damping=np.ones(n)*.05
    for _ in range(35):
        r=residual(q); cost=np.sum(r*r,axis=1)
        J=np.stack([(residual(q+np.eye(12)[i]*1e-5)-r)/1e-5 for i in range(12)],axis=2)
        A=np.swapaxes(J,1,2)@J; g=np.einsum('ndi,nd->ni',J,r)
        step=np.linalg.solve(A+np.eye(12)[None]*damping[:,None,None],-g[:,:,None])[:,:,0]
        step*=np.minimum(1,.3/np.maximum(np.linalg.norm(step,axis=1),1e-12))[:,None]
        candidate=np.clip(q+step,model.lower,model.upper)
        good=np.sum(residual(candidate)**2,axis=1)<cost
        q[good]=candidate[good]; damping=np.where(good,np.maximum(damping*.5,1e-6),np.minimum(damping*5,1e6))
    return q


def estimate_subject(subject_dir,audit_dir,per_trial=4,max_nfev=50):
    points,names,records=training_samples(subject_dir,per_trial)
    groups=select_groups(audit_dir)
    torso=anatomical_torso(points,names)
    good=np.isfinite(torso).all((1,2))
    points=points[good]; torso=torso[good]; records=[r for r,g in zip(records,good) if g]
    if len(points)<30: raise ValueError('Too few anatomical torso initialization frames')
    cfg,scales=scaled_geometry(points,names)
    cfg['subject']=Path(subject_dir).name
    geometry=Model(cfg,allow_estimated=True,require_bindings=False)
    targets,elbow_report=pose_seed(geometry,points,names,groups,torso)
    q0=initialize_angles(geometry,torso,targets)
    _,_,frames,_,_=forward(geometry,q0,torso)
    # Start offsets from geometrically initialized FK, then jointly refine q and offsets.
    bindings={}
    for segment,choice in groups.items():
        T=frames[segment]
        for name in choice['markers']:
            k=names.index(name); measured=points[:,k]
            valid=np.isfinite(measured).all(1)
            local=np.einsum('nji,nj->ni',T[valid,:3,:3],measured[valid]-T[valid,:3,3])
            if len(local)<10: continue
            bindings[name]=dict(segment=segment,local_m=np.median(local,axis=0).tolist(),
                sigma_m=(.006 if segment in ('thorax','scapula','hand') else .004),
                source='automatic constrained multi-pose calibration; bone assignment hypothesis',reviewed=False)
    cfg['markers']=bindings
    model=Model(cfg,allow_estimated=True)
    marker_names=list(bindings); M=len(marker_names); n=len(points)
    measured=points[:,[names.index(k) for k in marker_names]]
    valid=np.isfinite(measured).all(2)
    initial_local=np.array([b['local_m'] for b in bindings.values()])
    offset_sigma=np.array([.002 if b['segment']=='thorax' else .015 for b in bindings.values()])[:,None]
    nq=n*12; nl=M*3
    initial=np.r_[q0.ravel(),np.zeros(nl),np.zeros(4)]
    lower=np.r_[np.tile(model.lower,n),np.full(nl,-.04),np.full(4,np.log(.85))]
    upper=np.r_[np.tile(model.upper,n),np.full(nl,.04),np.full(4,np.log(1.15))]
    data_rows=n*M*3; contact_rows=n*2; angle_rows=n*12
    total=data_rows+contact_rows+angle_rows+nl+4
    sparsity=lil_matrix((total,len(initial)),dtype=np.int8)
    for i in range(n):
        for k in range(M):
            rows=slice(i*M*3+k*3,i*M*3+k*3+3)
            sparsity[rows,i*12:(i+1)*12]=1
            sparsity[rows,nq+k*3:nq+k*3+3]=1; sparsity[rows,nq+nl:]=1
        sparsity[data_rows+i*2:data_rows+i*2+2,i*12:i*12+5]=1
        sparsity[data_rows+i*2:data_rows+i*2+2,nq+nl:]=1
    sparsity[data_rows+contact_rows:data_rows+contact_rows+angle_rows,:nq]=np.eye(nq,dtype=np.int8)
    sparsity[data_rows+contact_rows+angle_rows:data_rows+contact_rows+angle_rows+nl,nq:nq+nl]=np.eye(nl,dtype=np.int8)
    sparsity[-4:,nq+nl:]=np.eye(4,dtype=np.int8)
    sigma=np.array([b['sigma_m'] for b in bindings.values()])
    def unpack(x):
        q=x[:nq].reshape(n,12); delta=x[nq:nq+nl].reshape(M,3)
        a=np.exp(x[-4:]); scale=np.array([a[0],a[0],a[0],a[1],a[2],a[2],a[3],a[3]])
        return q,delta,scale
    def residual(x,prior_multiplier=1.):
        q,delta,s=unpack(x)
        prediction,_,f,ax,ce=forward(model,q,torso,s,initial_local+delta)
        r=np.where(valid[:,:,None],(prediction-np.nan_to_num(measured))/sigma[None,:,None],0)
        cr,_=contact(model,f,ax,ce,s)
        return np.r_[r.ravel(),cr.ravel()/.05,((q-q0)/(.5*prior_multiplier)).ravel(),
                     (delta/(offset_sigma*prior_multiplier)).ravel(),x[-4:]/.05]
    result=least_squares(residual,initial,bounds=(lower,upper),jac_sparsity=sparsity.tocsr(),
        loss='soft_l1',max_nfev=max_nfev,ftol=2e-4,xtol=2e-4,gtol=1e-3)
    q,delta,refinement=unpack(result.x)
    for joint in cfg['joints']:
        joint['center_parent_m']=(np.array(joint['center_parent_m'])*refinement[SEGMENTS.index(joint['parent'])]).tolist()
        joint['center_child_m']=(np.array(joint['center_child_m'])*refinement[SEGMENTS.index(joint['child'])]).tolist()
    c=cfg['scapulothoracic']; T=np.array(c['T_thorax_rib']); T[:3,3]*=refinement[0]
    c['T_thorax_rib']=T.tolist(); c['radii_m']=(np.array(c['radii_m'])*refinement[0]).tolist()
    c['points_scapula_m']=(np.array(c['points_scapula_m'])*refinement[2]).tolist()
    for k,name in enumerate(marker_names): cfg['markers'][name]['local_m']=(initial_local[k]+delta[k]).tolist()
    cfg['initial_q_rad']=np.median(q[[r['task']==0 for r in records]],axis=0).tolist()
    cfg['calibration']=dict(method='anatomical torso + functional elbow + bounded sparse bundle adjustment',
        known_joint_angles_used=False,selected_groups=groups,training_records=records,
        initial_segment_scales=scales,scale_refinement=dict(zip(SEGMENTS,refinement.tolist())),
        functional_elbow=elbow_report,optimizer_success=bool(result.success),optimizer_message=str(result.message),
        optimizer_nfev=int(result.nfev),initial_cost=float(np.sum(2*(np.sqrt(1+residual(initial)**2)-1))),
        final_robust_cost=float(2*result.cost),
        prior_dominated_angles=[True]*8+[False,True,False,False],
        gauge_notes=['Shoulder-girdle reference orientation anchored to G99 at first calibration posture, not measured scapular anatomy',
            'SC axis3 locked at zero by chosen adapter; exact Hu value unknown',
            'Functional elbow hinge-axis coordinate is regularized along the unobservable axial direction',
            'RU absolute phase depends on unverified wrist marker anatomy',
            'Wrist palm plane and longitudinal direction use surface markers; absolute angles need external validation'],
        numerical_review='Automated numerical checks only; anatomy not independently reviewed')
    return Model(cfg,allow_estimated=True),dict(points=points,names=names,torso=torso,q=q,records=records)
