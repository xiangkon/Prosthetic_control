"""Multi-task marker rigidity audit; empirical clusters are not bone bindings.

Trials, not frames, receive equal weight in pair aggregation. Training uses the
first available valid repetition per task; remaining valid repetitions are held
out. Coordinates are never imputed. Templates live in an arbitrary cluster frame.
"""
from pathlib import Path
from dataclasses import dataclass, asdict
import csv
import json
import hashlib
import numpy as np
from scipy.io import loadmat
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.spatial.transform import Rotation
from .data_io import load_motion,kabsch


@dataclass
class AuditOptions:
    frames_per_trial: int = 100
    loo_frames_per_trial: int = 12
    template_frames_per_trial: int = 5
    template_iterations: int = 3
    min_pair_frames: int = 20
    pair_coverage: float = .8
    strict_mm: float = 3.0
    loose_mm: float = 6.0
    min_dynamic_tasks: int = 8


def quantile(x,q):
    a=np.asarray(x,float); a=a[np.isfinite(a)]
    return float(np.quantile(a,q)) if len(a) else np.nan


def robust_sigma(x,axis=0):
    med=np.nanmedian(x,axis=axis,keepdims=True)
    return 1.4826*np.nanmedian(abs(x-med),axis=axis)


def pair_statistics(points,min_frames=20):
    """Per-trial distance distribution; mm output, no missing-point filling."""
    m=points.shape[1]; ii,jj=np.triu_indices(m,1)
    d=np.linalg.norm(points[:,ii]-points[:,jj],axis=2)*1000
    count=np.isfinite(d).sum(0)
    result=np.full((len(ii),6),np.nan)
    eligible=count>=min_frames
    if np.any(eligible):
        good=d[:,eligible]
        result[eligible,0]=np.nanmedian(good,axis=0)
        result[eligible,1]=robust_sigma(good)
        result[eligible,2]=np.nanquantile(good,.95,axis=0)-np.nanquantile(good,.05,axis=0)
        result[eligible,3]=np.nanstd(good,axis=0)
        result[eligible,4]=np.nanmax(good,axis=0)-np.nanmin(good,axis=0)
        result[eligible,5]=count[eligible]
    return result


def aggregate_pairs(stats,options):
    """Conservative score combines motion within trials and median drift across trials."""
    n=stats.shape[1]; score=np.full(n,np.nan); summaries=[]
    for k in range(n):
        x=stats[:,k]; mask=np.isfinite(x[:,0]); valid=x[mask]
        coverage=float(mask.mean())
        row=dict(valid_trials=int(mask.sum()),trial_coverage=coverage,
            distance_median_mm=quantile(valid[:,0],.5),
            within_sigma_median_mm=quantile(valid[:,1],.5),
            within_sigma_p90_mm=quantile(valid[:,1],.9),
            within_sigma_max_mm=quantile(valid[:,1],1),
            within_span90_p90_mm=quantile(valid[:,2],.9),
            within_span90_max_mm=quantile(valid[:,2],1),
            raw_std_p90_mm=quantile(valid[:,3],.9),
            raw_range_p90_mm=quantile(valid[:,4],.9),
            between_trial_median_sigma_mm=float(robust_sigma(valid[:,0])) if len(valid) else np.nan)
        if coverage>=options.pair_coverage:
            # A strongly articulating minority task must not disappear behind
            # the median/p90 across many unrelated tasks. Convert each trial's
            # central 90% distance range to a comparable spread scale.
            score[k]=max(row['within_sigma_p90_mm'],row['between_trial_median_sigma_mm'],
                         row['within_span90_max_mm']/3.29)
        row['rigidity_score_mm']=float(score[k]); summaries.append(row)
    return score,summaries


def complete_clusters(pair_score,marker_count,threshold):
    """Complete linkage prevents A-B/B-C compatibility from forcing A-C into one group."""
    safe=np.where(np.isfinite(pair_score),pair_score,1e6)
    labels=fcluster(linkage(safe,method='complete'),t=threshold,criterion='distance')
    return [np.flatnonzero(labels==label).tolist() for label in np.unique(labels)]


def fit_template(frames,iterations=3):
    """Generalized Procrustes with median local points; no anatomical coordinate claim."""
    frames=np.asarray(frames,float)
    counts=np.isfinite(frames).all(axis=2).sum(1)
    if counts.max()<3: return None
    template=frames[np.argmax(counts)].copy()
    known=np.isfinite(template).all(axis=1)
    template[known]-=template[known].mean(0)
    for _ in range(iterations):
        estimates=[[] for _ in range(frames.shape[1])]
        for frame in frames:
            valid=np.isfinite(frame).all(axis=1)
            overlap=valid & np.isfinite(template).all(axis=1)
            try: T=kabsch(template[overlap],frame[overlap])
            except ValueError: continue
            local=(frame-T[:3,3])@T[:3,:3]
            for k in np.flatnonzero(valid): estimates[k].append(local[k])
        for k,values in enumerate(estimates):
            if values: template[k]=np.median(values,axis=0)
        known=np.isfinite(template).all(axis=1)
        if known.sum()<3: return None
        template[known]-=template[known].mean(0)
    return template


def fit_frame(template,frame):
    mask=np.isfinite(template).all(axis=1)&np.isfinite(frame).all(axis=1)
    try: T=kabsch(template[mask],frame[mask])
    except ValueError: return None,np.full(len(template),np.nan)
    errors=np.full(len(template),np.nan)
    prediction=template[mask]@T[:3,:3].T+T[:3,3]
    errors[mask]=np.linalg.norm(prediction-frame[mask],axis=1)*1000
    return T,errors


def leave_one_out(template,frame):
    errors=np.full(len(template),np.nan)
    valid=np.isfinite(template).all(axis=1)&np.isfinite(frame).all(axis=1)
    for k in np.flatnonzero(valid):
        others=valid.copy(); others[k]=False
        try: T=kabsch(template[others],frame[others])
        except ValueError: continue
        predicted=T[:3,:3]@template[k]+T[:3,3]
        errors[k]=np.linalg.norm(predicted-frame[k])*1000
    return errors


def region(name):
    if name in ('m_xp','m_jn','m_c7','m_t8') or name.startswith(('m_jn_h','m_c7_h','m_torso_')): return 'thorax_region'
    if name.startswith('m_acr'): return 'shoulder_region'
    if name.startswith('m_elb'): return 'elbow_region'
    if name.startswith('m_rp') or name=='m_up': return 'forearm_region'
    if name.startswith('m_mf'): return 'hand_region'
    return 'unresolved'


def named_groups(names):
    groups={r:[k for k,n in enumerate(names) if region(n)==r]
            for r in ('thorax_region','shoulder_region','elbow_region','forearm_region','hand_region')}
    groups['radius_named_subset']=[k for k,n in enumerate(names) if n.startswith('m_rp')]
    hypotheses={
        'thorax_anatomical4':['m_xp','m_jn','m_c7','m_t8'],
        'acromion_triplet':['m_acr','m_acr_h1','m_acr_h2'],
        'shoulder_aux_triplet':['m_acr_h3','m_acr_h4','m_acr_h5'],
        'elbow_aux_only':[f'm_elb_h{i}' for i in range(1,6)],
        'wrist_triplet':['m_rp','m_up','m_rp_h1'],
        'forearm_axis_point_group':['m_elb','m_rp_h2','m_rp_h3','m_rp_h4'],
        'hand_aux_only':[f'm_mf_h{i}' for i in range(1,6)],
    }
    for group,markers in hypotheses.items(): groups[group]=[names.index(n) for n in markers if n in names]
    return groups


def json_clean(value):
    if isinstance(value,dict): return {k:json_clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_clean(v) for v in value]
    if isinstance(value,np.ndarray): return json_clean(value.tolist())
    if isinstance(value,(float,np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer): return int(value)
    if isinstance(value,np.bool_): return bool(value)
    return value


def write_json(path,value):
    Path(path).write_text(json.dumps(json_clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def write_csv(path,rows):
    if not rows: return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def rotation_sweep(rotations):
    good=np.asarray(rotations)
    good=good[np.isfinite(good).all(axis=(1,2))]
    if len(good)<3: return np.nan
    angle=Rotation.from_matrix(np.einsum('ij,njk->nik',good[0].T,good)).magnitude()
    return quantile(np.rad2deg(angle),.95)


def classify_candidates(summaries,marker_rows,quality_rows,options):
    """A good median cannot promote a group containing an unreliable marker."""
    candidates=[]
    for row in summaries:
        name=row['group']
        member=[r for r in marker_rows if r['group']==name]
        per_marker=[float(r['holdout_loo_p90_mm']) if r['holdout_loo_p90_mm'] is not None else np.nan for r in member]
        all_loo_finite=len(member)==len(row['marker_names']) and np.isfinite(per_marker).all()
        worst=float(np.max(per_marker)) if all_loo_finite else np.nan
        cover=[]
        for marker in row['marker_names']:
            trials=[r for r in quality_rows if r['marker_name']==marker and r['split']=='holdout' and int(r['task'])!=0]
            cover.append(np.mean([int(r['sample_count'])>=options.min_pair_frames for r in trials]) if trials else 0)
        min_cover=float(min(cover))
        adequate=(row['training_dynamic_tasks']>=options.min_dynamic_tasks and row['holdout_trials']>=3
                  and row['holdout_coverage_median']>=.8 and row['secondary_spread_mm']>=5 and min_cover>=.8)
        if not adequate:
            status='insufficient_coverage_or_geometry'
        elif len(row['marker_names'])==3:
            status='three_marker_fit_only_no_leave_one_out_pose'
        elif not all_loo_finite:
            status='insufficient_marker_leaveout_evidence'
        elif row['holdout_fit_rms_trial_p90_mm']<=3 and row['holdout_loo_trial_p90_mm']<=6 and worst<=6:
            status='candidate_strict_typical_holdout'
        elif row['holdout_fit_rms_trial_p90_mm']<=6 and row['holdout_loo_trial_p90_mm']<=12 and worst<=12:
            status='candidate_loose_typical_holdout'
        else:
            status='nonrigid_or_unstable'
        candidates.append(dict(**row,rigidity_status=status,worst_marker_holdout_loo_p90_mm=worst,
            minimum_marker_holdout_trial_coverage=min_cover,anatomical_binding_reviewed=False,
            warning='Cluster coordinates and rigidity do not identify a G99 bone frame; joint-axis markers may be compatible with multiple bones'))
    return candidates


def write_candidate_tables(out,names,candidates):
    write_json(out/'candidate_groups.json',candidates)
    membership=[]
    for name in names:
        usable=[r for r in candidates if name in r['marker_names'] and r['rigidity_status'].startswith('candidate_')]
        membership.append(dict(marker_name=name,name_region_prior=region(name),
            supported_candidate_groups=';'.join(r['group'] for r in usable),
            support_count=len(usable),bone_local_coordinate_status='unresolved',anatomical_binding_reviewed=False))
    write_csv(out/'marker_candidates.csv',membership)


def audit_subject(subject_dir,out,options):
    subject_dir=Path(subject_dir); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    source=loadmat(subject_dir/'trial_info.mat',simplify_cells=True)['trial_info']
    records=[]; names=None; frame_rows=[]; all_stats=[]
    for task,row in enumerate(np.atleast_1d(source)):
        if task>30: continue  # exclude eye-only tasks as articulation evidence
        ids=[int(v) for v in np.atleast_1d(row['Kinematics']).ravel()]
        for repetition,trial in enumerate(ids):
            path=subject_dir/'MoCap'/f'trial_{trial:03d}.mat'
            data=load_motion(path)
            if names is None: names=data.names
            if data.names!=names: raise ValueError(f'Changing marker order: {path}')
            index=np.unique(np.linspace(0,len(data.t)-1,min(options.frames_per_trial,len(data.t))).astype(int))
            p=data.p[index]; t=data.t[index]
            split='train' if repetition==0 else 'holdout'
            stats=pair_statistics(p,options.min_pair_frames)
            all_stats.append(stats)
            record=dict(task=task,trial=trial,split=split,description=str(row['Description']),p=p,t=t,
                        sample_frames=index,source=str(path))
            records.append(record)
            for k,name in enumerate(names):
                valid=data.valid[:,k]
                adjacent=valid[1:] & valid[:-1]
                speeds=np.linalg.norm(np.diff(data.p[:,k],axis=0),axis=1)/np.diff(data.t)
                frame_rows.append(dict(subject=subject_dir.name,task=task,trial=trial,split=split,
                    marker_name=name,valid_ratio=float(valid.mean()),sample_count=int(np.isfinite(p[:,k]).all(axis=1).sum()),
                    speed_p99_m_s=quantile(speeds[adjacent],.99),
                    fraction_speed_above_10m_s=float(np.mean(speeds[adjacent]>10)) if adjacent.any() else np.nan))
    if not records: raise ValueError('No valid movement trials')
    stats=np.stack(all_stats); train=np.array([r['split']=='train' for r in records]); holdout=~train
    score,pairs=aggregate_pairs(stats[train],options)
    _,test_pairs=aggregate_pairs(stats[holdout],options) if holdout.any() else (None,[{}]*len(score))
    ii,jj=np.triu_indices(len(names),1)
    pair_rows=[]
    for k,row in enumerate(pairs):
        row=dict(marker_a=names[ii[k]],marker_b=names[jj[k]],**row)
        row.update({f'holdout_{key}':value for key,value in test_pairs[k].items()})
        pair_rows.append(row)
    write_csv(out/'pair_rigidity.csv',pair_rows)
    write_csv(out/'marker_quality.csv',frame_rows)
    np.savez_compressed(out/'trial_pair_statistics.npz',statistics=stats,
        fields=np.array(['median_mm','robust_sigma_mm','span90_mm','raw_std_mm','raw_range_mm','count']),
        marker_names=np.array(names),pair_i=ii,pair_j=jj,
        trials=np.array([r['trial'] for r in records]),tasks=np.array([r['task'] for r in records]),
        splits=np.array([r['split'] for r in records]))

    groups=named_groups(names)
    empirical={}
    for threshold in (options.strict_mm,options.loose_mm):
        clusters=complete_clusters(score,len(names),threshold)
        empirical[str(threshold)]=[[names[k] for k in c] for c in clusters]
        for c in clusters:
            if len(c)>=3:
                name=f'empirical_{threshold:g}mm_'+'_'.join(str(k+1) for k in c)
                if not any(c==v for v in groups.values()): groups[name]=c
    training_frames=[]
    for r in records:
        if r['split']=='train':
            idx=np.unique(np.linspace(0,len(r['p'])-1,min(options.template_frames_per_trial,len(r['p']))).astype(int))
            training_frames.extend(r['p'][idx])
    training_frames=np.array(training_frames)
    group_rows=[]; marker_group_rows=[]; transforms={}; templates={}
    all_loo={}; all_fit={}
    for group,indices in groups.items():
        template=fit_template(training_frames[:,indices],options.template_iterations)
        if template is None:
            continue
        templates[group]=dict(marker_names=[names[k] for k in indices],coordinates_m=template.tolist(),
            frame='arbitrary learned cluster frame; NOT G99 bone-local coordinates')
        known=np.isfinite(template).all(axis=1)
        s=np.linalg.svd(template[known]-template[known].mean(0),compute_uv=False)
        spread=s[1]/np.sqrt(known.sum())*1000 if len(s)>1 else 0
        loo_by_split={'train':[],'holdout':[]}; fit_by_split={'train':[],'holdout':[]}
        transforms[group]=[]; all_loo[group]=[]; all_fit[group]=[]
        for r in records:
            local=r['p'][:,indices]
            rotations=[]; errors=[]; per_marker=[]
            for frame in local:
                T,e=fit_frame(template,frame)
                rotations.append(T[:3,:3] if T is not None else np.full((3,3),np.nan))
                per_marker.append(e)
                errors.append(np.sqrt(np.nanmean(e*e)) if np.isfinite(e).any() else np.nan)
            rotations=np.array(rotations); per_marker=np.array(per_marker)
            transforms[group].append(rotations)
            loo_idx=np.unique(np.linspace(0,len(local)-1,min(options.loo_frames_per_trial,len(local))).astype(int))
            loo=np.array([leave_one_out(template,local[k]) for k in loo_idx])
            loo_medians=np.array([quantile(loo[:,k],.5) for k in range(len(indices))])
            fit_medians=np.array([quantile(per_marker[:,k],.5) for k in range(len(indices))])
            loo_by_split[r['split']].append(loo_medians)
            fit_by_split[r['split']].append(fit_medians)
            all_loo[group].append(loo_medians); all_fit[group].append(fit_medians)
            group_rows.append(dict(subject=subject_dir.name,group=group,task=r['task'],trial=r['trial'],split=r['split'],
                markers=len(indices),finite_template_markers=int(known.sum()),
                fit_frame_coverage=float(np.isfinite(errors).mean()),
                fit_rms_median_mm=quantile(errors,.5),fit_rms_p95_mm=quantile(errors,.95),
                loo_error_median_mm=quantile(loo,.5),loo_error_p95_mm=quantile(loo,.95),
                rotation_sweep_p95_deg=rotation_sweep(rotations),
                template_secondary_spread_mm=float(spread)))
        for k,index in enumerate(indices):
            row=dict(subject=subject_dir.name,group=group,marker_name=names[index])
            for split in ('train','holdout'):
                fits=np.array(fit_by_split[split]); loos=np.array(loo_by_split[split])
                row[f'{split}_fit_median_mm']=quantile(fits[:,k],.5) if len(fits) else np.nan
                row[f'{split}_loo_median_mm']=quantile(loos[:,k],.5) if len(loos) else np.nan
                row[f'{split}_loo_p90_mm']=quantile(loos[:,k],.9) if len(loos) else np.nan
            marker_group_rows.append(row)
    write_csv(out/'group_trial_fit.csv',group_rows)
    write_csv(out/'marker_group_fit.csv',marker_group_rows)
    write_json(out/'cluster_templates.json',templates)

    relative=[]
    group_pairs=[('thorax_region','shoulder_region'),('shoulder_region','elbow_region'),
                 ('elbow_region','forearm_region'),('elbow_region','radius_named_subset'),
                 ('forearm_region','hand_region'),('radius_named_subset','hand_region'),
                 ('thorax_region','acromion_triplet'),('thorax_region','shoulder_aux_triplet'),
                 ('thorax_anatomical4','acromion_triplet'),('thorax_anatomical4','elbow_aux_only'),
                 ('elbow_aux_only','forearm_axis_point_group'),
                 ('forearm_axis_point_group','wrist_triplet'),('wrist_triplet','hand_aux_only')]
    for a,b in group_pairs:
        if a not in transforms or b not in transforms: continue
        for r,Ra,Rb in zip(records,transforms[a],transforms[b]):
            good=np.isfinite(Ra).all(axis=(1,2))&np.isfinite(Rb).all(axis=(1,2))
            rel=np.einsum('nji,njk->nik',Ra[good],Rb[good])
            relative.append(dict(group_a=a,group_b=b,task=r['task'],trial=r['trial'],split=r['split'],
                relative_rotation_sweep_p95_deg=rotation_sweep(rel) if len(rel) else np.nan,
                note='empirical cluster relative motion, not anatomical joint angle'))
    write_csv(out/'relative_cluster_motion.csv',relative)

    summaries=[]
    for group,indices in groups.items():
        rows=[r for r in group_rows if r['group']==group]
        if not rows: continue
        test=[r for r in rows if r['split']=='holdout' and r['task']!=0]
        training=[r for r in rows if r['split']=='train' and r['task']!=0]
        summaries.append(dict(group=group,marker_names=[names[k] for k in indices],
            training_dynamic_tasks=len({r['task'] for r in training}),holdout_trials=len(test),
            holdout_fit_rms_trial_median_mm=quantile([r['fit_rms_median_mm'] for r in test],.5),
            holdout_fit_rms_trial_p90_mm=quantile([r['fit_rms_median_mm'] for r in test],.9),
            holdout_loo_trial_median_mm=quantile([r['loo_error_median_mm'] for r in test],.5),
            holdout_loo_trial_p90_mm=quantile([r['loo_error_median_mm'] for r in test],.9),
            holdout_fit_rms_worst_trial_median_mm=quantile([r['fit_rms_median_mm'] for r in test],1),
            holdout_loo_worst_trial_median_mm=quantile([r['loo_error_median_mm'] for r in test],1),
            holdout_coverage_median=quantile([r['fit_frame_coverage'] for r in test],.5),
            secondary_spread_mm=rows[0]['template_secondary_spread_mm']))
    candidates=classify_candidates(summaries,marker_group_rows,frame_rows,options)
    write_candidate_tables(out,names,candidates)
    write_json(out/'summary.json',dict(subject=subject_dir.name,options=asdict(options),
        source_trial_info_sha256=hashlib.sha256((subject_dir/'trial_info.mat').read_bytes()).hexdigest(),
        trials=len(records),training_trials=int(train.sum()),holdout_trials=int(holdout.sum()),
        tasks=sorted({r['task'] for r in records}),empirical_clusters=empirical,group_summaries=summaries,
        status='exploratory rigidity evidence, not reviewed anatomical bindings'))
    write_csv(out/'trial_selection.csv',[dict(task=r['task'],trial=r['trial'],split=r['split'],description=r['description'],
        source=r['source'],sampled_frames=len(r['t'])) for r in records])
    return dict(subject=subject_dir.name,trials=len(records),training_trials=int(train.sum()),holdout_trials=int(holdout.sum()),
                summaries=summaries,clusters=empirical)
