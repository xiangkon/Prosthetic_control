"""Compile a traced G99 reference geometry from visually transcribed tables.

No TUM marker attachment, subject scaling, physiological joint limits, or Hu
filter parameters are inferred here. The original 13 axes remain in the archive;
the 12-coordinate adapter explicitly locks SC axis 3 to zero as a chosen option.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

from .paths import PROJECT, REFERENCE_TABLES

SOURCE = REFERENCE_TABLES


def read_tables():
    return json.loads(SOURCE.read_text(encoding='utf-8'))


def rotation(axes):
    """Project printed 6-decimal direction cosines to nearest proper SO(3)."""
    raw=np.asarray(axes,float).T
    u,_,vt=np.linalg.svd(raw)
    return u@np.diag([1,1,np.linalg.det(u@vt)])@vt


def transform(record):
    T=np.eye(4)
    T[:3,:3]=rotation(record['axes'])
    T[:3,3]=np.asarray(record['origin'])*1e-3
    return T


def compile_reference(source=None):
    source=read_tables() if source is None else source
    joints=[]; next_index=0
    for row in source['table_III']:
        frame=source['table_I'][row['child']]
        R0=rotation(frame['in_parent']['axes'])
        axes=[]
        if 'axes_child' in row:
            for k,child_axis in enumerate(row['axes_child']):
                direction=R0@np.asarray(child_axis,float)
                entry=dict(direction=(direction/np.linalg.norm(direction)).tolist(),
                           source='G99 Table III distal axis transformed by Table I zero rotation')
                if row['name']=='SC' and k==2:
                    entry.update(q_index=None,fixed_rad=0.0)
                else:
                    entry['q_index']=next_index
                    next_index+=1
                axes.append(entry)
        else:
            direction=np.asarray(row['axis_parent'],float)
            axes=[dict(direction=(direction/np.linalg.norm(direction)).tolist(),q_index=next_index,
                       source='G99 Table III proximal axis, normalized')]
            next_index+=1
        joints.append(dict(parent=row['parent'],child=row['child'],
            center_parent_m=(np.asarray(row['center_parent'])*1e-3).tolist(),
            center_child_m=(np.asarray(row['center_child'])*1e-3).tolist(),
            R_parent_child_zero=R0.tolist(),axes=axes,
            source=dict(centers_axes='G99 Table III, printed p.120, PDF p.15',
                        zero_rotation=f'G99 Table I, printed p.{frame["printed_page"]}')))
    points=[r['in_segment'] for name in ('SM','IM') for r in source['table_II'] if r['name']==name]
    cfg=dict(schema='upper-limb-ordered-axes-v1',units='m,rad,s',status='reference_geometry',
        subject=None,geometry_reviewed=True,
        provenance='G99 Tables I–III visual transcription and numerical cross-check; single VHP reference, not TUM calibration',
        source_tables='configs/motion_reconstruction/garner1999_tables.json',
        source_tables_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        reference_pose=source['reference_pose'],
        normalization='Each printed rotation projected to nearest SO(3) by SVD; raw values remain in source archive',
        adapter_assumptions=dict(SC_axis3_fixed_rad=0.0,
            SC_lock_source='Chosen 13-to-12 adapter; exact Hu lock value unverified',
            wrist_order='G99 RC-FE then RC-RD; Hu q11/q12 labels/signs unverified'),
        lower_rad=None,upper_rad=None,initial_q_rad=[0.0]*12,
        joint_limits_source='Not supplied here: reference FK does not require solver bounds',
        joints=joints,markers={},
        scapulothoracic=dict(T_thorax_rib=transform(source['table_I']['rib_cage']['in_thorax']).tolist(),
            radii_m=(np.asarray(source['ellipsoid']['radii'])*1e-3).tolist(),
            points_scapula_m=(np.asarray(points)*1e-3).tolist(),source='G99 p.115 and Table II p.116'),
        rhythm=None)
    cfg['coordinate_definitions']=[dict(q_index=i+1,joint=name,motion=motion,
        sign='right-hand rotation about the compiled ordered axis; Hu sign correspondence unverified')
        for i,(name,motion) in enumerate([
            ('SC','Elevation/Depression'),('SC','Protraction/Retraction'),
            ('AC','Lateral Abduction/Adduction'),('AC','Internal/External Rotation'),
            ('AC','Posterior Abduction/Adduction'),('GH','Flexion/Extension'),
            ('GH','Abduction/Adduction'),('GH','External/Internal Rotation'),
            ('HU','Flexion/Extension'),('RU','Pronation/Supination'),
            ('RC_FE','Flexion/Extension'),('RC_RD','Radial/Ulnar Deviation')])]
    pdf=PROJECT/'references'/source['source']['file']
    if pdf.exists(): cfg['source_pdf_sha256']=hashlib.sha256(pdf.read_bytes()).hexdigest()
    return cfg




def audit_reference(model,source=None):
    source=read_tables() if source is None else source
    frames=model.frames(np.zeros(12))
    raw_checks=[]; chains=[]; features=[]; axes=[]
    for segment,row in source['table_I'].items():
        for frame_key in ('in_thorax','in_parent'):
            if frame_key not in row: continue
            raw=np.asarray(row[frame_key]['axes']).T
            corrected=rotation(row[frame_key]['axes'])
            raw_checks.append(dict(segment=segment,frame=frame_key,
                raw_orthogonality_fro=float(np.linalg.norm(raw.T@raw-np.eye(3))),
                raw_determinant=float(np.linalg.det(raw)),
                max_entry_correction=float(np.max(abs(corrected-raw)))))
        if segment in frames:
            target=transform(row['in_thorax']); actual=frames[segment]
            chains.append(dict(segment=segment,
                origin_error_mm=float(np.linalg.norm(actual[:3,3]-target[:3,3])*1000),
                rotation_max_entry_error=float(np.max(abs(actual[:3,:3]-target[:3,:3])))))
    for row in source['table_II']:
        T=frames[row['segment']]
        x=np.asarray(row['in_segment'],float)
        actual=T[:3,:3]@x
        if row['kind']=='point': actual+=T[:3,3]*1000
        error=float(np.linalg.norm(actual-row['in_thorax']))
        features.append(dict(segment=row['segment'],name=row['name'],kind=row['kind'],
                             error=error,unit='mm' if row['kind']=='point' else 'dimensionless'))
    for row,joint in zip(source['table_III'],model.joints):
        R0=np.asarray(joint['R_parent_child_zero'])
        if 'axes_child' in row:
            actual=R0@row['axes_child'][0]; expected=np.asarray(row['axis1_parent'])
        else:
            actual=R0@row['axis_child']; expected=np.asarray(row['axis_parent'])
        axes.append(dict(joint=row['name'],proximal_distal_axis_error=float(np.linalg.norm(actual-expected))))
    contact=model.constraints(np.zeros(12))
    summary=dict(max_chain_origin_error_mm=max(r['origin_error_mm'] for r in chains),
        max_chain_rotation_entry_error=max(r['rotation_max_entry_error'] for r in chains),
        max_point_feature_error_mm=max(r['error'] for r in features if r['kind']=='point'),
        max_direction_feature_error=max(r['error'] for r in features if r['kind']=='direction'),
        max_axis_representation_error=max(r['proximal_distal_axis_error'] for r in axes),
        max_raw_rotation_correction=max(r['max_entry_correction'] for r in raw_checks),
        reference_contact_residual=contact.tolist(),table_II_features=len(features),original_axis_count=13,adapter_coordinate_count=12)
    # Printed origins have two decimals; one clavicle orientation has a larger
    # internal discrepancy than six-decimal rounding alone can explain.
    passed=(summary['max_chain_origin_error_mm']<.1 and
            summary['max_chain_rotation_entry_error']<2e-4 and
            summary['max_point_feature_error_mm']<.1 and
            summary['max_direction_feature_error']<2e-4 and
            summary['max_axis_representation_error']<2e-4 and
            np.max(abs(contact))<1e-3)
    return dict(passed=bool(passed),summary=summary,rotation_checks=raw_checks,
        chain_checks=chains,feature_checks=features,axis_checks=axes,
        source_note='Printed Table I clavicle rotation is not exactly orthonormal; archive preserved and SVD correction reported. Do not equate corrected values with exact author code.')
