"""Explicit ordered-axis model. Missing geometry fails instead of being guessed."""
from pathlib import Path
import copy
import json
import numpy as np
from scipy.spatial.transform import Rotation

SEGMENTS = ['thorax','clavicle','scapula','humerus','ulna','radius','carpal','hand']
Q_NAMES = ['SC1','SC2','AC1','AC2','AC3','GH1','GH2','GH3','HU_FE','RU_PS','RC_FE','RC_RD']


def finite_difference(fun, q, step=1e-6):
    q = np.asarray(q,float)
    return np.column_stack([(np.asarray(fun(q+np.eye(len(q))[i]*step)).ravel() -
                             np.asarray(fun(q-np.eye(len(q))[i]*step)).ravel())/(2*step)
                            for i in range(len(q))])


def vector(value, size, label):
    if value is None or np.shape(value) != (size,):
        raise ValueError(f'Missing or malformed {label}: expected {size} numbers')
    a = np.asarray(value, float)
    if not np.isfinite(a).all():
        raise ValueError(f'Nonfinite {label}')
    return a


class Model:
    def __init__(self, config, allow_demo=False, require_bindings=True, allow_reference=False, allow_estimated=False):
        self.config = copy.deepcopy(config)
        if config.get('schema') != 'upper-limb-ordered-axes-v1' or config.get('units') != 'm,rad,s':
            raise ValueError('Unsupported schema or units')
        if config.get('status') == 'template':
            raise ValueError('Incomplete G99 template: fill Tables I–III geometry and subject calibration first')
        reference = config.get('status') == 'reference_geometry'
        if reference and not allow_reference:
            raise ValueError('G99 reference geometry is not a calibrated subject model; use allow_reference=True for FK inspection only')
        self.demo = config.get('status') == 'synthetic_demo'
        if self.demo and not allow_demo:
            raise ValueError('Synthetic demo geometry cannot be used for real reconstruction')
        self.estimated = config.get('status') == 'estimated_subject'
        if self.estimated and not allow_estimated:
            raise ValueError('Automatically estimated subject model: explicitly use allow_estimated=True and preserve calibration quality flags')
        if not self.demo and not (self.estimated and allow_estimated) and config.get('geometry_reviewed') is not True:
            raise ValueError('Subject geometry must be reviewed and geometry_reviewed=true')
        self.lower = np.full(12,-np.inf) if reference else vector(config.get('lower_rad'),12,'lower_rad')
        self.upper = np.full(12,np.inf) if reference else vector(config.get('upper_rad'),12,'upper_rad')
        self.q0 = vector(config.get('initial_q_rad'),12,'initial_q_rad')
        if np.any(self.lower >= self.upper) or np.any(self.q0 < self.lower) or np.any(self.q0 > self.upper):
            raise ValueError('Invalid bounds or initial q')
        self.joints = config['joints']
        if len(self.joints) != 7:
            raise ValueError('Expected seven serial joints')
        used = []
        patterns=[[0,1,None],[2,3,4],[5,6,7],[8],[9],[10],[11]]
        for j,(parent,child) in enumerate(zip(SEGMENTS,SEGMENTS[1:])):
            joint = self.joints[j]
            if (joint['parent'],joint['child']) != (parent,child):
                raise ValueError('Unexpected joint topology')
            if [a.get('q_index') for a in joint['axes']] != patterns[j]:
                raise ValueError(f'Unexpected coordinate order: {child}')
            vector(joint.get('center_parent_m'),3,child+' parent center')
            vector(joint.get('center_child_m'),3,child+' child center')
            R=np.asarray(joint.get('R_parent_child_zero'),float)
            if R.shape != (3,3) or not np.allclose(R.T@R,np.eye(3),atol=2e-4) or not np.isclose(np.linalg.det(R),1,atol=2e-4):
                raise ValueError(f'Invalid zero rotation: {child}')
            for axis in joint['axes']:
                v=vector(axis.get('direction'),3,child+' axis')
                if not np.isclose(np.linalg.norm(v),1,atol=2e-4):
                    raise ValueError('Axis direction must be unit length')
                if axis.get('q_index') is not None:
                    used.append(axis['q_index'])
                elif not np.isfinite(axis.get('fixed_rad',np.nan)):
                    raise ValueError('Fixed coordinate needs an explicit fixed_rad')
        if sorted(used) != list(range(12)):
            raise ValueError('Each q index 0..11 must occur exactly once')
        self.bindings = config.get('markers',{})
        if require_bindings and not self.bindings:
            raise ValueError('No calibrated marker bindings')
        for name,binding in self.bindings.items():
            if binding.get('segment') not in SEGMENTS:
                raise ValueError(f'Invalid segment for {name}')
            vector(binding.get('local_m'),3,name+' local coordinate')
            sigma=float(binding.get('sigma_m',0))
            if sigma <= 0 or not np.isfinite(sigma):
                raise ValueError(f'Positive sigma_m required: {name}')
        self.contact = config.get('scapulothoracic')
        if self.contact:
            T=np.asarray(self.contact['T_thorax_rib'],float)
            if T.shape != (4,4) or not np.allclose(T[3],[0,0,0,1]) or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=2e-4) or np.linalg.det(T[:3,:3]) < 0:
                raise ValueError('Invalid Rib Cage transform')
            if np.any(vector(self.contact['radii_m'],3,'ellipsoid radii') <= 0):
                raise ValueError('Positive ellipsoid radii required')
            if np.shape(self.contact['points_scapula_m']) != (2,3) or not np.isfinite(self.contact['points_scapula_m']).all():
                raise ValueError('Expected two finite shoulder contact points')
        rhythm=config.get('rhythm')
        if rhythm:
            A=np.asarray(rhythm['A'],float); b=np.asarray(rhythm['b_rad'],float)
            if A.ndim!=2 or A.shape[1]!=12 or b.shape!=(A.shape[0],) or not np.isfinite(A).all() or not np.isfinite(b).all():
                raise ValueError('Invalid explicitly chosen rhythm A,b')
            if rhythm.get('source') is None:
                raise ValueError('Record the source of the chosen rhythm rule')

    @classmethod
    def load(cls,path,**kwargs):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')),**kwargs)

    def frames(self,q):
        q=vector(q,12,'q')
        frames={'thorax':np.eye(4)}
        for j in self.joints:
            R=np.eye(3)
            for a in j['axes']:
                angle=q[a['q_index']] if a.get('q_index') is not None else a['fixed_rad']
                R=R@Rotation.from_rotvec(np.asarray(a['direction'])*angle).as_matrix()
            R=R@np.asarray(j['R_parent_child_zero'])
            T=np.eye(4); T[:3,:3]=R
            T[:3,3]=np.asarray(j['center_parent_m'])-R@np.asarray(j['center_child_m'])
            frames[j['child']]=frames[j['parent']]@T
        return frames

    def predict(self,q,names,world_thorax):
        frames=self.frames(q)
        points=[]
        for name in names:
            b=self.bindings[name]; T=world_thorax@frames[b['segment']]
            points.append(T[:3,:3]@np.asarray(b['local_m'])+T[:3,3])
        return np.asarray(points).reshape(-1,3)

    def constraints(self,q):
        residual=[]
        if self.contact:
            c=self.contact
            T=np.linalg.inv(c['T_thorax_rib'])@self.frames(q)['scapula']
            p=np.asarray(c['points_scapula_m'])@T[:3,:3].T+T[:3,3]
            residual.extend(np.sum((p/np.asarray(c['radii_m']))**2,axis=1)-1)
        return np.asarray(residual)

    def rhythm(self,q):
        r=self.config.get('rhythm')
        return np.asarray(r['A'])@q-np.asarray(r['b_rad']) if r else np.empty(0)

    def save(self,path):
        Path(path).write_text(json.dumps(self.config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
