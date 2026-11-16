"""Check full-cohort output integrity, not anatomical angle accuracy."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from .pipeline import jobs,code_hash
from .marker_analysis import write_json
from .paths import OUTPUT_ROOT, DATA_ROOT


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default=str(OUTPUT_ROOT)); p.add_argument('--data',default=str(DATA_ROOT))
    p.add_argument('--subjects',default='1,2,3,4,5,6')
    args=p.parse_args(argv); root=Path(args.root); version=code_hash()
    counts=dict(trials=0,method_files=0,frames=0,emg_label_files=0,emg_unavailable=0,empirical_sync_max_error_ms=0.)
    errors=[]
    for s in [int(v) for v in args.subjects.split(',')]:
        name=f'subj_{s:03d}'; folder=root/name
        try:
            model=json.loads((folder/'estimated_model.json').read_text(encoding='utf-8'))
            fingerprint=hashlib.sha256(json.dumps(model,sort_keys=True).encode()).hexdigest()
        except (OSError,ValueError) as exc:
            errors.append(dict(subject=name,error=str(exc))); continue
        prior=np.array(model['calibration']['prior_dominated_angles'])
        for job in jobs(Path(args.data)/name):
            dest=folder/'trials'/f'trial_{job["trial"]:03d}'
            try:
                report=json.loads((dest/'report.json').read_text(encoding='utf-8'))
                assert report['model_sha256']==fingerprint,'model hash mismatch'
                assert report.get('cache_code_sha256',report['code_sha256'])==version,'code version mismatch'
                count=None
                for method in ('ik','ekf'):
                    with np.load(dest/f'{method}_angles.npz',allow_pickle=False) as z:
                        t=z['t']; q=z['q']; flags=z['angle_quality_flag']
                        assert q.shape==(len(t),12) and flags.shape==q.shape,'shape mismatch'
                        assert len(t)==report['output_frames'],'frame count mismatch'
                        assert np.all(np.diff(t)>0),'nonmonotonic time'
                        if report['output_rate_hz']==500:
                            assert len(t)==report['source_frames'],'full rate frame count changed'
                            assert np.allclose(np.diff(t),.002,atol=1e-9,rtol=0),'not 500Hz'
                        assert np.all((flags[:,prior]&64)!=0),'lost prior flags'
                        assert np.all(z['quality_flag']==np.bitwise_or.reduce(flags,axis=1)),'frame flags inconsistent'
                        assert np.all(flags[~np.isfinite(q)]!=0),'nonfinite unflagged angle'
                        finite=np.isfinite(q)
                        lower=np.repeat(np.array(model['lower_rad'])[None],len(q),axis=0)
                        upper=np.repeat(np.array(model['upper_rad'])[None],len(q),axis=0)
                        assert np.all(q[finite]>=lower[finite]-1e-5) and np.all(q[finite]<=upper[finite]+1e-5),'bounds violated'
                        count=len(t)
                    counts['method_files']+=1
                counts['frames']+=count; counts['trials']+=1
                if report['sync']['status']=='duration_validated_pulse_candidate':
                    with np.load(dest/'emg_aligned_distal.npz',allow_pickle=False) as z:
                        assert z['q_distal_estimated'].shape==(len(z['emg_time']),4),'EMG label shape'
                        assert np.array_equal(z['strict_valid'],z['angle_quality_flag']==0),'EMG mask mismatch'
                        assert np.all(z['angle_quality_flag'][~np.isfinite(z['q_distal_estimated'])]!=0),'unflagged EMG gap'
                        assert np.all(~z['strict_valid'][:,1]),'unverified RU phase promoted to strict label'
                    counts['emg_label_files']+=1
                    counts['empirical_sync_max_error_ms']=max(counts['empirical_sync_max_error_ms'],abs(report['sync']['duration_error_s'])*1000)
                elif report['sync']['status']=='not_available': counts['emg_unavailable']+=1
                else: errors.append(dict(subject=name,trial=job['trial'],error='unresolved synchronization',detail=report['sync']))
            except (OSError,ValueError,KeyError,AssertionError) as exc:
                errors.append(dict(subject=name,trial=job['trial'],error=str(exc)))
    result=dict(passed=not errors,scope='file/time/model/quality integrity only; not external angle accuracy',counts=counts,errors=errors)
    write_json(root/'output_integrity.json',result)
    print(json.dumps(result,indent=2))
    if errors: raise SystemExit(1)


if __name__=='__main__': main()
