"""Resumable process scheduler for the same deterministic pipeline calculations."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import hashlib
import json
import time
import shutil
import tempfile
import numpy as np
from .pipeline import code_hash,jobs,validate_subject,run_trial
from .calibration import estimate_subject
from .model import Model
from .marker_analysis import write_json
from .paths import DATA_ROOT, OUTPUT_ROOT, ANALYSIS_ROOT


def prepare_candidates(folder, audit):
    """Keep the candidate groups needed for calibration after temporary analysis."""
    target = audit / 'candidate_groups.json'
    if target.exists():
        return
    from .marker_analysis import audit_subject, AuditOptions
    print(f'{folder.name}: analysing marker groups', flush=True)
    with tempfile.TemporaryDirectory(prefix='tum_marker_analysis_') as temporary:
        generated = Path(temporary)
        audit_subject(folder, generated, AuditOptions())
        audit.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(generated / 'candidate_groups.json', target)


def review_calibration(model,folder,audit):
    """Retry a finite optimization budget, and propagate a functional fallback.

    Retrying does not relax fit/quality thresholds. A geometrical elbow fallback
    cannot silently become a fully observed elbow absolute coordinate.
    """
    changed=False
    if not model.config['calibration']['optimizer_success']:
        model,_=estimate_subject(folder,audit,max_nfev=150)
        model.config['calibration']['optimization_retry']=dict(max_nfev=150,reason='original finite evaluation budget exhausted; thresholds unchanged')
        changed=True
    f=model.config['calibration']['functional_elbow']
    if 'fallback' in f or not f['sufficient_excitation']:
        if not model.config['calibration']['prior_dominated_angles'][8]: changed=True
        model.config['calibration']['prior_dominated_angles'][8]=True
        reason='Functional calibration fell back to anatomical prior; absolute q9 is prior-dependent'
        if model.config['calibration'].get('elbow_quality_reason')!=reason: changed=True
        model.config['calibration']['elbow_quality_reason']=reason
    return model,changed


def worker(task):
    folder,job,model_path,dest,iterations,rate=task
    try:
        model=Model.load(model_path,allow_estimated=True)
        return model.config['subject'],run_trial(folder,job,model,dest,iterations,rate)
    except Exception as exc:
        return Path(folder).name,dict(**job,status='failed',error=f'{type(exc).__name__}: {exc}')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=DATA_ROOT); p.add_argument('--audit',type=Path,default=ANALYSIS_ROOT)
    p.add_argument('--output',type=Path,default=OUTPUT_ROOT); p.add_argument('--subjects',default='1,2,3,4,5,6')
    p.add_argument('--workers',type=int,default=3); p.add_argument('--rate-hz',type=float,default=500)
    p.add_argument('--iterations',type=int,default=35)
    p.add_argument('--self-check',action='store_true',help='Check numerical routines and any existing selected results')
    args=p.parse_args(argv)
    if args.self_check:
        from .self_check import run_checks
        run_checks(args.root)
        if args.output.exists():
            from .integrity import main as check_outputs
            check_outputs(['--root',str(args.output),'--data',str(args.root),'--subjects',args.subjects])
        return
    if not 1<=args.workers<=8 or not 0<args.rate_hz<=500 or args.iterations<1: p.error('Invalid worker/rate/iteration count')
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    version=code_hash(); start=time.time(); results={}; tasks=[]; model_states={}; expected={}
    for s in [int(v) for v in args.subjects.split(',')]:
        name=f'subj_{s:03d}'; folder=Path(args.root)/name; out=output/name; out.mkdir(exist_ok=True)
        path=out/'estimated_model.json'
        try:
            model=Model.load(path,allow_estimated=True) if path.exists() else None
            if model is not None and model.config.get('cache_code_sha256',model.config.get('pipeline_code_sha256'))!=version: model=None
            if model is None:
                prepare_candidates(folder,Path(args.audit)/name)
                model,_=estimate_subject(folder,Path(args.audit)/name)
                model,_=review_calibration(model,folder,Path(args.audit)/name)
                validation=validate_subject(folder,model)
                model.config['pipeline_code_sha256']=version; model.save(path)
                write_json(out/'calibration_report.json',model.config['calibration'])
                write_json(out/'validation_report.json',validation)
            else:
                model,changed=review_calibration(model,folder,Path(args.audit)/name)
                if changed:
                    validation=validate_subject(folder,model)
                    model.config['pipeline_code_sha256']=version; model.save(path)
                    write_json(out/'calibration_report.json',model.config['calibration'])
                    write_json(out/'validation_report.json',validation)
            model_states[name]=model.config['calibration_validation']['geometry_failed']
        except Exception as exc:
            write_json(out/'calibration_failure.json',dict(error=f'{type(exc).__name__}: {exc}'))
            print(f'{name}: calibration failed: {exc}',flush=True)
            model_states[name]=True
            continue
        results[name]={}; expected[name]=jobs(folder)
        expected_model_hash=hashlib.sha256(json.dumps(model.config,sort_keys=True).encode()).hexdigest()
        for job in expected[name]:
            dest=out/'trials'/f'trial_{job["trial"]:03d}'; report=None
            try:
                old=json.loads((dest/'report.json').read_text(encoding='utf-8'))
                if (old.get('cache_code_sha256',old.get('code_sha256'))==version and old.get('requested_rate_hz')==args.rate_hz
                        and old.get('solver_iterations')==args.iterations
                        and old.get('model_sha256')==expected_model_hash
                        and all((dest/f'{method}_angles.npz').exists() for method in ('ik','ekf'))):
                    report=old
            except (OSError,ValueError): pass
            if report is not None: results[name][job['trial']]=report
            else: tasks.append((str(folder),job,str(path),str(dest),args.iterations,args.rate_hz))
        print(f'{name}: model ready, reused {len(results[name])}/{len(expected[name])} trials',flush=True)

    def checkpoint():
        manifest=[]
        for name,rows in results.items():
            ordered=[rows[k] for k in sorted(rows)]
            write_json(output/name/'trial_manifest.json',ordered)
            manifest.append(dict(subject=name,calibration_failed=model_states[name],trials=len(ordered),
                expected_trials=len(expected[name]),computed=sum(r['status']=='computed_estimates' for r in ordered),
                failed=sum(r['status']=='failed' for r in ordered),
                sync_matched=sum(r.get('sync',{}).get('status')=='duration_validated_pulse_candidate' for r in ordered)))
        write_json(output/'manifest.json',manifest)
    checkpoint()
    completed=0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending=[pool.submit(worker,task) for task in tasks]
        for future in as_completed(pending):
            name,result=future.result(); results[name][result['trial']]=result; completed+=1
            if completed%10==0 or completed==len(tasks):
                checkpoint()
                done=sum(len(v) for v in results.values())
                print(f'Completed {done}/{sum(map(len,expected.values()))}; new {completed}/{len(tasks)}; elapsed {time.time()-start:.1f}s',flush=True)
                failed=[(n,r['trial'],r.get('error')) for n,rr in results.items() for r in rr.values() if r['status']=='failed']
                if failed: print('Failures: '+str(failed),flush=True)
    checkpoint()
    print('All scheduled trials finished.',flush=True)
    from .reporting import main as report_main
    report_main(['--root',str(output),'--data',str(args.root),'--subjects',args.subjects])
    from .integrity import main as integrity_main
    integrity_main(['--root',str(output),'--data',str(args.root),'--subjects',args.subjects])


if __name__=='__main__': main()
