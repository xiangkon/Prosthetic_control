"""Guarded TUM EMG clock decoding and empirically checked pulse alignment."""
import numpy as np
from scipy.io import loadmat
from ._matlab import _tum_workspace


def read_emg_record(path):
    """Return the validated clock, full source array, and clock metadata."""
    d=loadmat(path,simplify_cells=True)
    handle=d.get('None')
    if type(handle).__name__!='MatlabOpaque': raise ValueError('Unsupported EMG MAT layout')
    h=handle.ravel()[0]
    expected=np.array([3707764736,2,1,1,1,5],dtype=np.uint32)
    if h['s2']!=b'timeseries' or not np.array_equal(h['arr'].ravel(),expected):
        raise ValueError('Unsupported EMG timeseries handle')
    s=_tum_workspace(d['__function_workspace__'])
    if len(s)!=21: raise ValueError('Unsupported uniform EMG MCOS property layout')
    for slot,obj,cls in [(4,3,1),(6,2,2),(14,4,3),(18,5,4)]:
        if not np.array_equal(np.asarray(s[slot]).ravel(),[3707764736,2,1,1,obj,cls]):
            raise ValueError('Unexpected EMG object property reference')
    if str(s[20][3].Units)!='seconds': raise ValueError('EMG time unit is not seconds')
    data=np.asarray(s[19],float); n=int(s[8]); dt=float(s[11]); start=float(s[12])
    if data.shape!=(n,74) or not np.isfinite([dt,start]).all() or dt<=0:
        raise ValueError('Unexpected EMG dimensions or clock')
    if not np.isclose(dt,.0005,atol=1e-10) or not np.allclose(np.diff(data[:,73]),1):
        raise ValueError('EMG sample period/counter is inconsistent')
    time=start+np.arange(n)*dt
    return time,data,dict(samples=n,dt_s=dt,start_s=start,
        source_time_unit='seconds',sample_counter_contiguous=True,raw_channels=74,
        bipolar_definition='source emg_uni2bi.m: raw(1)-raw(2), ..., raw(63)-raw(64)')


def read_emg_clock(path):
    """Keep the existing pulse-only interface used by reconstruction."""
    time,data,metadata=read_emg_record(path)
    return time,data[:,72],metadata


def infer_pulse_alignment(emg_t,pulse,mocap_t,tolerance_s=.006):
    values=np.unique(pulse)
    if not np.array_equal(values,[254,255]): raise ValueError('No supported 254/255 pulse channel')
    edges=np.r_[0,np.flatnonzero(np.diff(pulse)!=0)+1,len(pulse)]
    dt=float(np.median(np.diff(emg_t))); md=float(np.median(np.diff(mocap_t)))
    duration=float(mocap_t[-1]-mocap_t[0]+md)
    candidates=[]
    for begin,end in zip(edges[:-1],edges[1:]):
        if pulse[begin]!=254 or begin==0 or end==len(pulse): continue
        measured=(end-begin)*dt
        if abs(measured-duration)<=tolerance_s:
            candidates.append((begin,end,measured))
    if len(candidates)!=1:
        raise ValueError(f'Expected one duration-matched pulse; found {len(candidates)}')
    begin,end,measured=candidates[0]
    return dict(status='duration_validated_pulse_candidate',offset_s=float(emg_t[begin]-mocap_t[0]),
        clock_scale=1.0,pulse_start_index=int(begin),pulse_end_index_exclusive=int(end),
        pulse_duration_s=float(measured),mocap_nominal_duration_s=duration,
        duration_error_s=float(measured-duration),timing_resolution_s=max(dt,md),
        evidence='EMG column 73 low interval uniquely matches MoCap nominal duration; column74 is contiguous sample counter',
        limitation='Channel semantics/latency not independently documented; pulse-based alignment is empirical, not assumed common t=0')


def resample_estimates(t,q,angle_flags,target,max_gap_s=.01):
    target=np.asarray(target); output=np.full((len(target),q.shape[1]),np.nan,dtype=np.float32)
    flags=np.full((len(target),q.shape[1]),1,dtype=np.uint16)
    right=np.searchsorted(t,target)
    exact_target=np.zeros(len(target),bool)
    in_range=right<len(t)
    exact_target[in_range]=np.isclose(t[right[in_range]],target[in_range],atol=1e-9,rtol=0)
    for j in range(q.shape[1]):
        exact=(right<len(t))
        idx=right[exact]; hit=np.isclose(t[idx],target[exact],atol=1e-9,rtol=0)
        positions=np.flatnonzero(exact)[hit]; source=idx[hit]
        output[positions,j]=q[source,j]; flags[positions,j]=angle_flags[source,j]
        inside=(right>0)&(right<len(t))&~exact_target
        dest=np.flatnonzero(inside); r=right[inside]; l=r-1
        good=(t[r]-t[l]<=max_gap_s)&np.isfinite(q[l,j])&np.isfinite(q[r,j])
        dest,r,l=dest[good],r[good],l[good]
        ratio=(target[dest]-t[l])/(t[r]-t[l])
        output[dest,j]=(1-ratio)*q[l,j]+ratio*q[r,j]
        flags[dest,j]=angle_flags[l,j]|angle_flags[r,j]
    return output,flags
