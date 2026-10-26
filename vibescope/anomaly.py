"""Healthy-reference novelty screening. It cannot identify damage or diagnose faults."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import signal

FEATURE_NAMES=['log_rms_px','peak_hz','harmonic_ratio','off_peak_energy']


def _signal(values,fps):
    y=np.asarray(values,float)
    if y.ndim!=1 or len(y)<40 or not np.isfinite(y).all():
        raise ValueError('Reference screening needs at least40 finite one-dimensional samples.')
    if not np.isfinite(fps) or fps<=2.5 or fps>2000:
        raise ValueError('Reference screening needs a valid constant capture fps above2.5.')
    return y


def features(values, fps):
    y=signal.detrend(_signal(values,fps));n=len(y)
    f,p=signal.periodogram(y,fs=fps,window='hann')
    valid=(f>1)&(f<fps*.4)
    if not valid.any() or np.std(y)<1e-7:raise ValueError('No observable vibration for reference screening.')
    i=np.where(valid)[0][np.argmax(p[valid])];peak=f[i]
    total=np.sum(p[valid])+1e-20
    fundamental=np.sum(p[np.abs(f-peak)<max(fps/n*2,.4)])
    harmonic=np.sum(p[np.abs(f-2*peak)<max(fps/n*2,.4)]) if 2*peak<fps/2 else 0
    return np.array([np.log(np.sqrt(np.mean(y*y))+1e-9),peak,np.log1p(harmonic/max(fundamental,1e-20)*10),1-fundamental/total])


def train_reference(training, calibration, fps, scope):
    training=list(training);calibration=list(calibration)
    if len(training)<10 or len(calibration)<5:
        raise ValueError('Use at least10 independent training and5 independent calibration recordings.100+ calibration recordings are recommended for tail estimates.')
    if not isinstance(scope,str) or not scope.strip() or len(scope)>512:
        raise ValueError('Describe the acquisition setup and region in a nonempty reference scope (up to512 characters).')
    data=np.array([features(y,fps) for y in training])
    center=np.median(data,axis=0)
    scale=np.maximum(1.4826*np.median(abs(data-center),axis=0),[.02,.15,.015,.02])
    scores=np.array([np.sqrt(np.mean(((features(y,fps)-center)/scale)**2)) for y in calibration])
    lengths=[len(y) for y in training+calibration]
    return dict(version=1,scope=scope.strip(),fps=float(fps),sample_count_range=[min(lengths),max(lengths)],feature_names=FEATURE_NAMES,
                center=center.tolist(),scale=scale.tolist(),threshold=float(np.quantile(scores,.99)),
                training_records=len(training),calibration_records=len(calibration),
                threshold_kind='Empirical99th percentile; no distribution-free false-alarm guarantee',
                calibration_tail_resolution=1/(len(calibration)+1),
                acquisition_match='User must confirm same camera, ROI, axis, scale, exposure and processing method.',
                limitations='Novelty relative to this reference and acquisition setup only. Not a diagnostic probability. Small calibration sets do not resolve rare false-alarm rates.')


def validate_reference(model):
    if not isinstance(model,dict) or model.get('version')!=1 or model.get('feature_names')!=FEATURE_NAMES:
        raise ValueError('Unsupported reference model. Generate it with the current VibeScope baseline command.')
    try:
        center=np.asarray(model['center'],float);scale=np.asarray(model['scale'],float)
        threshold=float(model['threshold']);fps=float(model['fps']);counts=model['sample_count_range']
        valid=(center.shape==(4,) and scale.shape==(4,) and np.isfinite(center).all() and np.isfinite(scale).all() and np.all(scale>0)
               and np.isfinite([threshold,fps]).all() and threshold>=0 and 2.5<fps<=2000 and len(counts)==2
               and 40<=counts[0]<=counts[1] and model['training_records']>=10 and model['calibration_records']>=5
               and isinstance(model['scope'],str) and bool(model['scope'].strip()) and len(model['scope'])<=512)
        if not valid:raise ValueError('Invalid numeric metadata.')
    except (KeyError,TypeError,ValueError,IndexError) as e:
        raise ValueError('Invalid reference model metadata. Rebuild the baseline from original recordings.') from e
    return model


def score(values, fps, model, confirmed_scope=None):
    model=validate_reference(model);values=_signal(values,fps)
    if not np.isclose(fps,model['fps'],rtol=1e-4):
        raise ValueError('Reference capture fps does not match this recording.')
    if not model['sample_count_range'][0]<=len(values)<=model['sample_count_range'][1]:
        raise ValueError('Recording length is outside the reference range. Compare the same acquisition duration.')
    if confirmed_scope is not None and confirmed_scope!=model['scope']:
        raise ValueError('Reference acquisition scope does not match the confirmed setup.')
    z=(features(values,fps)-np.array(model['center']))/np.array(model['scale'])
    value=float(np.sqrt(np.mean(z*z)))
    return dict(status='outside_reference' if value>model['threshold'] else 'within_reference',score=value,
                threshold=model['threshold'],reference_scope=model['scope'],diagnostic=False,
                acquisition_match='user_confirmed' if confirmed_scope else 'unverified',
                explanation='Distance from reference recordings; outside-reference does not imply damage or identify a fault. '+('Setup match was user-declared.' if confirmed_scope else 'Setup match is unverified.'))


def load_reference(path):
    path=Path(path)
    if path.stat().st_size>64*1024:raise ValueError('Reference JSON is too large (64KiB maximum).')
    try:return validate_reference(json.loads(path.read_text()))
    except (json.JSONDecodeError,UnicodeDecodeError) as e:raise ValueError('Reference must be valid UTF-8 JSON.') from e
