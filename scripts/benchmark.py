#!/usr/bin/env python3
"""Reproduce controlled synthetic measurement and reference-screening experiments.

No independently measured physical recordings are used or implied. Different noise/axis conditions reuse a texture seed. Reported intervals resample
texture-seed clusters, not correlated frames or conditions. Runtime covers estimation, excluding IO, plotting and rendering.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import scipy
from scipy import signal, stats
from scipy.ndimage import map_coordinates, gaussian_filter
from vibescope.motion import estimate_pair,phase_pair
from vibescope.learned import estimate_pair as learned_pair,baseline_pair
from vibescope.signals import analyze_signal,fit_damping
from vibescope.anomaly import train_reference,score

METHODS=('classical','phase','learned','without_learned_calibration')


def hashes():
    paths=['vibescope/motion.py','vibescope/learned.py','vibescope/signals.py','vibescope/anomaly.py','vibescope/models/noise_fusion.npz','scripts/benchmark.py']
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}


def render_sequence(seed,noise,axis,n=180,fps=60.,family='raster',frequency=7.3,amplitude=.25,harmonic=0.,brightness=0.):
    rng=np.random.default_rng(seed)
    yy,xx=np.mgrid[16:80,16:80]
    t=np.arange(n)/fps; truth=amplitude*np.sin(2*np.pi*frequency*t)+harmonic*np.sin(4*np.pi*frequency*t+.3)
    if family=='analytic':
        # Continuous analytic field evaluated at coordinates, unlike Fourier-shift training.
        terms=[(rng.uniform(.02,.06),rng.uniform(.06,.9),rng.uniform(.06,.9),rng.uniform(0,2*np.pi)) for _ in range(9)]
        def field(x,y):
            return .5+sum(a*np.sin(kx*x+ky*y+phi) for a,kx,ky,phi in terms)
    else:
        texture=gaussian_filter(rng.normal(size=(96,96)),1.2)
        texture=.5+.14*texture/texture.std()
        def field(x,y):return map_coordinates(texture,[y,x],order=3,mode='reflect')
    images=[]
    for i,d in enumerate(truth):
        dx,dy=(d,0) if axis=='x' else (0,d)
        image=field(xx-dx,yy-dy)+rng.normal(0,noise,xx.shape)+brightness*np.sin(2*np.pi*1.1*t[i])
        images.append(np.clip(image,0,1).astype(np.float32))
    return np.array(images),truth


def estimate_sequence(frames,method,axis):
    fn={'classical':estimate_pair,'phase':phase_pair,'learned':learned_pair,'without_learned_calibration':baseline_pair}[method]
    values=[];qualities=[];reference=frames[0]
    start=time.perf_counter()
    for frame in frames:
        d=fn(reference,frame,axis=axis) if method=='phase' else fn(reference,frame)
        values.append(d['dx' if axis=='x' else 'dy'])
        if 'quality' in d:
            qualities.append(d['quality'])
        else:
            # The correction ablation retains the same observation gate as the MLP.
            grad=max(d['gradient_rms'],1e-8)
            q=float(np.clip(np.exp(-2*d['residual']/grad)/(1+.15*d['noise_estimate']/grad),0,1))
            if d['gradient_rms']<.001 or d['eigenvalue_ratio']<.01 or max(abs(d['dx']),abs(d['dy']))>1.3:q=0.
            qualities.append(q)
    return np.asarray(values),np.asarray(qualities),time.perf_counter()-start


def evaluate_sequence(frames,truth,method,axis,fps=60.):
    values,q,elapsed=estimate_sequence(frames,method,axis)
    # Static estimator offsets are removed with a documented single DC alignment.
    y=values-values.mean();gt=truth-truth.mean();err=y-gt
    m=analyze_signal(values,fps,4,12)['metrics']
    return dict(method=method,displacement_rmse_px=float(np.sqrt(np.mean(err**2))),
                displacement_bias_px=float(np.mean(values-truth)),
                signal_to_error_db=float(10*np.log10(max(np.mean(gt**2),1e-20)/max(np.mean(err**2),1e-20))),
                frequency_hz=m['frequency_hz'],frequency_error_hz=abs(m['frequency_hz']-7.3) if m['frequency_hz'] is not None else None,
                amplitude_px=m['amplitude_px'],amplitude_abs_error_px=abs(m['amplitude_px']-.25) if m['amplitude_px'] is not None else None,
                quality_median=float(np.median(q)),rejected_frame_fraction=float(np.mean(q<.25)),tracking_status='valid' if np.isfinite(q).all() and np.all(q>=.25) else 'invalid',
                inference_fps=len(values)/elapsed,elapsed_inference_s=elapsed),y


def bootstrap_mean(values,seed=73001):
    y=np.asarray(values,float);rng=np.random.default_rng(seed)
    if len(y)<2:return [float(y[0]),float(y[0])]
    samples=rng.choice(y,(2000,len(y)),replace=True).mean(1)
    return np.quantile(samples,[.025,.975]).tolist()


def aggregate(rows):
    result=[]
    for method in METHODS:
        method_rows=[r for r in rows if r['method']==method]
        selected=[r for r in method_rows if r.get('tracking_status','valid')=='valid']
        if not selected:continue
        item={'method':method,'sequences':len(selected),'excluded_unobservable_sequences':len(method_rows)-len(selected),'independent_texture_seeds':len({r['seed'] for r in selected})}
        for key in ['displacement_rmse_px','frequency_error_hz','amplitude_abs_error_px','signal_to_error_db','inference_fps','rejected_frame_fraction']:
            vals=[r[key] for r in selected if r[key] is not None]
            groups=[[r[key] for r in selected if r['seed']==seed and r[key] is not None] for seed in sorted({r['seed'] for r in selected})]
            groups=[g for g in groups if g]
            if vals:
                # Preserve cluster sizes when rendering families have unequal condition counts.
                rng=np.random.default_rng(73001);ix=rng.integers(0,len(groups),(2000,len(groups)))
                sums=np.array([sum(g) for g in groups]);counts=np.array([len(g) for g in groups])
                draws=sums[ix].sum(axis=1)/counts[ix].sum(axis=1)
                item[key]={'mean':float(np.mean(vals)),'texture_cluster_bootstrap_95pct':np.quantile(draws,[.025,.975]).tolist()}
            else:item[key]=None
        result.append(item)
    return result


def healthy_wave(seed,kind='healthy',n=240,fps=60.):
    rng=np.random.default_rng(seed);t=np.arange(n)/fps
    frequency=rng.normal(7.3,.10);amplitude=np.exp(rng.normal(np.log(.25),.08));noise=rng.uniform(.003,.012);harmonic=rng.uniform(0,.014)
    if kind=='frequency_shift':frequency+=rng.uniform(.35,1.1)
    elif kind=='amplitude_change':amplitude*=rng.uniform(1.25,1.8)
    elif kind=='harmonic_change':harmonic+=rng.uniform(.025,.09)
    elif kind=='broadband_noise':noise+=rng.uniform(.025,.07)
    y=amplitude*np.sin(2*np.pi*frequency*t+rng.uniform(0,2*np.pi))+harmonic*np.sin(4*np.pi*frequency*t+.3)+rng.normal(0,noise,n)
    return y


def wilson(k,n):
    if not n:return [None,None]
    z=1.95996398454;p=k/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d;r=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(max(0,c-r)),float(min(1,c+r))]


def auc(labels,scores):
    labels=np.asarray(labels,bool);scores=np.asarray(scores,float)
    ranks=stats.rankdata(scores)
    n1=labels.sum();n0=(~labels).sum()
    return float((ranks[labels].sum()-n1*(n1+1)/2)/(n1*n0))


def anomaly_experiment(output,quick=False):
    train=[healthy_wave(s) for s in range(710000,710200)]
    calibration=[healthy_wave(s) for s in range(720000,720100)]
    model=train_reference(train,calibration,60.,'Synthetic 7.3Hz target reference, 60fps, 240samples, pixels; not real equipment.')
    model['benchmark_seed_ranges']={'train':[710000,710199],'calibration':[720000,720099]}
    kinds=['healthy','frequency_shift','amplitude_change','harmonic_change','broadband_noise']
    count=30 if quick else 100
    rows=[]
    for ki,kind in enumerate(kinds):
        for seed in range(730000+ki*1000,730000+ki*1000+count):
            value=score(healthy_wave(seed,kind),60.,model)
            rows.append(dict(seed=seed,kind=kind,label_changed=kind!='healthy',**value))
    labels=np.array([r['label_changed'] for r in rows]);scores=np.array([r['score'] for r in rows]);positive=scores>model['threshold']
    rng=np.random.default_rng(73801);indices=np.arange(len(rows));aucs=[]
    for _ in range(1000):
        ix=rng.choice(indices,len(indices),replace=True)
        if labels[ix].any() and (~labels[ix]).any():aucs.append(auc(labels[ix],scores[ix]))
    tp=int(np.sum(positive&labels));fn=int(np.sum(~positive&labels));fp=int(np.sum(positive&~labels));tn=int(np.sum(~positive&~labels))
    result=dict(scope='Controlled synthetic displacement-waveform experiment only; no physical fault/diagnostic validation.',
                train_records=len(train),calibration_records=len(calibration),test_records=len(rows),
                seed_splits_disjoint=True,auroc=auc(labels,scores),auroc_bootstrap_95pct=np.quantile(aucs,[.025,.975]).tolist(),
                sensitivity=tp/(tp+fn),sensitivity_wilson_95pct=wilson(tp,tp+fn),specificity=tn/(tn+fp),specificity_wilson_95pct=wilson(tn,tn+fp),
                confusion={'tp':tp,'fn':fn,'fp':fp,'tn':tn},threshold=model['threshold'],
                by_kind={kind:dict(n=count,flagged=sum(r['status']=='outside_reference' for r in rows if r['kind']==kind)) for kind in kinds},
                limitations=['Reference distribution authored for this target, not generic machinery.','Threshold set on independent calibration signals; test untouched.','Input here is synthetic displacement, not camera image. Image-to-signal errors are tested separately.','Outside-reference means statistical novelty, not damage or defect identification.'])
    (output/'synthetic_reference.json').write_text(json.dumps(model,indent=2)+'\n')
    (output/'anomaly_records.json').write_text(json.dumps(rows,indent=2)+'\n')
    (output/'anomaly.json').write_text(json.dumps(result,indent=2)+'\n')
    return result,rows



def image_reference_check(output, quick=False):
    """Untuned transfer check: waveform reference applied to independent image estimates.

    This intentionally exposes the component boundary. It does not silently train
    a new threshold to make the image subset look better.
    """
    model=json.loads((output/'synthetic_reference.json').read_text())
    count=2 if quick else 10
    records=[]
    for changed in [False,True]:
        for index in range(count):
            seed=810000+int(changed)*1000+index
            rng=np.random.default_rng(seed)
            frequency=rng.normal(7.3,.10);amplitude=np.exp(rng.normal(np.log(.25),.08));harmonic=rng.uniform(0,.014)
            kind='reference_like'
            if changed:
                if index%2:amplitude*=1.7;kind='amplitude_change'
                else:frequency+=.8;kind='frequency_change'
            images,truth=render_sequence(seed,.01,'x',240,family='analytic',frequency=frequency,amplitude=amplitude,harmonic=harmonic)
            values,quality,elapsed=estimate_sequence(images,'classical','x')
            assessment=score(values,60.,model)
            records.append(dict(seed=seed,kind=kind,label_changed=changed,rmse_px=float(np.sqrt(np.mean(((values-values.mean())-(truth-truth.mean()))**2))),
                                quality_median=float(np.median(quality)),elapsed_estimation_s=elapsed,**assessment))
    labels=np.array([r['label_changed'] for r in records]);positive=np.array([r['status']=='outside_reference' for r in records])
    tp=int(np.sum(positive&labels));tn=int(np.sum(~positive&~labels))
    result=dict(scope='Small end-to-end simulated-image transfer check; reference threshold unchanged from waveform calibration.',
                method='classical',frames_per_record=240,fps=60.,shape=[64,64],records=records,
                sensitivity=tp/count,sensitivity_wilson_95pct=wilson(tp,count),specificity=tn/count,specificity_wilson_95pct=wilson(tn,count),
                note='Only frequency/amplitude changes included; no claim of general anomaly reliability. Different image noise can shift the reference distribution.')
    (output/'anomaly_image_check.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def failure_cases():
    t=np.arange(180)/60
    aliased=analyze_signal(np.sin(2*np.pi*52*t),60.,4.,12.)['metrics']
    flat=np.full((64,64),.5)
    damping,dwarning=fit_damping(np.exp(-.012*2*np.pi*8*t)*np.sin(2*np.pi*8*t),60,8)
    outband={str(f):analyze_signal(np.sin(2*np.pi*f*t),60.,4.,12.)['metrics']['frequency_hz'] for f in [1.,2.,3.,15.,20.]}
    noise_resolved=[seed for seed in range(840000,840020) if analyze_signal(np.random.default_rng(seed).normal(size=len(t)),60.,4.,12.)['metrics']['frequency_hz'] is not None]
    cases=dict(out_of_band_tone_reports_hz=outband,white_noise_screen=dict(seeds=[840000,840019],records=20,resolved_seeds=noise_resolved,scope='Small deterministic abstention check, not a false-alarm guarantee.'),aliasing=dict(true_hz=52,sampled_fps=60,reported_hz=aliased['frequency_hz'],recoverable=False,
                            explanation='52Hz and 8Hz are indistinguishable at these sample times. Correct acquisition is mandatory.'),
               flat_texture={m:(learned_pair(flat,flat) if m=='learned' else estimate_pair(flat,flat) if m=='classical' else phase_pair(flat,flat))['quality'] for m in ['classical','phase','learned']},
               free_ringdown=dict(true_damping_ratio=.012,estimate=damping,warning=dwarning))
    return cases


def figures(output,rows,anomaly_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'classical':'#12615b','phase':'#c28733','learned':'#4169a2','without_learned_calibration':'#8b9192'}
    labels={'classical':'Physical fit','phase':'Quadrature phase','learned':'Noise fusion MLP','without_learned_calibration':'No learned correction'}
    with plt.rc_context({'axes.spines.top':False,'axes.spines.right':False,'font.size':10}):
        fig,axs=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
        for method in METHODS:
            r=[r for r in rows if r['method']==method and r['family']=='raster'];noise=sorted(set(x['noise_std'] for x in r))
            axs[0].plot(noise,[np.mean([x['displacement_rmse_px'] for x in r if x['noise_std']==n]) for n in noise],'-o',color=colors[method],label=labels[method],ms=4)
            r=[r for r in rows if r['method']==method]
            axs[1].scatter([METHODS.index(method)]*len(r),[x['frequency_error_hz'] for x in r],color=colors[method],s=12,alpha=.65)
        axs[0].set(xlabel='Added image noise σ (normalized intensity)',ylabel='Displacement RMSE (px)',title='Known subpixel motion · synthetic')
        axs[0].legend(fontsize=8,frameon=False)
        axs[1].set(xticks=range(4),xticklabels=['Physical','Phase','MLP','Ablation'],ylabel='Absolute frequency error (Hz)',title='Sequence errors across conditions')
        healthy=[r['score'] for r in anomaly_rows if not r['label_changed']];changed=[r['score'] for r in anomaly_rows if r['label_changed']]
        axs[2].hist(healthy,bins=18,alpha=.7,color='#12615b',label='Reference-like')
        axs[2].hist(changed,bins=25,alpha=.5,color='#c28733',label='Changed simulation')
        axs[2].axvline(anomaly_rows[0]['threshold'],color='#263a35',ls='--',label='Held-out calibration threshold')
        axs[2].set(xlabel='Reference-distance score (not probability)',ylabel='Records',title='Synthetic reference screening')
        axs[2].legend(fontsize=8,frameon=False)
        fig.savefig(output/'benchmark.png',dpi=180);fig.savefig(output/'benchmark.svg');plt.close(fig)



def refresh_signal_report(output):
    """Reuse image estimates only when their estimator/model/reference sources match."""
    initial=hashes();report=json.loads((output/'results.json').read_text());curves=json.loads((output/'curves.json').read_text())
    inference_sources=['vibescope/motion.py','vibescope/learned.py','vibescope/anomaly.py','vibescope/models/noise_fusion.npz']
    changed=[p for p in inference_sources if report['source_hashes'].get(p)!=initial[p]]
    if changed:raise SystemExit('Image/reference sources changed; run full benchmark: '+', '.join(changed))
    curve_map={(c['seed'],c['noise_std'],c['axis'],c['family'],c['method']):c for c in curves}
    for row in report['rows']:
        key=(row['seed'],row['noise_std'],row['axis'],row['family'],row['method'])
        if key not in curve_map:raise SystemExit('Saved curves are incomplete; run the full benchmark once.')
        row['tracking_status']='valid' if row['rejected_frame_fraction']==0. and row['quality_median']>=.25 else 'invalid'
        y=np.array(curve_map[key]['estimate_px'])
        if row['method']=='without_learned_calibration' and not report.get('ablation_observation_quality_refreshed'):
            frames,_=render_sequence(row['seed'],row['noise_std'],row['axis'],row['frames'],family=row['family'],brightness=.008 if row['family']=='analytic' else 0.)
            check,q,elapsed=estimate_sequence(frames,row['method'],row['axis'])
            check-=check.mean()
            if not np.allclose(check,y,rtol=1e-10,atol=1e-10):raise SystemExit('Cached ablation estimates do not match current seeded inputs. Run full benchmark.')
            row.update(quality_median=float(np.median(q)),rejected_frame_fraction=float(np.mean(q<.25)),inference_fps=len(y)/elapsed,elapsed_inference_s=elapsed)
        m=analyze_signal(y,report['fps'],4.,12.)['metrics']
        row.update(frequency_hz=m['frequency_hz'],frequency_error_hz=None if m['frequency_hz'] is None else abs(m['frequency_hz']-report['frequency_hz']),
                   amplitude_px=m['amplitude_px'],amplitude_abs_error_px=None if m['amplitude_px'] is None else abs(m['amplitude_px']-report['amplitude_px']))
    report.setdefault('original_inference_source_hashes',report['source_hashes'].copy())
    report.setdefault('original_inference_source_changed_during_run',report['source_changed_during_run'])
    report['signal_metrics_refreshed_from_saved_estimates']=True
    report['ablation_observation_quality_refreshed']=True
    report['refresh_protocol']='Image estimates and reference-screening results retained; spectral metrics, failure cases, aggregates and figure regenerated. Ablation diagnostics recomputed from exact seeded inputs with displacement equality asserted, and its timings refreshed. Estimator/checkpoint/reference hashes required to match.'
    report['source_hashes']=initial
    report['summary']=aggregate(report['rows'])
    report['raster_summary']=aggregate([r for r in report['rows'] if r['family']=='raster'])
    report['analytic_summary']=aggregate([r for r in report['rows'] if r['family']=='analytic'])
    report['failure_cases']=failure_cases()
    arows=json.loads((output/'anomaly_records.json').read_text())
    figures(output,report['rows'],arows)
    report['source_changed_during_run']=hashes()!=initial
    (output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (output/'measurements.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(report['rows'][0]));writer.writeheader();writer.writerows(report['rows'])
    print(json.dumps({'status':report['status'],'signal_metrics_refreshed_from_saved_estimates':True,'source_changed_during_run':report['source_changed_during_run']},indent=2))
    if report['source_changed_during_run']:raise SystemExit('Source changed during signal refresh; rerun refresh.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--quick',action='store_true');p.add_argument('--refresh-signals',action='store_true',help='Regenerate only spectral/report metrics from complete cached estimates after signal-only changes');p.add_argument('--output',type=Path,default=ROOT/'experiments/controlled');args=p.parse_args()
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    if args.refresh_signals:
        refresh_signal_report(output);return
    initial=hashes();rows=[];curves=[];start=time.perf_counter()
    seeds=[82101] if args.quick else [82101,82102,82103]
    noises=[.01] if args.quick else [0.,.01,.04]
    n=120 if args.quick else 180
    for seed in seeds:
        for noise in noises:
            for axis in ['x','y']:
                frames,truth=render_sequence(seed,noise,axis,n)
                for method in METHODS:
                    result,values=evaluate_sequence(frames,truth,method,axis)
                    rows.append(dict(seed=seed,noise_std=noise,axis=axis,family='raster',frames=n,**result))
                    curves.append(dict(seed=seed,noise_std=noise,axis=axis,family='raster',method=method,time_s=(np.arange(n)/60).tolist(),truth_px=truth.tolist(),estimate_px=values.tolist()))
                print(f'Raster seed={seed} noise={noise} axis={axis}',flush=True)
    for seed in ([83101] if args.quick else [83101,83102,83103]):
        frames,truth=render_sequence(seed,.01,'x',n,family='analytic',brightness=.008)
        for method in METHODS:
            result,values=evaluate_sequence(frames,truth,method,'x')
            curves.append(dict(seed=seed,noise_std=.01,axis='x',family='analytic',method=method,time_s=(np.arange(n)/60).tolist(),truth_px=truth.tolist(),estimate_px=values.tolist()))
            rows.append(dict(seed=seed,noise_std=.01,axis='x',family='analytic',frames=n,**result))
    anomaly,arows=anomaly_experiment(output,args.quick)
    image_check=image_reference_check(output,args.quick)
    report=dict(status='complete',evidence_scope='Controlled synthetic tests only; no independent physical validation.',quick=args.quick,
                fps=60.,frequency_hz=7.3,amplitude_px=.25,shape=[64,64],source_hashes=initial,
                source_changed_during_run=hashes()!=initial,
                environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),blas_threads=os.environ.get('OPENBLAS_NUM_THREADS')),
                rows=rows,summary=aggregate(rows),raster_summary=aggregate([r for r in rows if r['family']=='raster']),analytic_summary=aggregate([r for r in rows if r['family']=='analytic']),
                failure_cases=failure_cases(),anomaly=anomaly,anomaly_image_check=image_check,elapsed_seconds=time.perf_counter()-start,
                protocol=dict(training_seeds='10000..19999 for supplied MLP',benchmark_seeds=seeds,
                              dc_alignment='Estimate and truth each centered once per clip; no fitted gain correction.',
                              uncertainty='95% percentile bootstrap over texture-seed clusters with their original condition counts (2000 replicates); repeats at different noise/axes stay in the same cluster. Only three texture seeds per family: exploratory, not instrument calibration.',
                              method_notes='Phase is an original quadrature-band estimator, not the complete Wadhwa pyramid algorithm. Ablation uses the MLP middle-scale physical estimate without correction and retains its observation/domain quality gate.',
                              timing='Per-frame estimator CPU only; excludes image generation, video IO, stabilization, enhancement, plotting and warmup. Not end-to-end throughput.'))
    (output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'curves.json').write_text(json.dumps(curves)+'\n')
    with (output/'measurements.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    figures(output,rows,arows)
    print(json.dumps({k:v for k,v in report.items() if k not in ['rows','failure_cases']},indent=2))
    if report['source_changed_during_run']:raise SystemExit('Source changed during benchmark. Rerun on final sources before claiming current evidence.')

if __name__=='__main__':main()
