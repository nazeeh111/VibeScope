"""Controlled external neural magnification comparison. Synthetic, not a physical experiment."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, json, time, hashlib, platform, argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import cv2, numpy as np, torch
from scipy.ndimage import gaussian_filter, map_coordinates
from vibescope.motion import estimate_pair, phase_pair
from vibescope.learned import estimate_pair as learned_pair
from vibescope.signals import analyze_signal
from run_neural_baseline import load_model, infer_pair, COMMIT, SOURCE_SHA256, WEIGHT_SHA256

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,required=True,help='Pinned EulerMormer checkout')
parser.add_argument('--output',type=Path,default=ROOT/'experiments/neural_baseline')
parser.add_argument('--refresh-signals',action='store_true',help='Recompute only spectral metrics from saved curves after a signal-only code change')
args=parser.parse_args()
out=args.output;out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1)
if args.refresh_signals:
 report=json.loads((out/'comparison.json').read_text())
 curves=json.loads((out/'curves.json').read_text())
 for path in ['vibescope/motion.py','vibescope/learned.py','vibescope/models/noise_fusion.npz','scripts/run_neural_baseline.py']:
  if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=report['source_hashes'].get(path):raise SystemExit('Inference sources changed; full rerun required: '+path)
 report.setdefault('original_inference_source_hashes',report['source_hashes'].copy())
 for row in report['rows']:
  curve=next(c for c in curves if c['seed']==row['seed'] and c['noise_std_0to1']==row['noise_std_0to1'])
  values=np.array(curve['estimates_px'][row['method']]);values-=values.mean()
  q=np.asarray(curve['tracking_quality'][row['method']])
  row.update(tracking_quality_median=float(np.median(q)),tracking_rejected_fraction=float(np.mean(q<.25)),tracking_status='valid' if np.isfinite(q).all() and np.all(q>=.25) else 'invalid')
  metrics=analyze_signal(values,report['fps'],4,12)['metrics']
  row.update(frequency_hz=metrics['frequency_hz'],frequency_error_hz=None if metrics['frequency_hz'] is None else abs(metrics['frequency_hz']-report['true_frequency_hz']),amplitude_px=metrics['amplitude_px'],amplitude_abs_error_px=None if metrics['amplitude_px'] is None else abs(metrics['amplitude_px']-report['true_amplitude_px']))
 report['spectral_metrics_recomputed_from_saved_original_estimates']=True
 report['limits']=[x.replace('median>=0.25 and <=10% rejected frames.','every frame must be finite with quality>=0.25; any rejected frame invalidates the sequence.') for x in report['limits']]
 report['source_hashes']['vibescope/signals.py']=hashlib.sha256((ROOT/'vibescope/signals.py').read_bytes()).hexdigest()
 report['source_hashes']['scripts/benchmark_neural.py']=hashlib.sha256((ROOT/'scripts/benchmark_neural.py').read_bytes()).hexdigest()
 # Inference remains valid only if the estimator/checkpoint sources match their original hashes.
 report['source_changed_during_run']=any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h for p,h in report['source_hashes'].items())
 (out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
 (out/'source_hashes.json').write_text(json.dumps(report['source_hashes'],indent=2)+'\n')
 print(json.dumps(report,indent=2));sys.exit(0)
hash_paths=['vibescope/motion.py','vibescope/learned.py','vibescope/signals.py','vibescope/models/noise_fusion.npz','scripts/run_neural_baseline.py','scripts/benchmark_neural.py']
source_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in hash_paths}
(out/'source_hashes.json').write_text(json.dumps(source_hashes,indent=2)+'\n')
model=load_model(args.source)
rows=[]; curves=[]; fps=60.; n=120; alpha=10.; f0=7.3
for seed in [91341,91342]:
 for noise in [0.,.03]:
  rng=np.random.default_rng(seed)
  texture=gaussian_filter(rng.normal(size=(96,96)),.9)
  texture=.5+.18*texture/texture.std()
  yy,xx=np.mgrid[16:80,16:80]
  t=np.arange(n)/fps; truth=.25*np.sin(2*np.pi*f0*t)
  frames=[]
  for dx in truth:
   a=map_coordinates(texture,[yy,xx-dx],order=3,mode='reflect')
   a=np.clip(a+rng.normal(0,noise,a.shape),0,1)
   frame=np.round(a*255).astype(np.uint8)
   frames.append(frame)
  estimates={m:[] for m in ['classical','phase','learned','EulerMormer']}
  timings={m:0. for m in estimates}
  qualities={m:[] for m in estimates}
  # One untimed inference per case excludes first-call model/backend initialization.
  first_color=cv2.cvtColor(frames[0],cv2.COLOR_GRAY2BGR)
  warm_start=time.perf_counter();infer_pair(model,first_color,first_color,alpha);warmup_s=time.perf_counter()-warm_start
  reference_output=None
  for i,frame in enumerate(frames):
   a=frames[0].astype(np.float32)/255;b=frame.astype(np.float32)/255
   for name,fn in [('classical',estimate_pair),('phase',phase_pair),('learned',learned_pair)]:
    tick=time.perf_counter();d=fn(a,b);timings[name]+=time.perf_counter()-tick
    estimates[name].append(d['dx']);qualities[name].append(d['quality'])
   rgb=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
   first=cv2.cvtColor(frames[0],cv2.COLOR_GRAY2BGR)
   tick=time.perf_counter();enhanced=infer_pair(model,first,rgb,alpha);timings['EulerMormer']+=time.perf_counter()-tick
   gray=cv2.cvtColor(enhanced,cv2.COLOR_BGR2GRAY)
   if reference_output is None: reference_output=gray
   output_fit=estimate_pair(reference_output,gray)
   estimates['EulerMormer'].append(output_fit['dx']/alpha);qualities['EulerMormer'].append(output_fit['quality'])
  for name,values in estimates.items():
   v=np.array(values);v=v-v.mean();target=truth-truth.mean()
   analysis=analyze_signal(v,fps,4,12)['metrics']
   rows.append({'seed':seed,'noise_std_0to1':noise,'method':name,'displacement_rmse_px':float(np.sqrt(np.mean((v-target)**2))),
    'frequency_hz':analysis['frequency_hz'],'frequency_error_hz':None if analysis['frequency_hz'] is None else abs(analysis['frequency_hz']-f0),
    'amplitude_px':analysis['amplitude_px'],'amplitude_abs_error_px':None if analysis['amplitude_px'] is None else abs(analysis['amplitude_px']-.25),
    'inference_fps':n/timings[name],'elapsed_inference_s':timings[name],
    'tracking_quality_median':float(np.median(qualities[name])),'tracking_rejected_fraction':float(np.mean(np.asarray(qualities[name])<.25)),
    'tracking_status':'valid' if np.isfinite(qualities[name]).all() and np.all(np.asarray(qualities[name])>=.25) else 'invalid',
    'neural_warmup_seconds':warmup_s if name=='EulerMormer' else None})
  curves.append({'seed':seed,'noise_std_0to1':noise,'truth_px':truth.tolist(),'estimates_px':estimates,'tracking_quality':qualities})
  (out/'comparison.json').write_text(json.dumps({'rows':rows,'status':'running'},indent=2))
  print(seed,noise,'finished',flush=True)
report={'status':'complete','evidence':'controlled synthetic pixel translations; no physical experiment','source_commit':COMMIT,
 'model_sha256':SOURCE_SHA256,'weights_sha256':WEIGHT_SHA256,'seeds':[91341,91342],
 'fps':fps,'frames':n,'shape':[64,64],'true_frequency_hz':f0,'true_amplitude_px':.25,'requested_total_gain':alpha,
 'noise_std_0to1':[0,.03],'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),'threads':1,
 'model_parameters':sum(p.numel() for p in model.parameters()),'source_hashes':source_hashes,
 'source_changed_during_run':any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h for p,h in source_hashes.items()),
 'measurement':'Classical, phase and learned operate on original frames. EulerMormer is measured by original physical tracker on rendered outputs then divided by requested gain. Reconstruction artifacts and tracker error are included.',
 'limits':['Tiny synthetic patches only; no evidence of comparative performance on real structures or full-resolution videos.',
 'EulerMormer is static first-frame mode without temporal filtering; baseline is not full paper reproduction.',
 'Model inference FPS excludes per-case warmup, output tracking and all decoding/encoding. Original estimator FPS excludes video IO.',
 'Tracking quality is measured on every frame; invalid sequences must not be interpreted as valid displacement. Rejection uses quality<0.25, every frame must be finite with quality>=0.25; any rejected frame invalidates the sequence.'],
 'rows':rows}
(out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
(out/'curves.json').write_text(json.dumps(curves))
if report['source_changed_during_run']:raise SystemExit('Source changed during inference; rerun or refresh after inspecting the changed sources.')
print(json.dumps(report,indent=2))
