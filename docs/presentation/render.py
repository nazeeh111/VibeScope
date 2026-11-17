"""Render actual paired-gain exports; never run an estimator or modify source data."""
from pathlib import Path
import json,hashlib,xml.etree.ElementTree as ET,re,sys
import numpy as np,cv2,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
read=lambda n:json.loads((OUT/n).read_text());a=read('gain5-report.json');b=read('gain20-report.json')
assert a['signal']==b['signal'] and a['metrics']==b['metrics']
assert a['metadata']['source_sha256']==b['metadata']['source_sha256']
frames={};inputs={}
previous=json.loads((OUT/'manifest.json').read_text()) if (OUT/'manifest.json').exists() else {}
for kind in ['original','enhanced']:
 source=OUT/f'{kind}-frame72.png'
 frame=cv2.imread(str(source));assert frame is not None and frame.shape[:2]==(192,320)
 rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);frames[kind]=rgb
 prior=previous.get('frame_inputs',{}).get(kind,{})
 inputs[kind]={**prior,'path':source.name,'frame_index':72,'native_shape':[192,320,3],'png_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'decoded_rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest()}
s=b['signal'];t=np.asarray(s['time_s']);raw=np.asarray(s['raw_px']);f=np.asarray(s['frequency_hz']);psd=np.asarray(s['psd_px2_hz']);assert len(t)==len(raw)==240 and np.isfinite(raw).all() and np.isfinite(psd).all() and np.all(psd[1:]>0)
for mobile in [False,True]:
 with plt.rc_context({'font.family':['DejaVu Sans','Arial','sans-serif'],'font.size':12 if not mobile else 14,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','path.simplify':False}):
  fig=plt.figure(figsize=(12,8.1) if not mobile else (6.4,13.6),layout='constrained',facecolor='white')
  grid=fig.add_gridspec(2,2) if not mobile else fig.add_gridspec(4,1)
  for i,kind in enumerate(['original','enhanced']):
   ax=fig.add_subplot(grid[0,i] if not mobile else grid[i,0]);ax.imshow(frames[kind],interpolation='nearest');ax.axis('off');ax.set_title('Original decoded frame' if kind=='original' else 'Gain 20 visualization only',loc='left',pad=10)
  ax=fig.add_subplot(grid[1,0] if not mobile else grid[2,0]);l=ax.plot(t,np.asarray(a['signal']['raw_px']),color='#12615b',lw=2,label='Gain5')[0];l.set_gid('gain5-signal');l=ax.plot(t,raw,color='#263638',lw=1.2,ls='--',label='Gain20 (same values)')[0];l.set_gid('gain20-signal');ax.set(xlabel='Capture time (s)',ylabel='Displacement (px)',title='Gain 5 and gain 20: same measurement',xlim=(float(t[0]),float(t[-1])));
  ax=fig.add_subplot(grid[1,1] if not mobile else grid[3,0]);l=ax.semilogy(f[1:],psd[1:],color='#12615b',lw=1.5)[0];l.set_gid('psd-source');ax.axvspan(4,12,color='#12615b',alpha=.10);ax.set(xlabel='Frequency (Hz)',ylabel='PSD (px²/Hz, log scale)',title='One spectrum from original frames',xlim=(float(f[1]),float(f[-1])))
  fig.suptitle('Magnify the view. Keep the measurement.',fontsize=19 if not mobile else 17,fontweight='bold')
  stem='measurement-story-mobile' if mobile else 'measurement-story';fig.savefig(OUT/f'{stem}.svg');fig.savefig(OUT/f'{stem}.png',dpi=160);plt.close(fig)
  svg=(OUT/f'{stem}.svg').read_text()
  # Matplotlib mathtext tspans otherwise retain a single-family declaration.
  svg=re.sub(r'font-family:\s*[^;"\n]+', "font-family: 'DejaVu Sans', Arial, sans-serif", svg)
  svg=svg.replace('<text>', '<text style="font-family: \'DejaVu Sans\', Arial, sans-serif">')
  (OUT/f'{stem}.svg').write_text(svg)
  counts={}
  for gid,n in [('gain5-signal',240),('gain20-signal',240),('psd-source',len(f)-1)]:
   d=re.search('id="'+gid+'"[\\s\\S]*?<path d="([^"]+)"',svg);assert d,gid;count=len(re.findall('[ML]',d[1]));assert count==n,(gid,count,n);counts[gid]=count
  inputs[stem]={'encoded_vertices':counts,'png_sha256':hashlib.sha256((OUT/f'{stem}.png').read_bytes()).hexdigest(),'svg_sha256':hashlib.sha256((OUT/f'{stem}.svg').read_bytes()).hexdigest()}
manifest={'source_commit':'db2594d6ff15d75151785c388ed35c473533d3b9','source_tree':'0386b5004fabaa1408c455c50b47c8613b9a7675','kind':'new actual controlled-synthetic browser measurements, gains5and20, same source','source_sha256':b['metadata']['source_sha256'],'software_version':b['metadata']['software_version'],'signal_arrays_equal':True,'metrics_equal':True,'frame_inputs':inputs,'transformations':['Native320×192 decoded frame72 from each actual run output; no pixel adjustments. Nearest-neighbor enlargement is display only.','All240 stored unfiltered displacement samples from both gains retained without simplification.','Spectrum uses all positive-frequency stored bins; DC bin explicitly excluded from logarithmic plot. No new fit, filtering, smoothing or numerical normalization.','Video gain is already the actual application visualization; it never scales the measured displacement.'],'limits':['Controlled synthetic source; no physical calibration, arbitrary-media or diagnostic-accuracy claim.','The report frequency resolution is not a calibrated confidence interval.'],'font_policy':['DejaVu Sans','Arial','sans-serif'],'svg_font_policy':'Explicit DejaVu Sans/Arial/sans-serif fallback on text elements, mathtext parents and tspans; numeric geometry unchanged','runtime':{'python':sys.version,'numpy':np.__version__,'matplotlib':matplotlib.__version__,'opencv':cv2.__version__},'input_hashes':{n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['gain5-report.json','gain20-report.json','gain5-displacement.csv','gain20-displacement.csv']}}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'signal_samples':240,'spectrum_bins':len(f)-1,'verified_vertices':True,'metrics':b['metrics']},indent=2))
