"""Conservative original-signal spectral resolution and free-ringdown fitting.

The resolver reports a dominant single tone only when it is an interior spectral
peak, >=15 dB above median PSD, and explains >=15% of original detrended variance.
These are engineering gates, not a calibrated false-alarm probability. Weak modes
buried beneath strong out-of-band motion can remain unresolved by this policy.
"""
from __future__ import annotations
import numpy as np
from scipy import signal, stats, optimize


def bandpass(values, fps, low, high):
    values = np.asarray(values, float)
    if values.ndim != 1:
        raise ValueError('Use a one-dimensional displacement signal.')
    if not np.isfinite([fps, low, high]).all() or fps <= 0 or not 0 < low < high < fps/2:
        raise ValueError('Choose 0 < lower frequency < upper frequency < Nyquist (fps ÷ 2).')
    if len(values) < 40 or not np.isfinite(values).all():
        raise ValueError('At least 40 finite samples are required.')
    if len(values)/fps * low < 2:
        raise ValueError('Recording is too short: capture at least two cycles of the lower band frequency.')
    sos = signal.butter(3, [low, high], btype='bandpass', fs=fps, output='sos')
    return signal.sosfiltfilt(sos, values)


def fit_damping(values, fps, frequency):
    """Only valid for an isolated free ringdown, not forced/ambient vibration."""
    centered = signal.detrend(np.asarray(values, float), type='constant')
    indices, _ = signal.find_peaks(centered, distance=max(2,int(fps/frequency*.75)), prominence=np.std(centered)*.15)
    if len(indices) < 6:
        return None, 'Too few positive ringdown peaks for damping.'
    times, peaks = indices/fps, centered[indices]
    keep = peaks > max(np.max(peaks)*.12, 1e-8)
    times, peaks = times[keep], peaks[keep]
    if len(times) < 6:
        return None, 'Too few peaks above the ringdown noise tail.'
    result = stats.linregress(times, np.log(peaks))
    decay = -result.slope
    r2 = result.rvalue**2
    if decay <= 0 or r2 < .90:
        return None, 'Signal does not support a single exponentially decaying mode (R² < 0.90).'
    ratio = decay/np.sqrt((2*np.pi*frequency)**2+decay**2)
    natural=float(np.hypot(frequency,decay/(2*np.pi)))
    resonance=natural*np.sqrt(1-2*ratio**2) if ratio<np.sqrt(.5) else None
    return dict(ratio=float(ratio),decay_per_s=float(decay),envelope_r2=float(r2),peaks=len(peaks),
                natural_frequency_hz=natural,sdof_resonance_hz_model=float(resonance) if resonance is not None else None,
                resonance_assumptions='Conditional force-to-displacement SDOF model; excitation and resonance are not experimentally verified.',
                assumptions='User-declared free ringdown; isolated linear mode; no forcing.'), None


def analyze_signal(values, fps, low, high, ringdown=False):
    values = np.asarray(values, float)
    filtered = bandpass(values, fps, low, high)
    raw = signal.detrend(values)
    freq, psd = signal.periodogram(raw, fs=fps, window='hann', detrend=False, scaling='density')
    mask = (freq >= low) & (freq <= high)
    if not mask.any():
        raise ValueError('Frequency band contains no resolvable FFT bins. Use a longer recording or wider band.')
    resolution = fps/len(raw)
    t = np.arange(len(raw))/fps
    total_energy=float(np.sum(raw**2))
    floor=float(np.median(psd[1:]))
    candidates,_=signal.find_peaks(psd)
    # Require peak and both neighbours inside the band. Never turn a clipped
    # monotonic leakage tail at the band edge into a new frequency measurement.
    candidates=[i for i in candidates if 0<i<len(psd)-1 and mask[i-1:i+2].all()]
    candidates=sorted(candidates,key=lambda i:psd[i],reverse=True)
    peak_hz=amplitude=None;prominence=0.;explained=0.;residual=raw
    for i in candidates:
        prominence_i=float(10*np.log10(max(psd[i],1e-30)/max(floor,1e-30)))
        if prominence_i<15:continue
        def fit(f):
            design=np.column_stack([np.sin(2*np.pi*f*t),np.cos(2*np.pi*f*t),np.ones(len(t)),t])
            coef=np.linalg.lstsq(design,values,rcond=None)[0]
            error=values-design@coef
            return float(error@error),coef,error
        bounds=(max(low,freq[i]-resolution*.6),min(high,freq[i]+resolution*.6))
        refined=optimize.minimize_scalar(lambda f:fit(f)[0],bounds=bounds,method='bounded',options={'xatol':1e-7})
        energy,coef,error=fit(refined.x)
        explained_i=max(0.,1-energy/max(total_energy,1e-30))
        amp=float(np.hypot(coef[0],coef[1]))
        if explained_i>=.15 and amp>1e-5 and refined.x*len(raw)/fps>=3:
            peak_hz=float(refined.x);amplitude=amp;prominence=prominence_i;explained=explained_i;residual=error
            break
    identifiable=peak_hz is not None
    noise=float(1.4826*np.median(np.abs(residual-np.median(residual))))
    damping,damping_warning=(fit_damping(values,fps,peak_hz) if ringdown and identifiable else (None,None))
    metrics=dict(frequency_hz=peak_hz,frequency_resolution_hz=float(resolution),
                 frequency_interval_hz=[max(low,peak_hz-resolution),min(high,peak_hz+resolution)] if identifiable else None,
                 frequency_interval_kind='± one FFT bin; resolution range, not a calibrated confidence interval',
                 amplitude_px=amplitude,rms_px=float(np.sqrt(np.mean(filtered**2))),
                 noise_floor_px=noise,residual_mad_px=noise,peak_to_median_psd_db=prominence,
                 tone_explained_variance=explained,status='resolved' if identifiable else 'unresolved',
                 resolver_policy={'min_peak_to_median_db':15,'min_full_signal_variance_fraction':.15,'interior_peak_required':True},
                 amplitude_kind='Peak amplitude of best-fit sinusoid to unfiltered original-frame displacement',
                 damping_ratio=damping['ratio'] if damping else None,damping_fit=damping,
                 natural_frequency_hz=damping['natural_frequency_hz'] if damping else None,
                 sdof_resonance_hz_model=damping['sdof_resonance_hz_model'] if damping else None)
    warnings=[]
    if not identifiable:warnings.append('No periodic component passed the resolver gates. Weak selected modes, peaks at band edges, short records and noise may remain unresolved; widen the band or improve acquisition.')
    if high>.8*(fps/2):warnings.append('Band approaches Nyquist; aliasing and exposure attenuation are especially consequential.')
    if damping_warning:warnings.append(damping_warning)
    warnings.append('A spectral peak is a dominant response frequency; resonance requires a controlled excitation or transfer-function experiment.')
    return dict(metrics=metrics,time_s=t,raw_px=values-values[0],filtered_px=filtered,frequency_hz=freq,psd_px2_hz=psd,warnings=warnings)
