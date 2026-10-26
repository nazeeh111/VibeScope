import numpy as np
import pytest
from scipy.ndimage import shift
from vibescope.motion import estimate_pair,track,validate_roi
from vibescope.synthetic import texture,patch_sequence
from vibescope.signals import analyze_signal,bandpass,fit_damping


def test_subpixel_sign_and_brightness_nuisance():
    a=texture(892)
    b=shift(a,(.31,-.42),order=3,mode='reflect')*1.04+.01
    m=estimate_pair(a,b)
    assert abs(m['dx']+.42)<.025
    assert abs(m['dy']-.31)<.025
    assert m['quality']>.8


def test_two_axes_and_independent_camera_subtraction():
    a=texture(188,(72,144)); t=np.arange(160)/60
    target=.25*np.sin(2*np.pi*7.3*t);camera=.18*np.sin(2*np.pi*1.7*t)
    frames=[]
    for d,c in zip(target,camera):
        b=a.copy();b[:,72:]=shift(a[:,72:],(0,d),order=3,mode='reflect')
        frames.append(shift(b,(0,c),order=3,mode='reflect'))
    v,_,_=track(np.array(frames),[80,8,56,56]);r,_,_=track(np.array(frames),[8,8,56,56])
    assert np.sqrt(np.mean((v-r-target)**2))<.025


@pytest.mark.parametrize('axis',['x','y'])
def test_frequency_and_amplitude_against_image_motion(axis):
    frames,truth,_=patch_sequence(seed=965,axis=axis,noise=.005,frequency=7.3,amplitude=.25)
    v,q,_=track(frames,[0,0,64,64],axis=axis)
    result=analyze_signal(v,60,4,12)
    assert abs(result['metrics']['frequency_hz']-7.3)<.05
    assert abs(result['metrics']['amplitude_px']-.25)<.02
    assert np.median(q)>.8


@pytest.mark.parametrize('band',[(0,12),(4,30),(12,4),(2,40),(-1,3)])
def test_nyquist_and_invalid_band_rejected(band):
    with pytest.raises(ValueError):bandpass(np.zeros(180),60,*band)


def test_short_record_and_flat_signal_do_not_issue_frequency():
    with pytest.raises(ValueError):bandpass(np.zeros(30),60,4,12)
    assert analyze_signal(np.zeros(180),60,4,12)['metrics']['frequency_hz'] is None
    assert estimate_pair(np.zeros((32,32)),np.zeros((32,32)))['quality']==0


def test_aperture_patch_is_rejected():
    a=np.tile(np.sin(np.arange(64)*.4),(64,1))*.2+.5
    assert estimate_pair(a,shift(a,(0,.2),order=3))['quality']==0


def test_damping_requires_decay_fit():
    t=np.arange(600)/120;zeta=.012;f=8
    y=np.exp(-2*np.pi*f*zeta*t)*np.sin(2*np.pi*f*t)
    result,warning=fit_damping(y,120,f)
    assert result and abs(result['ratio']-zeta)<.001
    assert result['envelope_r2']>.99
    y=np.sin(2*np.pi*f*t)*(1+.4*np.sin(2*np.pi*.5*t))
    result,_=fit_damping(y,120,f)
    assert result is None


def test_aliasing_cannot_be_recovered_from_sampling():
    t=np.arange(180)/60
    # This failure case is fundamental: 52 Hz sampled at 60 fps aliases to 8 Hz.
    a=analyze_signal(np.sin(2*np.pi*52*t),60,4,12)
    assert abs(a['metrics']['frequency_hz']-8)<.05
    assert any('dominant response' in w for w in a['warnings'])


def test_invalid_roi():
    for roi in [[-1,0,24,24],[0,0,10,24],[0,0,80,30],[.5,0,24,24]]:
        with pytest.raises(ValueError):validate_roi(roi,(64,64))


@pytest.mark.parametrize('outside_hz',[1.,2.,3.,15.,20.])
def test_out_of_band_motion_does_not_become_measurement(outside_hz):
    t=np.arange(180)/60
    result=analyze_signal(.25*np.sin(2*np.pi*outside_hz*t),60,4,12)
    assert result['metrics']['status']=='unresolved'
    assert result['metrics']['amplitude_px'] is None


def test_seeded_noise_does_not_resolve_a_tone():
    for n in (40,60,180,600):
        for seed in range(30):
            noise=np.random.default_rng(seed).normal(0,.01,n)
            result=analyze_signal(noise,60,4,12)
            assert result['metrics']['frequency_hz'] is None,(n,seed)


def test_dominant_mode_frequency_refinement_and_amplitude():
    t=np.arange(180)/60
    y=.25*np.sin(2*np.pi*7.31*t+.3)+.02*np.sin(2*np.pi*17*t)
    m=analyze_signal(y,60,4,12)['metrics']
    assert abs(m['frequency_hz']-7.31)<.003
    assert abs(m['amplitude_px']-.25)<.002


def test_unobservable_peak_samples_are_identified():
    frames,truth,_=patch_sequence(seed=965,noise=.005)
    corrupted=frames.copy()
    missing=np.argsort(np.abs(truth))[-17:]
    corrupted[missing]=.5
    _,quality,_=track(corrupted,[0,0,64,64])
    assert np.all(quality[missing]<.25)


def test_ringdown_natural_and_conditional_resonance():
    t=np.arange(600)/120;fd=8.;decay=.6
    result=analyze_signal(.25*np.exp(-decay*t)*np.sin(2*np.pi*fd*t),120,4,12,ringdown=True)['metrics']
    expected=np.hypot(fd,decay/(2*np.pi))
    assert result['natural_frequency_hz']==pytest.approx(expected,abs=.005)
    assert result['sdof_resonance_hz_model']<result['natural_frequency_hz']
    assert 'not experimentally verified' in result['damping_fit']['resonance_assumptions']
