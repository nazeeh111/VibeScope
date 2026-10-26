"""Tests use analytic textures unlike the Fourier-translated training generator."""
import numpy as np
import pytest
from vibescope.learned import estimate_pair, baseline_pair, DEFAULT_MODEL


def analytic_pair(dx, dy, noise=0, seed=913):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:48, :48]
    def field(xx, yy):
        return (0.5 + .10 * np.sin(.41 * xx + .18 * yy) + .12 * np.cos(.2 * xx - .37 * yy)
                + .08 * np.cos(.53 * xx + .57 * yy))
    a = field(x, y) + rng.normal(0, noise, x.shape)
    b = field(x - dx, y - dy) + rng.normal(0, noise, x.shape)
    return np.clip(a, 0, 1), np.clip(b, 0, 1)


@pytest.mark.parametrize('dx,dy',[(.35,-.2),(-.45,.65),(.0,.0),(.8,-.75)])
def test_displacement_direction_and_accuracy(dx,dy):
    a,b=analytic_pair(dx,dy)
    result=estimate_pair(a,b)
    assert np.linalg.norm(np.array([result['dx'],result['dy']])-[dx,dy]) < .055
    assert result['quality'] > .6
    assert result['within_training_domain']


def test_noise_and_lighting_offset():
    a,b=analytic_pair(.25,-.35,noise=.025)
    result=estimate_pair(a,b+.01)
    assert np.linalg.norm(np.array([result['dx'],result['dy']])-[.25,-.35]) < .09
    assert 0 < result['sigma_px'] < 1
    assert result['uncertainty_kind']=='synthetic_error_scale_not_confidence_interval'


def test_textureless_region_rejected():
    a=np.ones((40,40))*.5
    result=estimate_pair(a,a)
    assert result['quality']==0
    assert not result['within_training_domain']


def test_aperture_region_rejected():
    a=np.tile(.5+.1*np.sin(np.arange(40)*.4),(40,1))
    result=estimate_pair(a,a)
    assert result['quality']==0
    assert not result['within_training_domain']


def test_baseline_has_no_model_dependency():
    a,b=analytic_pair(.3,.4)
    result=baseline_pair(a,b)
    assert abs(result['dx']-.3)<.03
    assert abs(result['dy']-.4)<.03


def test_input_contract():
    a,b=analytic_pair(0,0)
    with pytest.raises(ValueError): estimate_pair(a*255,b)
    with pytest.raises(ValueError): estimate_pair(a[:10],b[:10])
    with pytest.raises(ValueError): estimate_pair(a,b,device='cuda')
    a[0,0]=np.nan
    with pytest.raises(ValueError): estimate_pair(a,b)


def test_weights_are_small_portable_array_file():
    assert DEFAULT_MODEL.stat().st_size < 20000
    with np.load(DEFAULT_MODEL,allow_pickle=False) as z:
        assert z['w1'].shape==(12,24)
        assert z['w2'].shape==(24,2)
