"""Reference screening tests use recordings withheld from center/threshold fitting."""
import json
import numpy as np
import pytest
from vibescope.anomaly import features,train_reference,score,load_reference


def recording(seed,frequency=7.3,amplitude=.25,harmonic=0):
    rng=np.random.default_rng(seed);t=np.arange(240)/60
    return amplitude*np.sin(2*np.pi*(frequency+rng.normal(0,.025))*t+rng.uniform(0,6.28))+harmonic*np.sin(4*np.pi*frequency*t)+rng.normal(0,.006,len(t))


def reference():
    return train_reference([recording(i) for i in range(1000,1100)],
                           [recording(i) for i in range(2000,2100)],60.,'Independent synthetic acquisition A')


def test_independent_healthy_and_changed_recordings():
    model=reference()
    healthy=[score(recording(i),60.,model)['status'] for i in range(3000,3020)]
    changed=[score(recording(i,frequency=9.1),60.,model)['status'] for i in range(4000,4020)]
    assert healthy.count('within_reference')>=16
    assert changed.count('outside_reference')==20
    assert model['training_records']==100 and model['calibration_records']==100


def test_amplitude_and_harmonic_changes_are_reference_novelty():
    model=reference()
    for values in [recording(5010,amplitude=.7),recording(5011,harmonic=.12)]:
        result=score(values,60.,model)
        assert result['status']=='outside_reference'
        assert result['diagnostic'] is False
        assert result['reference_scope']=='Independent synthetic acquisition A'
        assert 'damage' in result['explanation']


def test_flat_input_cannot_produce_reference_assessment():
    with pytest.raises(ValueError,match='observable'):
        features(np.zeros(240),60.)


def test_reference_round_trip_preserves_score(tmp_path):
    model=reference();path=tmp_path/'reference.json';path.write_text(json.dumps(model))
    before=score(recording(6000),60.,model)
    after=score(recording(6000),60.,load_reference(path))
    assert after==before


def test_reference_rejects_incompatible_acquisition():
    model=reference()
    with pytest.raises(ValueError,match='fps'):
        score(recording(6200),120.,model)
    with pytest.raises(ValueError,match='length'):
        score(recording(6200)[:120],60.,model)
    with pytest.raises(ValueError,match='scope'):
        score(recording(6200),60.,model,confirmed_scope='Different machine and camera')
    assert score(recording(6200),60.,model,confirmed_scope=model['scope'])['acquisition_match']=='user_confirmed'


def test_invalid_reference_and_inadequate_calibration_are_rejected(tmp_path):
    from vibescope.anomaly import validate_reference
    model=reference();model['scale'][0]=0
    with pytest.raises(ValueError):validate_reference(model)
    with pytest.raises(ValueError,match='at least'):
        train_reference([recording(i) for i in range(9)],[recording(i) for i in range(5)],60.,'scope')
    for values in [np.full(240,np.nan),np.arange(20),np.ones((40,2))]:
        with pytest.raises(ValueError):features(values,60.)
    invalid=tmp_path/'bad.json';invalid.write_text('{bad json')
    with pytest.raises(ValueError):load_reference(invalid)
