import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from vibescope import history
from vibescope.server import create_app
from vibescope.synthetic import make_demo


def test_real_record_restart_reopen_and_rerun(tmp_path):
    video, truth=make_demo(tmp_path/'fixture',frames=120)
    original=hashlib.sha256(video.read_bytes()).hexdigest()
    root=tmp_path/'runs'
    with TestClient(create_app(root)) as client:
        with video.open('rb') as stream:
            source=client.post('/api/upload',files={'file':('fixture.mp4',stream,'video/mp4')}).json()
        config={'video_id':source['id'],'roi':truth['roi'],'reference_roi':truth['reference_roi'],
                'axis':'x','low_hz':4,'high_hz':12,'gain':10,'fps_override':60,'assume_cfr':True}
        with patch('vibescope.video.inspect_timing', return_value={'verified':False,'reason':'test unavailable'}):
            response=client.post('/api/analyze',json=config)
        assert response.status_code==200,response.text
        first=response.json(); run=first['id']
    with TestClient(create_app(root)) as client:
        listing=client.get('/api/history').json()
        assert any(r['id']==source['id'] and r['state']=='complete' for r in listing['recordings'])
        listing=client.get('/api/history/'+source['id']).json()
        assert {'id':run,'state':'completed'} in listing['runs']
        opened=client.get('/api/history/'+source['id']+'/'+run)
        assert opened.status_code==200,opened.text
        saved=opened.json(); assert saved['report']==first
        for key,value in config.items():assert saved['parameters'][key]==value
        assert saved['report']['metadata']['source_kind']=='user_recording'
        assert saved['report']['metrics']['amplitude_mm'] is None
        assert saved['report']['metadata']['timing']==first['metadata']['timing']
        assert saved['report']['metadata']['timing']['verified'] is False
        assert any('unverified' in warning.lower() for warning in saved['report']['warnings'])
        cli = subprocess.run([sys.executable, '-m', 'vibescope.cli', 'history', '--data-dir', str(root),
                              '--recording', source['id'], '--run', run], capture_output=True, text=True, check=True)
        assert json.loads(cli.stdout)['parameters']['roi']==config['roi']
        again=client.post('/api/analyze',json={**config,'gain':15})
        assert again.status_code==200,again.text
        assert again.json()['id'] != run
        assert client.get('/api/history/'+source['id']+'/'+run).json()['report']==first
        other=root/('a'*32);other.mkdir()
        (other/'source.json').write_text(json.dumps({'id':other.name,'name':'other','preview_url':f'/data/{other.name}/preview.png','width':320,'height':192,'fps':60}))
        (other/'input.video').write_bytes(b'unchanged');(other/'preview.png').write_bytes(b'preview')
        scan=history.entries;hidden=root/'hidden-source'
        def disappear_recording(parent):
            result=scan(parent)
            if parent==root and other.exists():other.rename(hidden)
            return result
        with patch.object(history,'entries',side_effect=disappear_recording):
            listing=client.get('/api/history')
        assert listing.status_code==200
        assert any(item['id']==source['id'] and item['state']=='complete' for item in listing.json()['recordings'])
        assert any(item['id']==other.name and item['state']=='unavailable' for item in listing.json()['recordings'])
        hidden.rename(other)
        transient=root/source['id']/('b'*32);transient.mkdir();hidden_run=root/'hidden-run'
        def disappear_run(parent):
            result=scan(parent)
            if parent==root/source['id'] and transient.exists():transient.rename(hidden_run)
            return result
        with patch.object(history,'entries',side_effect=disappear_run):
            listing=client.get('/api/history/'+source['id'])
        assert listing.status_code==200
        assert {'id':run,'state':'completed'} in listing.json()['runs']
        assert {'id':transient.name,'state':'unavailable'} in listing.json()['runs']
        hidden_run.rename(transient)
        artifact=root/source['id']/run/'heatmap.png'
        artifact.write_bytes(b'changed')
        assert client.get('/api/history/'+source['id']+'/'+run).status_code==404
        assert {'id':run,'state':'partial'} in client.get('/api/history/'+source['id']).json()['runs']
    assert hashlib.sha256(video.read_bytes()).hexdigest()==original


def test_partial_incompatible_symlink_and_concurrent_manifest(tmp_path):
    folder=tmp_path/('a'*32);folder.mkdir()
    (folder/'source.json').write_text(json.dumps({'id':'a'*32,'name':'legacy','preview_url':f'/data/{"a"*32}/preview.png','width':320,'height':192,'fps':60}))
    (folder/'input.video').write_bytes(b'input'); (folder/'preview.png').write_bytes(b'preview')
    partial=folder/('b'*32);partial.mkdir();(partial/'report.json').write_text('{}')
    newer=folder/('c'*32);newer.mkdir();(newer/'run-manifest.json').write_text('{"schema":"vibescope.run.v9"}')
    outside=tmp_path/'outside';outside.mkdir();(folder/('d'*32)).symlink_to(outside,target_is_directory=True)
    with TestClient(create_app(tmp_path)) as client:
        result=client.get('/api/history/'+folder.name).json()
        assert result['state']=='legacy'
        assert result['runs']==[{'id':'b'*32,'state':'partial'},{'id':'c'*32,'state':'incompatible'}]
        assert client.get('/api/history/'+folder.name+'/'+partial.name).status_code==404
        assert client.get('/api/history/'+folder.name+'/'+newer.name).status_code==409
        assert client.get('/api/history/'+folder.name+'/'+('d'*32)).status_code==404
        assert client.get('/api/history/not-a-uuid').status_code==404
    target=tmp_path/'publication';target.mkdir()
    def save(i):
        try:history.publish(target,'manifest.json',{'value':i});return 'ok'
        except FileExistsError:return 'exists'
    with ThreadPoolExecutor(max_workers=2) as pool:outcomes=list(pool.map(save,range(2)))
    assert sorted(outcomes)==['exists','ok']
    assert json.loads((target/'manifest.json').read_text())['value'] in (0,1)


def test_changed_source_and_interrupted_publication_are_not_completed(tmp_path):
    folder=tmp_path/('a'*32);folder.mkdir()
    (folder/'source.json').write_text(json.dumps({'id':folder.name,'name':'source','preview_url':f'/data/{folder.name}/preview.png','width':320,'height':192,'fps':60}))
    (folder/'input.video').write_bytes(b'original');(folder/'preview.png').write_bytes(b'preview')
    history.source_manifest(folder)
    assert history.check_source(folder)[1]=='complete'
    (folder/'input.video').write_bytes(b'changed!')
    with pytest.raises(ValueError,match='changed'):history.check_source(folder)
    run=folder/('b'*32);run.mkdir()
    with patch('vibescope.history.os.link',side_effect=OSError('interrupted')):
        with pytest.raises(OSError):history.publish(run,'run-manifest.json',{'schema':'vibescope.run.v1'})
    assert not (run/'run-manifest.json').exists()
    assert history.classify_run(run,folder)=='partial'


def test_listing_is_bounded_and_ignores_invalid_or_linked_names(tmp_path):
    for index in range(history.MAX_ENTRIES+5):
        (tmp_path/f'{index:032x}').mkdir()
    outside=tmp_path/'outside';outside.mkdir()
    (tmp_path/('f'*32)).symlink_to(outside,target_is_directory=True)
    items=history.entries(tmp_path)
    assert len(items)<=history.MAX_ENTRIES
    assert 'f'*32 not in items


def test_missing_new_source_manifest_is_partial_but_legacy_remains_reopenable(tmp_path):
    for name, schema in [('a'*32, 'vibescope.source.v1'), ('b'*32, None), ('c'*32, 'vibescope.source.v9')]:
        folder=tmp_path/name;folder.mkdir()
        (folder/'source.json').write_text(json.dumps({'id':name,'name':'recording','history_schema':schema,'preview_url':f'/data/{name}/preview.png','width':320,'height':192,'fps':60}))
        (folder/'input.video').write_bytes(b'input');(folder/'preview.png').write_bytes(b'preview')
    with TestClient(create_app(tmp_path)) as client:
        states={row['id']:row['state'] for row in client.get('/api/history').json()['recordings']}
        assert states=={'a'*32:'partial','b'*32:'legacy','c'*32:'incompatible'}
        assert client.get('/api/history/'+'a'*32).status_code==404
        assert client.get('/api/history/'+'b'*32).status_code==200
        assert client.get('/api/history/'+'c'*32).status_code==409


def test_demo_refuses_linked_folder_without_touching_target(tmp_path):
    target=tmp_path/'other';target.mkdir();(target/'input.video').write_bytes(b'untouched')
    root=tmp_path/'runs';root.mkdir();(root/('f'*32)).symlink_to(target,target_is_directory=True)
    with TestClient(create_app(root)) as client:
        assert client.get('/api/demo').status_code==409
    assert (target/'input.video').read_bytes()==b'untouched'


def test_deeply_nested_corruption_does_not_hide_healthy_sibling(tmp_path):
    for letter in ('a','b'):
        folder=tmp_path/(letter*32);folder.mkdir()
        (folder/'source.json').write_text(json.dumps({'id':folder.name,'name':letter,
            'preview_url':f'/data/{folder.name}/preview.png','width':320,'height':192,'fps':60}))
        (folder/'input.video').write_bytes(b'unchanged');(folder/'preview.png').write_bytes(b'preview')
    run=tmp_path/('a'*32)/('c'*32);run.mkdir()
    (run/'run-manifest.json').write_text('{"value":'+'['*20000+'0'+']'*20000+'}')
    with TestClient(create_app(tmp_path)) as client:
        assert {row['id'] for row in client.get('/api/history').json()['recordings']} == {'a'*32,'b'*32}
        history_result=client.get('/api/history/'+'a'*32)
        assert history_result.status_code==200
        assert history_result.json()['runs']==[{'id':'c'*32,'state':'partial'}]
        assert client.get('/api/history/'+'a'*32+'/'+'c'*32).status_code==404
        assert client.get('/api/history/'+'b'*32).status_code==200
    assert (tmp_path/('a'*32)/'input.video').read_bytes()==b'unchanged'


@pytest.mark.parametrize('schema', [None, '', 1, []])
def test_invalid_manifest_versions_are_partial(tmp_path, schema):
    for letter in ('a', 'b'):
        folder = tmp_path / (letter * 32)
        folder.mkdir()
        (folder / 'source.json').write_text(json.dumps({
            'id': folder.name, 'name': letter,
            'preview_url': f'/data/{folder.name}/preview.png',
            'width': 320, 'height': 192, 'fps': 60,
        }))
        (folder / 'input.video').write_bytes(b'unchanged')
        (folder / 'preview.png').write_bytes(b'preview')
    manifest = {} if schema is None else {'schema': schema}
    (tmp_path / ('a' * 32) / 'source-manifest.json').write_text(json.dumps(manifest))
    run = tmp_path / ('b' * 32) / ('c' * 32)
    run.mkdir()
    (run / 'run-manifest.json').write_text(json.dumps(manifest))
    with TestClient(create_app(tmp_path)) as client:
        states = {row['id']: row['state'] for row in client.get('/api/history').json()['recordings']}
        assert states == {'a' * 32: 'partial', 'b' * 32: 'legacy'}
        assert client.get('/api/history/' + 'a' * 32).status_code == 404
        response = client.get('/api/history/' + 'b' * 32)
        assert response.status_code == 200
        assert response.json()['runs'] == [{'id': 'c' * 32, 'state': 'partial'}]
        assert client.get('/api/history/' + 'b' * 32 + '/' + 'c' * 32).status_code == 404
    assert (tmp_path / ('a' * 32) / 'input.video').read_bytes() == b'unchanged'


def test_explicit_future_run_version_remains_incompatible(tmp_path):
    run = tmp_path / ('c' * 32)
    run.mkdir()
    (run / 'run-manifest.json').write_text(json.dumps({'schema': 'vibescope.run.v9'}))
    assert history.classify_run(run, tmp_path) == 'incompatible'


def test_verified_json_uses_same_bytes_for_hash_and_parse(tmp_path):
    source=tmp_path/('a'*32);source.mkdir()
    metadata={'id':source.name,'name':'source','preview_url':f'/data/{source.name}/preview.png',
              'width':320,'height':192,'fps':60,'history_schema':'vibescope.source.v1'}
    (source/'source.json').write_text(json.dumps(metadata));(source/'input.video').write_bytes(b'original');(source/'preview.png').write_bytes(b'preview')
    history.source_manifest(source)
    run=source/('b'*32);run.mkdir()
    refs=tmp_path/'references';refs.mkdir(); reference='c'*32
    (refs/(reference+'.json')).write_text('{"scope":"original"}')
    params={'video_id':source.name,'roi':[1,2,24,24],'baseline_id':reference}
    urls={key:f'/data/{source.name}/{run.name}/{name}' for key,name in {'original':'original.mp4','enhanced':'enhanced.mp4','heatmap':'heatmap.png','csv':'displacement.csv','plot':'measurement.png','report':'report.json'}.items()}
    report={'id':run.name,'metadata':{'source_sha256':hashlib.sha256(b'original').hexdigest()},'parameters':{'roi':params['roi']},'artifacts':urls,'metrics':{'frequency':7.3}}
    (run/'report.json').write_text(json.dumps(report))
    for name in history.ARTIFACTS:
        if name!='report.json':(run/name).write_bytes(name.encode())
    history.run_manifest(run,source,params,tmp_path)
    original_capture=history.read_bounded
    def exercise(folder,name,replacement,action):
        path=folder/name;original=path.read_bytes()
        def transient(target,filename,limit=history.MAX_JSON):
            if target==folder and filename==name:
                path.write_bytes(replacement)
                try:return original_capture(target,filename,limit)
                finally:path.write_bytes(original)
            return original_capture(target,filename,limit)
        with patch.object(history,'read_bounded',side_effect=transient):
            with pytest.raises(ValueError):action()
        assert path.read_bytes()==original
    exercise(source,'source.json',json.dumps({**metadata,'fps':120}).encode(),lambda:history.check_source(source))
    exercise(run,'report.json',json.dumps({**report,'metrics':{'frequency':987}}).encode(),lambda:history.check_run(run,source))
    exercise(refs,reference+'.json',b'{"scope":"forged"}',lambda:history.check_run(run,source))
    assert history.check_source(source)[0]['fps']==60
    assert history.check_run(run,source)[0]['metrics']['frequency']==7.3
    assert (source/'input.video').read_bytes()==b'original'
    import os
    if hasattr(os, 'mkfifo'):
        for path, operation in ((source/'source.json', lambda:history.check_source(source)),
                                (run/'report.json', lambda:history.check_run(run,source)),
                                (refs/(reference+'.json'), lambda:history.check_run(run,source))):
            saved=path.read_bytes(); path.unlink(); os.mkfifo(path)
            try:
                with pytest.raises((ValueError,OSError)):
                    operation()
            finally:
                path.unlink();path.write_bytes(saved)
        assert history.check_run(run,source)[0]['metrics']['frequency']==7.3


@pytest.mark.skipif(not hasattr(__import__('os'), 'mkfifo'), reason='FIFO unavailable')
def test_fifo_sources_reports_and_references_are_bounded_and_fds_close(tmp_path):
    import os
    folder=tmp_path/('a'*32);folder.mkdir()
    names=('source.json','report.json','reference.json')
    for name in names:os.mkfifo(folder/name)
    # Run the formerly blocking operation in a killable child as the regression gate.
    code='''import sys
from pathlib import Path
from vibescope.history import read_bounded
for name in sys.argv[2:]:
 try: read_bounded(Path(sys.argv[1]),name)
 except (ValueError,OSError): pass
 else: raise AssertionError(name)
'''
    child=subprocess.run([sys.executable,'-c',code,str(folder),*names],capture_output=True,text=True,timeout=5)
    assert child.returncode==0,child.stderr
    before=len(os.listdir('/dev/fd'))
    for _ in range(50):
        for name in names:
            with pytest.raises((ValueError,OSError)):
                history.read_bounded(folder,name)
    assert len(os.listdir('/dev/fd'))==before
    # A FIFO at the source path is partial and cannot hide an intact sibling.
    healthy=tmp_path/('b'*32);healthy.mkdir()
    (healthy/'source.json').write_text(json.dumps({'id':healthy.name,'name':'healthy','preview_url':f'/data/{healthy.name}/preview.png','width':320,'height':192,'fps':60}))
    (healthy/'input.video').write_bytes(b'unchanged');(healthy/'preview.png').write_bytes(b'preview')
    with TestClient(create_app(tmp_path)) as client:
        rows=client.get('/api/history').json()['recordings']
        assert any(x['id']==folder.name and x['state']=='partial' for x in rows)
        assert any(x['id']==healthy.name and x['state']=='legacy' for x in rows)
        assert client.get('/api/history/'+folder.name).status_code==404
    assert (healthy/'input.video').read_bytes()==b'unchanged'


@pytest.mark.skipif(not hasattr(__import__('os'), 'mkfifo'), reason='FIFO unavailable')
def test_replaced_leaf_after_open_uses_owned_descriptor(tmp_path):
    import os
    folder=tmp_path/'data';folder.mkdir();path=folder/'source.json';path.write_bytes(b'original')
    opened=history.os.open
    def replace_after_open(name, flags, *args, **kwargs):
        fd=opened(name,flags,*args,**kwargs)
        if name=='source.json':
            path.rename(folder/'preserved.json')
            os.mkfifo(path)
        return fd
    with patch.object(history.os,'open',side_effect=replace_after_open):
        assert history.read_bounded(folder,'source.json')==b'original'
    with pytest.raises((ValueError,OSError)):
        history.read_bounded(folder,'source.json')
    assert (folder/'preserved.json').read_bytes()==b'original'
