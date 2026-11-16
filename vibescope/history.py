"""Bounded local recording and run discovery, with immutable receipts."""
from __future__ import annotations
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from pathlib import Path

ID = re.compile(r'^[a-f0-9]{32}$')
ARTIFACTS = ('report.json', 'original.mp4', 'enhanced.mp4', 'heatmap.png', 'displacement.csv', 'measurement.png')
MAX_JSON = 8 * 1024 * 1024
MAX_FILE = 256 * 1024 * 1024
MAX_ENTRIES = 100


def child(parent, identity):
    if not isinstance(identity, str) or not ID.fullmatch(identity):
        raise ValueError('Invalid recording or run identifier')
    path = parent / identity
    if path.is_symlink() or not path.is_dir() or path.resolve().parent != parent.resolve():
        raise ValueError('Recording or run is unavailable')
    return path


def regular(folder, name, limit=MAX_FILE):
    path = folder / name
    if path.is_symlink() or not path.is_file() or path.resolve().parent != folder.resolve():
        raise ValueError('Missing or linked file: ' + name)
    if path.stat().st_size > limit:
        raise ValueError('File exceeds limit: ' + name)
    return path


def open_regular(folder, name, limit):
    """Open a regular child once; reject symlinks at the directory and leaf."""
    if Path(name).name != name:
        raise ValueError('Invalid file name')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    directory = os.open(folder, flags | getattr(os, 'O_DIRECTORY', 0))
    try:
        # Opening a FIFO read-only can block before fstat can reject it.
        fd = os.open(name, flags | getattr(os, 'O_NONBLOCK', 0), dir_fd=directory)
    finally:
        os.close(directory)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError('File missing, non-regular or exceeds limit: ' + name)
        return os.fdopen(fd, 'rb')
    except BaseException:
        os.close(fd)
        raise


def read_bounded(folder, name, limit=MAX_JSON):
    with open_regular(folder, name, limit) as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('File exceeds limit: ' + name)
    return raw


def parse_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def invalid(_):
        raise ValueError('Non-finite JSON value')
    def number(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError('Non-finite JSON value')
        return value
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid, parse_float=number)
    except RecursionError as error:
        raise ValueError('JSON nesting exceeds the supported limit') from error
    if type(value) is not dict:
        raise ValueError('Expected object')
    return value


def read_json(folder, name):
    return parse_json(read_bounded(folder, name))


def digest(folder, name):
    h = hashlib.sha256()
    size = 0
    with open_regular(folder, name, MAX_FILE) as stream:
        while True:
            chunk = stream.read(min(1024 * 1024, MAX_FILE + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_FILE:
                raise ValueError('File exceeds limit: ' + name)
            h.update(chunk)
    return h.hexdigest()


def publish(folder, name, value):
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode() + b'\n'
    if len(raw) > MAX_JSON:
        raise ValueError('Manifest exceeds limit')
    fd, temp = tempfile.mkstemp(prefix='.history-', dir=folder)
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(raw); out.flush(); os.fsync(out.fileno())
        os.link(temp, folder / name)
    finally:
        os.unlink(temp)


def source_manifest(folder):
    publish(folder, 'source-manifest.json', {'schema': 'vibescope.source.v1',
        'files': {name: digest(folder, name) for name in ('source.json', 'input.video', 'preview.png')}})


def run_manifest(folder, source_folder, parameters, root):
    source_hash = digest(source_folder, 'input.video')
    report_bytes = read_bounded(folder, 'report.json')
    report = parse_json(report_bytes)
    if report.get('metadata', {}).get('source_sha256') != source_hash:
        raise ValueError('Source changed during measurement')
    baseline = None
    if parameters.get('baseline_id'):
        identity = parameters['baseline_id']
        if not ID.fullmatch(identity):
            raise ValueError('Invalid reference identifier')
        refs = root / 'references'
        if refs.is_symlink() or not refs.is_dir():
            raise ValueError('Reference unavailable')
        baseline = {'id': identity, 'sha256': digest(refs, identity + '.json')}
    publish(folder, 'run-manifest.json', {'schema': 'vibescope.run.v1',
        'source_sha256': source_hash, 'parameters': parameters, 'baseline': baseline,
        'files': {name: (hashlib.sha256(report_bytes).hexdigest() if name == 'report.json' else digest(folder, name)) for name in ARTIFACTS}})


def check_source(folder):
    source_bytes = read_bounded(folder, 'source.json')
    source = parse_json(source_bytes)
    if source.get('id') != folder.name:
        raise ValueError('Recording identity mismatch')
    if source.get('preview_url') != f'/data/{folder.name}/preview.png':
        raise ValueError('Invalid preview location')
    if any(type(source.get(key)) not in (int, float) or not math.isfinite(source[key]) or source[key] <= 0 for key in ('width', 'height', 'fps')):
        raise ValueError('Incomplete recording metadata')
    if source.get('history_schema') not in (None, 'vibescope.source.v1'):
        raise NotImplementedError('Unsupported source metadata version')
    if (folder / 'source-manifest.json').is_symlink():
        raise ValueError('Linked source manifest')
    try:
        manifest = read_json(folder, 'source-manifest.json')
    except FileNotFoundError:
        if source.get('history_schema'):
            raise ValueError('Source publication is incomplete')
        regular(folder, 'input.video'); regular(folder, 'preview.png')
        return source, 'legacy'
    schema = manifest.get('schema')
    if not isinstance(schema, str) or not schema:
        raise ValueError('Missing or invalid source manifest version')
    if schema != 'vibescope.source.v1':
        raise NotImplementedError('Unsupported source manifest version')
    expected = manifest.get('files')
    if type(expected) is not dict or set(expected) != {'source.json', 'input.video', 'preview.png'}:
        raise ValueError('Invalid source manifest')
    for name, value in expected.items():
        actual = hashlib.sha256(source_bytes).hexdigest() if name == 'source.json' else digest(folder, name)
        if actual != value:
            raise ValueError('Source changed: ' + name)
    return source, 'complete'


def check_run(folder, source_folder):
    manifest = read_json(folder, 'run-manifest.json')
    schema = manifest.get('schema')
    if not isinstance(schema, str) or not schema:
        raise ValueError('Missing or invalid run manifest version')
    if schema != 'vibescope.run.v1':
        raise NotImplementedError('Unsupported run manifest version')
    expected = manifest.get('files')
    if type(expected) is not dict or set(expected) != set(ARTIFACTS):
        raise ValueError('Invalid run manifest')
    if digest(source_folder, 'input.video') != manifest.get('source_sha256'):
        raise ValueError('Recording bytes changed')
    report_bytes = read_bounded(folder, 'report.json')
    for name, value in expected.items():
        actual = hashlib.sha256(report_bytes).hexdigest() if name == 'report.json' else digest(folder, name)
        if actual != value:
            raise ValueError('Run artifact changed: ' + name)
    report = parse_json(report_bytes)
    params = manifest.get('parameters')
    if type(params) is not dict or type(report.get('parameters')) is not dict:
        raise ValueError('Missing run parameters')
    if report.get('id') != folder.name or params.get('video_id') != source_folder.name:
        raise ValueError('Run identity mismatch')
    expected_urls = {key: f'/data/{source_folder.name}/{folder.name}/{name}' for key, name in {
        'original': 'original.mp4', 'enhanced': 'enhanced.mp4', 'heatmap': 'heatmap.png',
        'csv': 'displacement.csv', 'plot': 'measurement.png', 'report': 'report.json'}.items()}
    if report.get('artifacts') != expected_urls:
        raise ValueError('Invalid artifact locations')
    for key, value in report['parameters'].items():
        if params.get(key) != value:
            raise ValueError('Saved parameters do not match the report')
    reference = manifest.get('baseline')
    baseline = None
    if params.get('baseline_id'):
        if type(reference) is not dict or reference.get('id') != params['baseline_id'] or not ID.fullmatch(reference['id']):
            raise ValueError('Invalid reference binding')
        refs = source_folder.parent / 'references'
        baseline_bytes = read_bounded(refs, reference['id'] + '.json', 64 * 1024)
        if hashlib.sha256(baseline_bytes).hexdigest() != reference.get('sha256'):
            raise ValueError('Reference changed or unavailable')
        baseline = parse_json(baseline_bytes)
    elif reference is not None:
        raise ValueError('Unexpected reference binding')
    return report, params, baseline


def entries(parent):
    # Scan a bounded number of names, including irrelevant entries, to avoid
    # unbounded work in a directory that is not controlled by this process.
    found = []
    with os.scandir(parent) as scan:
        for index, item in enumerate(scan):
            if index >= MAX_ENTRIES:
                break
            if ID.fullmatch(item.name) and item.is_dir(follow_symlinks=False):
                found.append(item.name)
    return sorted(found)


def classify_run(folder, source_folder):
    try:
        check_run(folder, source_folder)
        return 'completed'
    except NotImplementedError:
        return 'incompatible'
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return 'partial'
