'use strict';

(() => {
  const $ = id => document.getElementById(id);
  const state = { source: null, result: null, roi: null, reference: null, tool: 'roi', drag: null, busy: false, stale: false, raf: 0, currentTime: 0, mediaDuration: 0, analysisFps: 0, baseline: null };
  const original = $('originalVideo');
  const enhanced = $('enhancedVideo');
  const canvas = $('roiCanvas');
  const color = { teal: '#12615b', reference: '#c28733', raw: '#b8c3bb', grid: '#e7ece5', text: '#7b8b7c' };
  const fmt = (value, digits = 3) => value !== null && value !== undefined && Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—';
  const numeric = value => value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value));
  const show = (id, visible = true) => $(id).classList.toggle('hidden', !visible);
  const text = (id, value) => { $(id).textContent = value; };
  const timecode = seconds => {
    const safe = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
    return `${String(Math.floor(safe / 60)).padStart(2, '0')}:${(safe % 60).toFixed(3).padStart(6, '0')}`;
  };
  function error(message) { text('errorBanner', message); show('errorBanner'); $('errorBanner').focus({preventScroll: false}); }
  function clearError() { show('errorBanner', false); }
  function setBusy(busy, label = '') {
    state.busy = busy;
    document.body.classList.toggle('busy', busy);
    $('demoButton').disabled = busy;
    $('fileInput').disabled = busy;
    $('controls').disabled = busy || !state.source;
    $('captureTiming').disabled = busy || !state.source;
    $('baselineControls').disabled = busy || !state.source;
    $('baselineConfirmed').disabled = busy || !state.baseline;
    $('clearBaseline').disabled = busy || !state.baseline;
    ['measurementTool', 'referenceTool', 'clearRoi', 'applyRoi'].forEach(id => { $(id).disabled = busy || !state.source; });
    $('analyzeButton').disabled = busy || !state.source || !state.roi;
    $('jobStatus').classList.toggle('pending', busy);
    text('jobStatus', label);
    show('jobStatus', Boolean(label));
  }
  async function request(url, options = {}) {
    const response = await fetch(url, options);
    const body = await response.text();
    let data;
    try { data = JSON.parse(body); } catch { throw new Error(response.ok ? 'The server returned an unreadable response.' : `Request failed (${response.status}).`); }
    if (!response.ok) {
      let message = data.detail || data.error || `Request failed (${response.status}).`;
      if (Array.isArray(message)) message = message.map(item => `${item.loc?.slice(-1)[0] || 'Input'}: ${item.msg || 'invalid value'}`).join(' · ');
      throw new Error(typeof message === 'string' ? message : JSON.stringify(message));
    }
    return data;
  }
  function stopPlayback() {
    original.pause(); enhanced.pause();
    cancelAnimationFrame(state.raf);
    $('playButton').setAttribute('aria-label', 'Play synchronized videos');
    $('playButton').innerHTML = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="m6 3 10 7-10 7Z"/></svg>';
  }
  function clearResult() {
    stopPlayback();
    state.result = null; state.stale = false; state.currentTime = 0;
    $('analyzeButton').querySelector('span').textContent = 'Run measurement';
    original.removeAttribute('src'); original.load(); enhanced.removeAttribute('src'); enhanced.load();
    show('originalVideo', false); show('enhancedVideo', false); show('enhancedEmpty'); show('enhancementWatermark', false);
    show('heatmap', false); show('heatmapEmpty'); show('waveformEmpty'); show('spectrumEmpty');
    ['frequencyValue', 'amplitudeValue', 'qualityValue', 'dampingValue'].forEach(id => text(id, '—'));
    text('amplitudeUnit', 'px');
    text('frequencyNote', 'Awaiting source measurement'); text('amplitudeNote', 'Original displacement signal'); text('qualityNote', 'Texture and tracking agreement'); text('dampingNote', 'Enable for an isolated ringdown');
    text('analysisStatus', 'No measurement yet'); text('analysisSummary', 'Select a textured measurement region and an independent stationary reference, then run measurement.');
    text('rmsCaption', 'RMS — px'); text('resolutionCaption', 'Resolution — Hz'); text('bandCaption', 'No frequency band processed');
    show('anomalyResult', false); $('timeline').disabled = true; $('playButton').disabled = true; $('timeline').value = 0;
    text('currentTime', timecode(0));
    ['exportCsv', 'exportPlot', 'exportReport'].forEach(id => { $(id).classList.add('disabled'); $(id).setAttribute('aria-disabled', 'true'); $(id).removeAttribute('href'); });
    drawCharts();
  }
  function setWarnings(warnings) {
    const list = $('warningList'); list.replaceChildren();
    const items = Array.isArray(warnings) ? warnings : warnings ? [String(warnings)] : [];
    items.forEach(warning => { const li = document.createElement('li'); li.textContent = typeof warning === 'string' ? warning : warning.message || JSON.stringify(warning); list.appendChild(li); });
  }
  function validRoi(roi) {
    if (!state.source || !Array.isArray(roi) || roi.length !== 4 || !roi.every(Number.isFinite)) return false;
    const [x, y, w, h] = roi;
    return x >= 0 && y >= 0 && w >= 24 && h >= 24 && x + w <= state.source.width && y + h <= state.source.height;
  }
  function loadSource(source) {
    if (!source || !source.id || !(source.width > 0) || !(source.height > 0) || !(source.fps > 0)) throw new Error('The video metadata is incomplete. Try another recording.');
    clearResult(); state.source = source; clearBaseline();
    state.roi = validRoi(source.recommended_roi) ? source.recommended_roi.map(Math.round) : null;
    state.reference = validRoi(source.recommended_reference_roi) ? source.recommended_reference_roi.map(Math.round) : null;
    state.tool = 'roi';
    $('measurementTool').classList.add('active'); $('referenceTool').classList.remove('active');
    text('workspaceTitle', source.name || 'Untitled recording');
    const synthetic = String(source.source_kind).toLowerCase().includes('synthetic');
    $('calibration').value = ''; $('ringdown').checked = false;
    $('captureFps').value = ''; $('captureFps').placeholder = `${fmt(source.fps, 3)} from file metadata`; $('assumeCfr').checked = synthetic;
    show('captureTiming', !synthetic); $('axis').value = source.recommended_axis || (synthetic ? 'x' : 'y');
    state.mediaDuration = source.duration; state.analysisFps = source.fps;
    text('workspaceSubtitle', synthetic ? 'Controlled synthetic motion · known input, independent image-based measurement' : 'Original-frame analysis · select a textured region to begin');
    text('sessionTag', synthetic ? 'SYNTHETIC SOURCE' : 'SOURCE LOADED');
    $('evidenceBanner').classList.toggle('synthetic', synthetic);
    $('evidenceBanner').querySelector('p').textContent = synthetic ? 'Controlled synthetic recording. These results test the image pipeline under known inputs; they do not validate real-world displacement accuracy or diagnostic reliability.' : 'Source timing and calibration are unverified. Confirm the recording frame rate, use an in-plane scale and inspect the acquisition limitations before interpreting physical units.';
    text('sourceFile', source.name || 'Untitled recording');
    text('sourceFacts', `${fmt(source.fps, 3)} fps · ${source.frame_count} frames\n${source.width} × ${source.height} px · ${fmt(source.duration, 2)} s`);
    $('sourceFacts').style.whiteSpace = 'pre-line';
    text('sourceKind', synthetic ? 'CONTROLLED SYNTHETIC SOURCE' : 'USER RECORDING · UNVERIFIED CALIBRATION');
    text('sourceDimensions', `${source.width} × ${source.height} · ${fmt(source.fps, 2)} fps`);
    text('sourceProvenance', synthetic ? 'SYNTHETIC' : 'ORIGINAL');
    text('totalTime', timecode(source.duration));
    text('nyquistLabel', `Nyquist ${fmt(source.fps / 2, 2)} Hz`);
    $('highHz').max = String(source.fps / 2 - 0.001);
    const band = source.recommended_band || source.recommended_band_hz;
    $('lowHz').value = band?.[0] ?? (synthetic ? 4 : Math.max(0.1, Math.min(1, source.fps / 12)));
    $('highHz').value = band?.[1] ?? Math.min(synthetic ? 12 : 15, source.fps / 2 * 0.85);
    const ratio = `${source.width} / ${source.height}`;
    $('sourceStage').style.aspectRatio = ratio; $('enhancedStage').style.aspectRatio = ratio;
    canvas.width = source.width; canvas.height = source.height;
    $('previewImage').src = source.preview_url;
    show('sourceEmpty', false); show('previewImage'); show('roiCanvas'); show('sourceMetadata'); show('sourceOverlay', !state.roi);
    setWarnings(source.warnings || []);
    updateRoiUI(); drawRois();
  }
  async function importVideo(file) {
    if (state.busy || !file) return;
    if (file.size > 120 * 1024 * 1024) { error('Video exceeds 120 MiB. Trim a copy of the source before importing.'); $('fileInput').value = ''; return; }
    clearError(); setBusy(true, 'Reading video frames and source timing…');
    try { const body = new FormData(); body.append('file', file); loadSource(await request('/api/upload', {method: 'POST', body})); setBusy(false, 'Source ready. Select a region to measure.'); }
    catch (e) { error(e.message); setBusy(false, 'Import failed.'); }
    $('fileInput').value = '';
  }
  $('fileInput').addEventListener('change', e => importVideo(e.target.files[0]));
  $('dropzone').addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); if (!state.busy) $('fileInput').click(); } });
  ['dragenter', 'dragover'].forEach(event => $('dropzone').addEventListener(event, e => { e.preventDefault(); $('dropzone').classList.add('dragover'); }));
  ['dragleave', 'drop'].forEach(event => $('dropzone').addEventListener(event, e => { e.preventDefault(); $('dropzone').classList.remove('dragover'); }));
  $('dropzone').addEventListener('drop', e => importVideo(e.dataTransfer.files[0]));
  $('demoButton').addEventListener('click', async () => {
    if (state.busy) return;
    clearError(); setBusy(true, 'Preparing the controlled synthetic recording…');
    try { loadSource(await request('/api/demo')); setBusy(false, state.roi ? 'Demo ready. Review the regions and run measurement.' : 'Demo ready. Select the moving region to begin.'); }
    catch (e) { error(e.message); setBusy(false, 'Demo could not be loaded.'); }
  });
  function roiPoint(event) {
    const rect = canvas.getBoundingClientRect();
    return [Math.max(0, Math.min(canvas.width, (event.clientX - rect.left) / rect.width * canvas.width)), Math.max(0, Math.min(canvas.height, (event.clientY - rect.top) / rect.height * canvas.height))];
  }
  function rectFromPoints(a, b) { const x = Math.round(Math.min(a[0], b[0])), y = Math.round(Math.min(a[1], b[1])); return [x, y, Math.min(canvas.width - x, Math.round(Math.abs(a[0] - b[0]))), Math.min(canvas.height - y, Math.round(Math.abs(a[1] - b[1])))]; }
  canvas.addEventListener('pointerdown', event => {
    if (!state.source || state.busy || event.button !== 0) return;
    event.preventDefault(); stopPlayback(); canvas.setPointerCapture(event.pointerId);
    const p = roiPoint(event); state.drag = {start: p, current: p}; drawRois();
  });
  canvas.addEventListener('pointermove', event => { if (!state.drag) return; state.drag.current = roiPoint(event); drawRois(); });
  canvas.addEventListener('pointerup', event => {
    if (!state.drag) return;
    const roi = rectFromPoints(state.drag.start, roiPoint(event)); state.drag = null;
    if (validRoi(roi)) { state[state.tool] = roi; clearError(); resetBaselineConfirmation(); markStale(); updateRoiUI(); }
    else error('Select a region at least 24 × 24 source pixels. A textured surface gives the most useful result.');
    drawRois();
  });
  canvas.addEventListener('pointercancel', () => { state.drag = null; drawRois(); });
  function drawRois() {
    const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, canvas.width, canvas.height);
    const scale = Math.max(1, canvas.width / Math.max(1, canvas.clientWidth));
    function draw(roi, reference = false, ghost = false) {
      if (!roi) return;
      const [x, y, w, h] = roi; const c = reference ? color.reference : color.teal;
      ctx.strokeStyle = c; ctx.fillStyle = reference ? '#c2873316' : '#12615b16'; ctx.lineWidth = 1.5 * scale;
      ctx.setLineDash(ghost ? [5 * scale, 3 * scale] : []); ctx.fillRect(x, y, w, h); ctx.strokeRect(x, y, w, h); ctx.setLineDash([]);
      if (ghost) return;
      const label = reference ? 'STATIONARY REFERENCE' : 'MEASUREMENT'; ctx.font = `600 ${8 * scale}px sans-serif`;
      const labelWidth = ctx.measureText(label).width + 10 * scale; const labelY = y > 19 * scale ? y - 18 * scale : y;
      ctx.fillStyle = c; ctx.fillRect(x, labelY, labelWidth, 16 * scale); ctx.fillStyle = '#fff'; ctx.fillText(label, x + 5 * scale, labelY + 11 * scale);
      const s = 3 * scale; ctx.fillStyle = c;
      [[x,y],[x+w,y],[x,y+h],[x+w,y+h]].forEach(([px, py]) => ctx.fillRect(px-s/2, py-s/2, s, s));
    }
    draw(state.roi); draw(state.reference, true);
    if (state.drag) draw(rectFromPoints(state.drag.start, state.drag.current), state.tool === 'reference', true);
  }
  function updateRoiUI() {
    const active = state[state.tool];
    ['roiX','roiY','roiW','roiH'].forEach((id, i) => { $(id).value = active ? active[i] : ''; });
    text('roiEditorLabel', `${state.tool === 'roi' ? 'Measurement' : 'Stationary reference'} region · source pixels`);
    const desc = roi => roi ? `${roi[2]} × ${roi[3]} px at (${roi[0]}, ${roi[1]})` : 'not selected';
    text('roiSummary', state.source ? `Measurement: ${desc(state.roi)} · Reference: ${state.reference ? 'selected' : 'none'}` : 'No measurement region selected');
    text('sourceOverlay', state.tool === 'reference' ? 'DRAG OVER A STATIONARY SURFACE' : 'DRAG TO SELECT A REGION');
    show('sourceOverlay', !state[state.tool]);
    $('analyzeButton').disabled = state.busy || !state.source || !state.roi;
  }
  function setTool(tool) {
    state.tool = tool; state.drag = null;
    $('measurementTool').classList.toggle('active', tool === 'roi'); $('referenceTool').classList.toggle('active', tool === 'reference'); updateRoiUI(); drawRois();
  }
  $('measurementTool').addEventListener('click', () => setTool('roi'));
  $('referenceTool').addEventListener('click', () => setTool('reference'));
  $('clearRoi').addEventListener('click', () => { state[state.tool] = null; resetBaselineConfirmation(); markStale(); updateRoiUI(); drawRois(); });
  $('applyRoi').addEventListener('click', () => {
    const roi = ['roiX','roiY','roiW','roiH'].map(id => Number($(id).value));
    if (!validRoi(roi) || roi.some(v => !Number.isInteger(v))) { error('Region coordinates must be whole pixels, inside the source image, with width and height at least 24.'); return; }
    state[state.tool] = roi; clearError(); resetBaselineConfirmation(); markStale(); updateRoiUI(); drawRois();
  });
  function markStale() {
    if (!state.result) return;
    state.stale = true; text('sessionTag', 'SETTINGS CHANGED');
    text('workspaceSubtitle', 'Settings changed. Displayed results belong to the previous run. Run measurement to update.');
    $('analyzeButton').querySelector('span').textContent = 'Update measurement';
  }
  ['method','axis','calibration','lowHz','highHz','ringdown'].forEach(id => $(id).addEventListener('change', markStale));
  $('gain').addEventListener('input', () => { text('gainValue', `${$('gain').value}×`); markStale(); });
  function effectiveFps() { return $('captureFps').value ? Number($('captureFps').value) : state.source?.fps || 0; }
  function updateTiming() {
    const fps = effectiveFps(); text('nyquistLabel', `Nyquist ${fmt(fps / 2, 2)} Hz`); $('highHz').max = String(fps / 2 - 0.001); markStale();
  }
  $('captureFps').addEventListener('input', updateTiming); $('assumeCfr').addEventListener('change', markStale);
  function clearBaseline() {
    state.baseline = null; $('baselineFile').value = ''; $('baselineConfirmed').checked = false;
    $('baselineConfirmed').disabled = true; $('clearBaseline').disabled = true;
    text('baselineSummary', 'No reference loaded. No anomaly assessment will be issued.');
  }
  $('clearBaseline').addEventListener('click', () => { clearBaseline(); markStale(); });
  $('baselineConfirmed').addEventListener('change', markStale);
  $('baselineFile').addEventListener('change', async event => {
    const file = event.target.files[0]; if (!file || state.busy) return;
    if (file.size > 64 * 1024) { error('Reference JSON exceeds 64 KiB. Select a reference-model file, not the underlying recordings.'); event.target.value = ''; return; }
    clearError(); setBusy(true, 'Validating healthy reference…');
    try {
      const body = new FormData(); body.append('file', file);
      const baseline = await request('/api/baseline', {method:'POST', body});
      if (!baseline.id || typeof baseline.scope !== 'string') throw new Error('The reference response is incomplete.');
      state.baseline = baseline; $('baselineConfirmed').checked = false;
      text('baselineSummary', `${baseline.scope}${numeric(baseline.fps) ? ` · ${fmt(baseline.fps, 3)} fps` : ''}${numeric(baseline.training_records) ? ` · ${baseline.training_records} training records` : ''}. Confirm the matching acquisition setup before screening.`);
      markStale(); setBusy(false, 'Reference loaded. Confirm the matching acquisition setup.');
    } catch (e) { error(e.message); setBusy(false, state.baseline ? 'Reference import failed. Previous reference retained.' : 'Reference import failed.'); }
    $('baselineFile').value = '';
  });
  function resetBaselineConfirmation() { if (state.baseline) { $('baselineConfirmed').checked = false; markStale(); } }
  ['method','axis','calibration','captureFps'].forEach(id => $(id).addEventListener('change', resetBaselineConfirmation));
  function settings() {
    const reference = state.baseline && $('baselineConfirmed').checked ? {baseline_id: state.baseline.id, baseline_scope: state.baseline.scope} : {};
    return {...reference, fps_override: $('captureFps').value ? Number($('captureFps').value) : null, assume_cfr: $('assumeCfr').checked, video_id: state.source.id, roi: state.roi, reference_roi: state.reference, axis: $('axis').value, low_hz: Number($('lowHz').value), high_hz: Number($('highHz').value), gain: Number($('gain').value), mm_per_px: $('calibration').value ? Number($('calibration').value) : null, method: $('method').value, ringdown: $('ringdown').checked};
  }
  $('analysisForm').addEventListener('submit', async event => {
    event.preventDefault(); if (state.busy) return; clearError();
    if (!state.source || !validRoi(state.roi)) { error('Select a measurement region before running the analysis.'); return; }
    const config = settings();
    if (state.baseline && !$('baselineConfirmed').checked) { error('Confirm the same acquisition setup as the loaded healthy reference, or remove the reference to run without screening.'); return; }
    if (config.reference_roi) {
      const [x,y,w,h]=config.roi, [rx,ry,rw,rh]=config.reference_roi;
      if (Math.min(x+w,rx+rw)>Math.max(x,rx) && Math.min(y+h,ry+rh)>Math.max(y,ry)) { error('Measurement and stationary reference regions must not overlap. Select a separate stationary surface.'); return; }
    }
    if (!String(state.source.source_kind).toLowerCase().includes('synthetic') && !config.assume_cfr) { error('Confirm constant capture timing and verify the actual camera capture frame rate before measuring. Slow-motion playback rate can differ from capture rate.'); return; }
    if (!(effectiveFps() > 0 && Number.isFinite(effectiveFps()))) { error('Enter a positive finite actual capture frame rate, or leave it blank to use file metadata.'); return; }
    if (!(config.low_hz > 0 && config.low_hz < config.high_hz && config.high_hz < effectiveFps() / 2)) { error(`Choose 0 < low < high < ${fmt(effectiveFps() / 2, 3)} Hz. The upper band must stay below Nyquist.`); return; }
    if (config.mm_per_px !== null && !(config.mm_per_px > 0 && Number.isFinite(config.mm_per_px))) { error('Calibration must be a positive millimeters-per-pixel scale, or leave it blank for pixels.'); return; }
    stopPlayback(); setBusy(true, 'Measuring original frames, estimating the spectrum and rendering visualization…');
    try { const result = await request('/api/analyze', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(config)}); renderResult(result, config); setBusy(false, 'Measurement complete. Inspect the signal and limitations below.'); }
    catch (e) { error(e.message); setBusy(false, state.result ? 'Measurement failed. Previous results are retained.' : 'Measurement failed. Review the source and settings.'); }
  });
  function renderResult(result, config) {
    if (!result || !result.metrics || !result.signal) throw new Error('The analysis response is missing measurements.');
    state.result = result; state.result.uiSettings = config; state.stale = false; state.currentTime = 0;
    const m = result.metrics, a = result.artifacts || {};
    state.analysisFps = Number(result.metadata?.fps) || config.fps_override || state.source.fps;
    state.mediaDuration = result.signal.time_s?.length ? result.signal.time_s[result.signal.time_s.length - 1] + 1 / state.analysisFps : state.source.duration;
    text('totalTime', timecode(state.mediaDuration));
    text('sessionTag', 'MEASUREMENT READY');
    text('workspaceSubtitle', `${config.axis.toUpperCase()}-axis image-plane motion · ${config.low_hz}–${config.high_hz} Hz selected band · ${config.reference_roi ? 'stationary reference applied' : 'no camera-motion reference'}`);
    $('analyzeButton').querySelector('span').textContent = 'Run measurement';
    text('frequencyValue', fmt(m.frequency_hz, 3));
    text('frequencyNote', numeric(m.frequency_resolution_hz) ? `Bin width ${fmt(m.frequency_resolution_hz, 3)} Hz · observed response` : 'Dominant observed response');
    const calibrated = numeric(m.amplitude_mm);
    text('amplitudeUnit', calibrated ? 'mm' : 'px'); text('amplitudeValue', fmt(calibrated ? m.amplitude_mm : m.amplitude_px, calibrated ? 4 : 3));
    text('amplitudeNote', calibrated ? `Scale supplied by user · ${fmt(m.amplitude_px, 3)} px` : `Original-signal fit · pixels, no physical scale`);
    text('qualityValue', numeric(m.quality) ? fmt(m.quality, 2) : '—');
    text('qualityNote', `Quality index, not probability${numeric(m.noise_floor_px) ? ` · fit scatter ${fmt(m.noise_floor_px, 3)} px` : ''}`);
    text('dampingValue', numeric(m.damping_ratio) ? fmt(m.damping_ratio, 4) : '—');
    text('dampingNote', numeric(m.damping_ratio) ? `Single-mode fit${numeric(m.natural_frequency_hz) ? ` · conditional natural frequency ${fmt(m.natural_frequency_hz, 3)} Hz` : ' · verify isolated free decay'}` : config.ringdown ? 'Fit unavailable or acceptance checks failed' : 'Not requested · free decay required');
    text('bandCaption', `${config.low_hz}–${config.high_hz} Hz · ${config.gain}× total gain`); text('watermarkGain', `${config.gain}×`);
    text('rmsCaption', `RMS ${fmt(m.rms_px, 3)} px`); text('resolutionCaption', `Bin width ${fmt(m.frequency_resolution_hz, 3)} Hz`);
    text('waveformCaption', `${config.axis.toUpperCase()}-axis original-frame signal · pixels`);
    text('analysisStatus', m.status ? String(m.status).replaceAll('_', ' ') : 'Inspect measurement quality before interpreting');
    text('analysisSummary', `Measurement from ${state.source.frame_count} source frames at ${fmt(state.analysisFps, 3)} capture fps. ${config.reference_roi ? 'Shared image translation was estimated from the stationary reference.' : 'Camera motion can be included in the displacement signal.'} Enhanced images are for visualization only.`);
    setWarnings([...(state.source.warnings || []), ...(result.warnings || [])]);
    if (m.anomaly) {
      const anomaly = m.anomaly;
      text('anomalyResult', `Reference distance: ${numeric(anomaly.score) ? fmt(anomaly.score, 3) : 'unavailable'}${numeric(anomaly.threshold) ? ` · threshold ${fmt(anomaly.threshold, 3)}` : ''}. ${String(anomaly.status || 'Unverified').replaceAll('_', ' ')}. ${anomaly.reference_scope ? `Reference: ${anomaly.reference_scope}. ` : ''}${anomaly.explanation || 'Reference distance is not a diagnostic probability.'}`);
      if (anomaly.status === 'not_assessed') text('anomalyResult', `Anomaly assessment not run. ${anomaly.explanation || 'A healthy reference matched to this acquisition setup is required.'}`);
      $('anomalyResult').classList.toggle('unassessed', anomaly.status === 'not_assessed');
      show('anomalyResult');
    } else show('anomalyResult', false);
    if (a.original) { original.src = a.original; show('originalVideo'); show('previewImage', false); }
    if (a.enhanced) { enhanced.src = a.enhanced; show('enhancedVideo'); show('enhancedEmpty', false); show('enhancementWatermark'); }
    if (a.heatmap) { $('heatmap').src = a.heatmap; show('heatmap'); show('heatmapEmpty', false); }
    $('playButton').disabled = !a.original || !a.enhanced; $('timeline').disabled = !a.original; $('timeline').max = String(state.mediaDuration); $('timeline').value = 0;
    text('currentTime', timecode(0));
    [['exportCsv','csv'],['exportPlot','plot'],['exportReport','report']].forEach(([id,key]) => { if (a[key]) { $(id).href = a[key]; $(id).setAttribute('download', ''); $(id).classList.remove('disabled'); $(id).setAttribute('aria-disabled','false'); } });
    show('waveformEmpty', !result.signal.time_s?.length); show('spectrumEmpty', !result.signal.frequency_hz?.length);
    drawCharts(); drawRois();
  }
  function seek(time) {
    if (!state.result) return;
    const duration = Math.max(0, state.mediaDuration - 1 / state.analysisFps);
    const bounded = Math.max(0, Math.min(duration, time)); state.currentTime = bounded;
    if (original.readyState >= 1) original.currentTime = bounded;
    if (enhanced.readyState >= 1) enhanced.currentTime = bounded;
    $('timeline').value = bounded; text('currentTime', timecode(bounded)); drawCharts();
  }
  function playbackTick() {
    state.currentTime = original.currentTime; $('timeline').value = original.currentTime; text('currentTime', timecode(original.currentTime));
    if (enhanced.readyState >= 2 && Math.abs(enhanced.currentTime - original.currentTime) > 0.075) enhanced.currentTime = original.currentTime;
    drawWaveform();
    if (!original.paused && !original.ended) state.raf = requestAnimationFrame(playbackTick);
  }
  $('playButton').addEventListener('click', async () => {
    if (!state.result || state.busy) return;
    if (!original.paused) { stopPlayback(); return; }
    if (original.ended || original.currentTime >= state.mediaDuration - 0.06) seek(0);
    try {
      enhanced.currentTime = original.currentTime;
      await Promise.all([original.play(), enhanced.play()]);
      $('playButton').setAttribute('aria-label', 'Pause synchronized videos');
      $('playButton').innerHTML = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 3h3v14H5Zm7 0h3v14h-3Z"/></svg>';
      state.raf = requestAnimationFrame(playbackTick);
    } catch { stopPlayback(); error('The browser could not play one of the videos. Try seeking, or download the report and inspect the source codec.'); }
  });
  original.addEventListener('ended', stopPlayback);
  original.addEventListener('seeked', () => { state.currentTime = original.currentTime; $('timeline').value = original.currentTime; text('currentTime', timecode(original.currentTime)); drawWaveform(); });
  $('timeline').addEventListener('input', e => seek(Number(e.target.value)));
  function chartFrame(id) {
    const el = $(id), dpr = Math.min(window.devicePixelRatio || 1, 2); const width = el.clientWidth, height = el.clientHeight;
    if (!width || !height) return null;
    if (el.width !== Math.round(width*dpr) || el.height !== Math.round(height*dpr)) { el.width = Math.round(width*dpr); el.height = Math.round(height*dpr); }
    const ctx = el.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,width,height);
    const area = {x:40, y:17, w:Math.max(1,width-54), h:Math.max(1,height-45)};
    ctx.font = '8px ui-monospace, SFMono-Regular, Consolas, monospace'; ctx.lineWidth = .7; ctx.strokeStyle = color.grid;
    for (let i=0;i<=4;i++) { const y=area.y+area.h*i/4; ctx.beginPath();ctx.moveTo(area.x,y);ctx.lineTo(area.x+area.w,y);ctx.stroke(); }
    for (let i=0;i<=5;i++) { const x=area.x+area.w*i/5; ctx.beginPath();ctx.moveTo(x,area.y);ctx.lineTo(x,area.y+area.h);ctx.stroke(); }
    return {ctx,area,width,height};
  }
  function line(ctx, xs, ys, area, xmin, xmax, ymin, ymax, stroke, linewidth=1.2) {
    const n = Math.min(xs.length,ys.length); if (!n) return;
    ctx.beginPath(); ctx.strokeStyle=stroke; ctx.lineWidth=linewidth;
    let started=false;
    for(let i=0;i<n;i++) { if(!numeric(xs[i]) || !numeric(ys[i])){started=false;continue;} const x=area.x+(Number(xs[i])-xmin)/(xmax-xmin)*area.w,y=area.y+(1-(Number(ys[i])-ymin)/(ymax-ymin))*area.h; if (!started) {ctx.moveTo(x,y);started=true;} else ctx.lineTo(x,y); }
    ctx.stroke();
  }
  function axisLabels(ctx, area, xmin, xmax, ymin, ymax, yFormat=v=>Math.abs(v)>=10 ? v.toFixed(0) : v.toFixed(2)) {
    ctx.fillStyle=color.text;ctx.textAlign='right';ctx.textBaseline='middle';
    for(let i=0;i<=4;i++)ctx.fillText(yFormat(ymax-(ymax-ymin)*i/4),area.x-7,area.y+area.h*i/4);
    ctx.textAlign='center';ctx.textBaseline='top';
    for(let i=0;i<=5;i++)ctx.fillText((xmin+(xmax-xmin)*i/5).toFixed(1),area.x+area.w*i/5,area.y+area.h+9);
  }
  function drawWaveform() {
    const frame=chartFrame('waveform');if(!frame)return;
    const {ctx,area}=frame; const s=state.result?.signal;
    if(!s?.time_s?.length){axisLabels(ctx,area,0,1,-1,1);return;}
    const xs=s.time_s, raw=s.raw_px||[],filtered=s.filtered_px||[];
    const values=[...raw,...filtered].filter(numeric).map(Number); const lo=values.length?values.reduce((a,b)=>Math.min(a,b),Infinity):0,hi=values.length?values.reduce((a,b)=>Math.max(a,b),-Infinity):0;
    const span=Math.max(hi-lo,.005), ymin=lo-span*.12,ymax=hi+span*.12; const xmax=Math.max(xs[xs.length-1],1e-6);
    axisLabels(ctx,area,0,xmax,ymin,ymax);
    ctx.save();ctx.beginPath();ctx.rect(area.x,area.y,area.w,area.h);ctx.clip();
    line(ctx,xs,raw,area,0,xmax,ymin,ymax,color.raw,1);line(ctx,xs,filtered,area,0,xmax,ymin,ymax,color.teal,1.4);
    const x=area.x+state.currentTime/xmax*area.w;ctx.strokeStyle='#a37b3a';ctx.lineWidth=1;ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(x,area.y);ctx.lineTo(x,area.y+area.h);ctx.stroke();ctx.restore();
    $('waveform')._plot={area,xmax};
  }
  function drawSpectrum() {
    const frame=chartFrame('spectrum');if(!frame)return;
    const {ctx,area}=frame;const s=state.result?.signal;
    if(!s?.frequency_hz?.length){axisLabels(ctx,area,0,state.analysisFps/2||30,0,1);return;}
    const xs=s.frequency_hz,ys=(s.psd_px2_hz||[]).map(v=>numeric(v)?Number(v):0);const xmax=Math.max(Number(xs[xs.length-1]),1e-6);const ymax=ys.reduce((a,b)=>Math.max(a,b),1e-9)*1.15;
    const cfg=state.result.uiSettings;
    ctx.fillStyle='#12615b0d';ctx.fillRect(area.x+cfg.low_hz/xmax*area.w,area.y,(cfg.high_hz-cfg.low_hz)/xmax*area.w,area.h);
    axisLabels(ctx,area,0,xmax,0,ymax,v=>v>=10?v.toFixed(0):v>=.01?v.toFixed(2):v.toExponential(0));
    ctx.save();ctx.beginPath();ctx.rect(area.x,area.y,area.w,area.h);ctx.clip();line(ctx,xs,ys,area,0,xmax,0,ymax,color.teal,1.4);ctx.restore();
    const f=state.result.metrics.frequency_hz;
    if(numeric(f)) {const x=area.x+f/xmax*area.w;ctx.strokeStyle=color.reference;ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(x,area.y);ctx.lineTo(x,area.y+area.h);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=color.reference;ctx.textAlign=x>area.x+area.w*.7?'right':'left';ctx.textBaseline='bottom';ctx.fillText(`${fmt(f,3)} Hz`,Math.min(area.x+area.w-3,Math.max(area.x+3,x+4)),area.y-4);}
    $('spectrum')._plot={area,xmax};
  }
  function drawCharts(){drawWaveform();drawSpectrum();}
  $('waveform').addEventListener('click',event=>{const p=$('waveform')._plot;if(!p||!state.result)return;const x=event.clientX-$('waveform').getBoundingClientRect().left;seek(Math.max(0,Math.min(1,(x-p.area.x)/p.area.w))*p.xmax);});
  $('spectrum').addEventListener('click',event=>{
    const p=$('spectrum')._plot;if(!p||!state.source||state.busy)return;const x=event.clientX-$('spectrum').getBoundingClientRect().left;
    const center=Math.max(.02,Math.min(effectiveFps()/2-.02,(x-p.area.x)/p.area.w*p.xmax));
    const width=Math.max(.1,Number($('highHz').value)-Number($('lowHz').value)); const low=Math.max(.01,center-width/2),high=Math.min(effectiveFps()/2-.01,center+width/2);
    $('lowHz').value=low.toFixed(2);$('highHz').value=high.toFixed(2);markStale();
    text('jobStatus',`Band set to ${low.toFixed(2)}–${high.toFixed(2)} Hz. Run measurement to apply.`);show('jobStatus');
  });
  let resizePending=false;
  new ResizeObserver(()=>{if(resizePending)return;resizePending=true;requestAnimationFrame(()=>{drawCharts();drawRois();resizePending=false;});}).observe(document.querySelector('main'));
  drawCharts();
})();
