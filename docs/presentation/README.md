# Paired-gain measurement evidence

This actual November 17, 2026 browser session used public source db2594d6ff15d75151785c388ed35c473533d3b9, tree 0386b5004fabaa1408c455c50b47c8613b9a7675,development version 0.1.1.dev0. The packaged controlled synthetic target was measured twice, using  gains 5 and 20. Complete 240-sample signal arrays, all  quantitative metrics and CSV bytes match exactly; visualization settings and enhanced video differ. The same workspace reopened the gain-20 run after the server process was restarted.

Known synthetic input is 7.3 Hz / 0.28 px. This run returned 7.299959660082684 Hz and 0.2798920570385787 px. Timing remained explicitly unverified because ffprobe was unavailable. No physical calibration, resonance, diagnosis or general accuracy is established.

The figure combines the actual original/enhanced decoded output frames at index 72 (time 1.2 s) with the actual report arrays. Native 320×192 RGB pixels are preserved as `original-frame72.png` and `enhanced-frame72.png`; nearest-neighbor enlargement changes display size only. No contrast, color, denoising or selective image adjustments were applied. The enhanced frame already includes the real application visualization.

All 240 displacement samples from each gain are encoded in SVG without path simplification. The spectrum uses all 120 stored positive-frequency bins; DC is explicitly excluded from the logarithmic axis. There is no new estimator, fit, filter, normalization or smoothing in the renderer. The 4–12 Hz band shading is the declared analysis setting. No confidence interval is drawn.

`gain5-report.json`,`gain20-report.json` and both displacement CSVs are the actual browser-exported artifacts. Their local `/data/`URLs identify the recorded workspace;use the README’s file links for preserved copies. `manifest.json` binds source identity, input/output hashes, runtime versions, transformations and exact encoded vertex counts. `workspace-current.png` is the actual post-restart reopened workspace, not an interface mock or replacement visual direction.

Regenerate only the presentation with the project’s existing dependencies: `python docs/presentation/render.py`. It reads the preserved native frames and report arrays, and runs no measurement or provider. New application measurements require the documented source workflow and fresh output paths.
