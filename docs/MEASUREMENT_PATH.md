# Measurement path

The [desktop](assets/measurement-path.svg) and [portrait](assets/measurement-path-portrait.svg) diagrams summarize the original-frame data path. Arrows show selected dependencies, not a processing time scale or a complete algorithm. Both SVGs are editable text; they contain no embedded benchmark traces or new experimental results.

| Diagram element | Implementation |
| --- | --- |
| Original input | [`video.read_video` and `metadata`](../vibescope/video.py) decode the local clip and inspect timing. [`pipeline._analyze`](../vibescope/pipeline.py) validates controls, preserves capture/source metadata and converts original frames to grayscale for tracking. |
| Region tracking | [`motion.track`](../vibescope/motion.py) compares each original patch with its first-frame patch using the selected classical, phase or learned estimator. An optional nonoverlapping stationary region is tracked classically in both axes; its selected-axis translation is subtracted. Without it, camera translation remains. |
| Tracking gate | [`pipeline._analyze`](../vibescope/pipeline.py) combines target/reference quality and refuses a run with any quality below 0.25. Quality is a fit/texture index, not physical confidence. |
| Signal analysis | [`signals.analyze_signal`](../vibescope/signals.py) computes the PSD from detrended original-frame displacement and fits sinusoidal amplitude on the unfiltered displacement. Its interior-peak, prominence, variance and cycle gates can leave the tone unresolved. The bandpass uses the whole clip and is noncausal. |
| Measured results | The report retains raw and band-limited displacement, frequency resolution, metrics, parameters and warnings. A declared mm/px scale and a declared free-ringdown fit have their existing conditional interpretation; the diagram does not replace the [measurement guide](MEASUREMENT.md). |
| Display products | [`pipeline.render_visuals`](../vibescope/pipeline.py) uses original RGB frames and camera translation estimates, plus filtered displacement or temporal phase, to form the display. Gain, blending, encoding and normalized intensity activity do not feed the quantitative tracking or sine fit. The activity map is not dense displacement or damage. |
| Completed bundle | `_analyze` renders the visual assets before writing the CSV, measurement plot and report. [`pipeline.analyze`](../vibescope/pipeline.py) stages the run and publishes the fresh result directory only after those steps succeed. The display branch is not optional or failure-independent. |

The diagram omits detailed estimator iterations, reference-novelty screening and optional ringdown/calibration formulas. A dominant response peak is not proof of resonance, and synthetic validation does not establish physical accuracy. No new inference, calibration or physical experiment was performed to create these illustrations.
