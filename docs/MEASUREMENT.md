# Measurement guide

VibeScope estimates small, approximately rigid image-plane translations relative to the first decoded frame. Coordinates are source pixels: x points right; y points down. A stationary reference region removes common image-plane translation. Rotation, deformation within a region, perspective and rolling shutter require acquisition controls rather than a more convincing rendering.

## Acquire a useful clip

Mount the camera on a stable support. Use fixed focus, exposure, white balance and zoom. Include a textured specimen region and a separate textured stationary region in the same depth plane. Avoid specular highlights, shadows that move independently, occlusion and saturated pixels.

Choose capture rate based on an upper bound on the motion frequency, with substantial margin above twice that frequency. The Nyquist limit `fps/2` is necessary but does not establish absence of higher-frequency motion. Short exposures reduce blur and amplitude attenuation. Smartphone slow-motion files may have playback FPS different from capture FPS; record the actual capture setting.

Use a known in-plane scale at the specimen depth if physical displacement is needed. A ruler elsewhere in the scene, changed camera zoom or out-of-plane motion invalidates a simple mm/px scale.

## Analyze

1. Import the video and inspect metadata and timing warnings.
2. Select a textured region moving approximately as a rigid patch. A large region spanning a bending shape can mix several motions.
3. Select an independent stationary reference; regions must not overlap. With no reference, displacement includes camera translation.
4. Confirm the actual constant capture rate, axis and frequency band. The lower band edge must fit at least two cycles in the recording.
5. Start with the classical iterative tracker. Try the trained noise-aware estimator as a comparison within its small-motion domain, not as an assumed improvement.
6. Inspect raw and filtered displacement, spectrum, quality and warnings together. Adjust display gain only after the original-frame measurement is credible.
7. Export the report and CSV along with the source recording and acquisition notes.

A sample with tracking quality below 0.25 fails the run. The program does not replace unobservable frames with zeros or bridge them with invented displacement. Quality is a model-fit/texture index, not a measurement confidence or probability.

## Interpret results

**Frequency.** The resolver requires an interior PSD peak, 15 dB peak-to-median power, at least three observed cycles and a fitted sinusoid explaining at least 15% of full original detrended variance. It refines frequency by minimizing original-signal sine-fit residuals. These conservative engineering gates are not a statistically calibrated detection probability. A genuine weak selected-band mode dominated by strong off-band motion can remain unresolved. A peak at the band boundary can remain unresolved; widen the band. The reported `fps/N` spacing and±one-bin range are spectral resolution, not a confidence interval. Multiple unresolved nearby modes can bias a single-tone fit.

**Amplitude.** `amplitude_px` is the peak amplitude of a best-fit sinusoid on the unfiltered original-frame displacement. It is neither peak-to-peak displacement nor the maximum of the magnified video. The band-limited RMS includes filtering effects; start/end transients are not removed. `residual_mad_px` summarizes robust residual scatter, including other motion and fit error, rather than an independently calibrated camera noise floor.

**Calibration.** `amplitude_mm=amplitude_px×mm_per_px`. A user-entered scale is recorded as declared calibration; no calibration certificate, uncertainty propagation or lens correction is inferred. Pixel uncertainty from local image-fit residuals or synthetic model error cannot stand in for calibrated physical uncertainty.

**Damping.** Enable only for a free ringdown dominated by one linear mode with no continued forcing. The program fits the logarithm of positive peak amplitudes against time, requires at least six usable peaks and envelope R²≥0.90, and uses `ζ=decay/sqrt((2πfd)²+decay²)`. This is a conditional model fit, not proof that the physical assumptions hold. For an accepted single-mode fit, the report also derives natural frequency `fn=hypot(fd,decay/(2π))` and the conditional SDOF force-to-displacement resonance `fr=fn×sqrt(1−2ζ²)` when that model has a resonance peak. These are model-dependent estimates, not an experimentally measured excitation-response resonance. Forced, beating, nonlinear, time-varying or multimode responses can invalidate them.

**Resonance.** A response peak identifies an observed frequency. Establishing resonance requires controlled input or a transfer-function/operational-modal-analysis study. This release does not infer a modal model from arbitrary video.

**Heatmap.** The activity view is band-limited intensity RMS normalized for display. It can reveal where image intensity changes with motion, noise or lighting. It is not a dense displacement field, a calibrated motion amplitude map or a damage map.

**Magnification.** Classical/learned measurement paths render the selected ROI using its measured band-limited translation. The phase baseline renders temporal phase changes of an oriented quadrature band. Soft-edge blending and clipping are display operations. Requested gain is not verified physical output gain; reconstruction and cropping can distort appearance. Original measurements are independent of display gain.

## Novelty relative to healthy reference recordings

Acquire independent healthy recordings with the same camera, specimen, region, axis, scale, exposure, capture FPS, duration and estimator. Do not treat windows from one recording as independent experiments.

```bash
.venv/bin/python -m vibescope.cli baseline healthy/*.csv \
  --fps 60 --scope 'FixtureA, cameraA, regionA, x, pixel units, classical' \
  --output healthy-reference.json
```

The CLI requires at least 15 independent CSV files, splitting them into training and calibration sets; 10 training and 5 calibration is only an interface minimum. Use substantially more independent calibration records, 100+ when studying rare outliers. Small sets cannot resolve a 1% tail. The threshold is an empirical 99th-percentile feature-distance threshold, not a distribution-free false-alarm guarantee.

Load the resulting JSON in **Reference screening**, verify the stated acquisition scope and explicitly confirm the same setup. FPS and sample-count range must match. The score measures distance from that reference. `outside_reference` can indicate setup, lighting, operating-condition or processing changes; it does not diagnose damage. CLI analysis supports `--baseline healthy-reference.json --baseline-scope 'the identical scope'`.

## Independent physical validation protocol

No such validation has been completed in this release. Before claiming a real-world accuracy specification:

1. Clamp a ruler/cantilever or use a small controlled translating assembly. Establish safe amplitudes and a repeatable release/excitation. Record camera, lens, distance, exposure, focus, true capture rate and a scale at the specimen plane.
2. Measure frequency independently using an appropriate reference sensor, encoder, tachometer or timing method. Measure displacement independently if making displacement-accuracy claims; another optical-flow estimate on the same frames is not independent ground truth.
3. Acquire stationary/noise, controlled periodic motion and free-ringdown recordings. Vary amplitude, frequency, lighting, compression and camera support within a stated operating envelope.
4. Separate training/tuning sessions from final held-out sessions. Include setup changes and null recordings. Retain originals, reference traces, calibration evidence, timing alignment and failures.
5. Report frequency error, displacement error/bias, uncertainty from repeated independent trials, rejected-measurement rate and runtime. For novelty screening, use independently labeled held-out operating states and report specificity/sensitivity with intervals and all failures.
6. Publish the actual acquisition manifest and data rights. A synthetic clip or re-rendered screenshot cannot satisfy independent experimental evidence.

The controlled examples and tests exercise the software with known synthetic inputs. They do not replace this protocol or validate diagnostic reliability.
