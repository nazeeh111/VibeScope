# Measurement evidence

VibeScope is a research prototype with controlled synthetic validation. No independently measured physical recordings or reference instruments were available for this release. Pixel errors below describe the stated image generators and acquisitions. They do not establish calibrated real-world displacement accuracy, diagnostic reliability, or general superiority over published methods.

![Controlled measurement and reference-screening experiments](../experiments/controlled/benchmark.png)

## What is compared

| Method | Input used for measurement | What the comparison establishes |
| --- | --- | --- |
| Physical fit, the default | Original grayscale frames | Iterative subpixel translation with brightness gain/offset nuisance fitting |
| Quadrature phase | Original grayscale frames | An original oriented quadrature-band estimator; not a reproduction of the full Wadhwa steerable-pyramid method |
| Noise fusion MLP | Original grayscale frames | A 362-parameter learned correction to multiscale differential translation estimates |
| No learned correction | Original grayscale frames | The middle-scale estimate and observation/domain gates retained, with only the learned correction removed |
| EulerMormer | Its genuinely inferred magnified output, tracked by the physical fitter, then divided by requested gain | A small external neural magnification comparison including reconstruction and output-tracker error; not direct neural displacement inference |

Magnification is not a calibrated sensor. Measurements from reconstructed videos are isolated to the external-baseline experiment and never feed ordinary VibeScope inference.

## Controlled video-estimator protocol

The main experiment uses three texture seeds, three added noise levels (0, 0.01 and 0.04 normalized intensity), both image axes, 180 frames at 60 fps, 7.3 Hz and 0.25 px sinusoidal amplitude. An additional three continuously evaluated analytic textures test a different rendering family from the Fourier-shift training generator, with 0.01 image noise and a small brightness drift. No training or calibration texture seeds are reused.

Each sequence is 64 × 64 pixels. Estimated and true displacement are each centered once per clip, with no fitted gain correction. Signal-to-error ratio compares known displacement variance against squared estimation error. Frequency error and amplitude error come from the signal pipeline applied to the original-frame estimates. Frequency refinement cannot improve the underlying finite-record resolving power.

Conditions reuse texture seeds. Uncertainty therefore resamples texture-seed clusters with their original condition counts, keeping the repeated axis/noise conditions together. The 95% percentile bootstrap uses 2,000 replicates and only three texture clusters per rendering family. These intervals are exploratory estimates of this small experiment, not calibrated instrument confidence intervals or population-wide accuracy guarantees.

Full per-sequence rows, original estimated curves, software versions and code/checkpoint hashes are retained in [results.json](../experiments/controlled/results.json), [measurements.csv](../experiments/controlled/measurements.csv) and [curves.json](../experiments/controlled/curves.json).

| Method | Mean displacement RMSE, px (95% cluster bootstrap) | Mean frequency error, Hz | Mean amplitude error, px | Mean signal-to-error, dB |
| --- | ---: | ---: | ---: | ---: |
| Physical fit (default) | 0.00460 (0.00435–0.00557) | 0.000337 | 0.00039 | 42.22 |
| Quadrature phase | 0.02102 (0.01883–0.02187) | 0.000599 | 0.02738 | 18.60 |
| Noise fusion MLP | 0.00875 (0.00754–0.01239) | 0.000323 | 0.00755 | 27.76 |
| No learned correction | 0.00549 (0.00513–0.00619) | 0.000419 | 0.00061 | 38.05 |

All 84 estimator/sequence combinations were observable under the strict every-frame quality gate; none were excluded. A sequence is invalid if any frame has nonfinite quality or quality below 0.25. Accuracy aggregates use valid sequences and report excluded counts. Frequency errors below one FFT bin describe estimation of these isolated synthetic tones; the record still has a 0.333 Hz FFT-bin spacing and cannot resolve arbitrarily close modes.

The mixed-family table weights each sequence equally. Cluster bootstraps preserve each texture cluster’s original condition count, including the unequal counts between rendering families. Separate raster/analytic summaries are also retained in the JSON.

Observed estimator-only CPU FPS ranges on this loaded host:

| Method | Observed FPS range |
| --- | ---: |
| Physical fit (default) | 2.7–100.3 |
| Quadrature phase | 8.4–374.8 |
| Noise fusion MLP | 7.4–318.3 |
| No learned correction | 127.7–458.7 |


## What the trained model contributes

The original compact model is a 12 → 24 → 2 residual MLP trained on known two-dimensional displacement labels. It has 362 learned parameters and a portable 4.3 KB NumPy checkpoint. Its physical estimates, noise/texture diagnostics and bounded corrections are described in the [model card](MODEL_CARD.md).

On its separately seeded, 1,500-pair synthetic test split, vector RMSE was **0.07705 px** with learned correction versus **0.08428 px** without it, an **8.58% reduction**. That is a pairwise test on the stated training-family distribution, with motions up to 0.95 px per axis. It is different from the low-amplitude video-sequence experiment above. The correction does not consistently improve the latter, and the iterative physical fitter remains the default. No broad neural superiority or publication-level novelty is claimed.

The model's reported synthetic error scale is a held-out error summary, not a physical standard deviation or calibrated confidence interval. Flat/aperture-limited patches and inferred shifts outside the small-motion domain are rejected. Distribution changes can still escape those gates.

## External neural baseline

The optional evaluator loads the real EulerMormer epoch-99 checkpoint from its pinned upstream checkout, verifies its code and checkpoint SHA-256 hashes, loads tensors with `weights_only=True`, and requires an exact state-dictionary match. The network has 1,512,056 parameters. Upstream code and weights remain external to the VibeScope distribution.

Two held-out texture seeds × two image-noise levels give four 120-frame, 64 × 64 RGB sequences at 60 fps, 7.3 Hz and 0.25 px amplitude. The requested total magnification is 10. Inference uses the first frame as a static reference and does not apply temporal filtering. Original estimators receive the same quantized input frames. Every magnified-output frame is checked by the physical tracker; rejected output tracking must not be treated as a valid displacement estimate.

[comparison.json](../experiments/neural_baseline/comparison.json), [curves.json](../experiments/neural_baseline/curves.json) and [source_hashes.json](../experiments/neural_baseline/source_hashes.json) retain the actual outputs, validity summaries, timings and provenance. The four clips are too small a sample for a general baseline ranking.

| Method | Mean displacement RMSE, px | Mean frequency error, Hz | Mean amplitude error, px | Observed CPU FPS | Valid sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Physical fit (default) | 0.00277 | 0.000222 | 0.00044 | 9.66–58.02 | 4/4 |
| Quadrature phase | 0.02357 | 0.000133 | 0.03223 | 42.93–234.42 | 4/4 |
| Noise fusion MLP | 0.00877 | 0.000313 | 0.00982 | 37.53–257.10 | 4/4 |
| EulerMormer | 0.00797 | 0.000207 | 0.00732 | 0.49–2.61 | 4/4 |

Every frame passed the output-tracker quality gate in these four EulerMormer clips. The benchmark now rejects a sequence if any tracking sample is nonfinite or below 0.25, matching the production inference policy. Tracking agreement is an image-model check, not an independently calibrated uncertainty bound.

The real neural inference and original estimated curves were retained while stricter final spectral checks were refreshed from those saved estimates. The reports preserve the original inference hashes and explicit signal-refresh provenance. Estimator/checkpoint/reference hashes were required to match. The ablation diagnostic gate was recomputed from exact seeded inputs with displacement equality asserted. No neural inference was synthesized or approximated for the refresh.

## Reference screening

The anomaly component screens statistical novelty against a same-setup reference. It does not identify faults or diagnose damage. Its features describe displacement RMS, dominant frequency, harmonic energy and off-peak energy. A robust center/scale is fit on 200 independent reference-like synthetic waveforms; a separate 100-waveform calibration set selects an empirical 99th-percentile distance threshold. The final 500-record test set is untouched by fitting and calibration.

The test includes 100 reference-like waveforms and 100 each with changed frequency, amplitude, harmonics or broadband noise. Seeds are disjoint between training, calibration and testing. AUROC uncertainty resamples independent test records; sensitivity and specificity use Wilson binomial intervals. The 99th-percentile threshold is empirical and carries no distribution-free false-alarm guarantee. Small calibration sets are especially unreliable for rare-event thresholds.

The reference, every test record and scores are saved in [synthetic_reference.json](../experiments/controlled/synthetic_reference.json), [anomaly_records.json](../experiments/controlled/anomaly_records.json) and [anomaly.json](../experiments/controlled/anomaly.json). These files are explicitly synthetic references, not healthy-machine baselines.

| Test metric | Result | 95% interval |
| --- | ---: | ---: |
| AUROC | 0.98255 | 0.97286–0.99034 (test-record bootstrap) |
| Sensitivity to the authored changes | 84.0% (336/400) | 80.1%–87.3% (Wilson) |
| Specificity on reference-like simulations | 99.0% (99/100) | 94.6%–99.8% (Wilson) |

At the fixed threshold **1.708008**, the detector flagged 90/100 frequency changes, 93/100 amplitude changes, 100/100 harmonic changes and 53/100 broadband-noise changes. The missed changes are retained in the record-level data. A high AUROC does not imply high sensitivity at the selected operating threshold.

An additional [image transfer check](../experiments/controlled/anomaly_image_check.json) applies the unchanged waveform-reference threshold to original-frame estimates from independent analytic-image sequences. It is a small end-to-end component check, including image-to-signal error, with no threshold retuning to improve its test score. Only frequency and amplitude changes are included. Its intervals and failures must be interpreted at that limited scope.

The 20-record image check flagged **10/10** changed sequences and accepted **10/10** reference-like sequences. Sensitivity and specificity were both 100% in this small authored subset, with Wilson 95% intervals of **72.2%–100%** and **72.2%–100%**, respectively. The wide lower bounds and restricted change types leave real-world reliability unverified.

## Speed and reproducibility limits

Reported inference FPS covers CPU estimator execution only. It excludes synthetic generation, video decoding/encoding, camera-reference processing, visualization, plotting and downloads. Neural timing additionally excludes an untimed per-case backend warmup and output tracking. End-to-end workspace processing takes longer. Measurements were collected on a loaded macOS ARM host with one configured BLAS/OpenMP thread; contention makes timings variable. They are observations of these runs, not real-time performance guarantees.

The UI and inference are CPU-compatible. Optional external EulerMormer execution can select a PyTorch device through its runner. Accelerated-device throughput is unverified unless a separate device result is supplied.

## Failure cases and nonclaims

- **Aliasing:** 52 Hz sampled at 60 fps appears at 8 Hz. The sample sequence cannot recover the true frequency. A frame-rate label and Nyquist check do not provide an anti-aliasing guarantee.
- **Insufficient spatial information:** flat patches or one-dimensional texture do not support a reliable two-axis translation fit. Observation gates reject them rather than issuing artificial certainty.
- **Illumination and deformation:** brightness nuisance fitting handles only limited global gain/offset changes. Specular motion, shadows, occlusion, rotation, strain and large displacements violate the rigid-patch model.
- **Damping:** an isolated exponentially decaying free response can support a log-envelope fit. A forced response or poor decay fit does not establish damping. A dominant spectral response is not proof of resonance.
- **Acquisition:** reference subtraction only addresses image-plane translation. Rolling shutter, exposure attenuation, perspective, lens distortion, frame drops and out-of-plane motion remain material limits.
- **Physical units:** a user-supplied in-plane mm/px scale converts pixel estimates, but does not validate the camera geometry or include calibration uncertainty. The synthetic experiments do not validate millimeter accuracy.
- **Clinical or industrial diagnosis:** none. The anomaly score is a reference distance, not a fault probability, safety assessment or remaining-life prediction.

## Reproduce

From the repository root, after `./setup.sh`:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/benchmark.py
.venv/bin/python -m pytest -q
```

The smaller `scripts/benchmark.py --quick --output /tmp/vibescope-smoke` is a CI smoke experiment. Its smaller sample does not reproduce the full-release table.

To retrain the compact estimator into a separate location while preserving the supplied checkpoint:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/train_model.py \
  --samples 10000 --epochs 140 --output experiments/retrained/noise_fusion.npz
```

To reproduce the optional external neural comparison, install the free optional dependencies and check out the exact source version:

```sh
.venv/bin/python -m pip install -e '.[accelerated]'
git clone https://github.com/VUT-HFUT/EulerMormer.git /tmp/vibescope-EulerMormer
git -C /tmp/vibescope-EulerMormer checkout 821385981c25e7ed89a082be2cc48e65be1a0870
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/benchmark_neural.py \
  --source /tmp/vibescope-EulerMormer
```

The pinned repository contains the checkpoint used by the evaluator. See [FOUNDATIONS.md](FOUNDATIONS.md) and [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for the source/licensing boundaries. Published tables must be regenerated when their recorded source hashes no longer match the code being presented.

A signal-only implementation change can be audited without repeating image inference using `scripts/benchmark.py --refresh-signals` and `scripts/benchmark_neural.py --source /tmp/vibescope-EulerMormer --refresh-signals`. Each refresh verifies the retained inference-source hashes and records the new spectral/report hashes. Changes to estimation, weights or reference screening require the full corresponding experiment.
