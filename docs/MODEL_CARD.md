# Noise Fusion MLP v1

**Status:** trained experimental motion estimator. Evidence is controlled synthetic translation, not independently measured real structures. No diagnostic reliability, physical calibration, or publication-level novelty is claimed.

## What is learned

The new component is a compact, noise-aware residual calibration of three physical translation estimates. It estimates displacement in pixels from two original grayscale patches. It never sees magnified frames and never generates a displacement waveform or preferred frequency.

1. Smooth the mean image at Gaussian scales 0.65, 1.25 and 2.0 pixels.
2. Fit the local brightness-constancy equation jointly for horizontal and vertical translation, after subtracting a constant lighting offset.
3. Form twelve features: middle-scale displacement, fine/coarse differences, noise-to-gradient ratio, gradient energy, aperture conditioning, gradient correlation, residual and brightness offset.
4. A **12 → 24 → 2 tanh MLP**, with **362 trainable parameters**, predicts a residual correction to the middle-scale estimate. Limit each correction to ±0.3 pixels.

Training uses displacement labels directly, with robust displacement loss and weak weight regularization. This is original project code and actual fitted weights. It is a constrained engineering experiment, not an implementation of STB-VMM, EulerMormer or Axial-MM. No external neural checkpoint is embedded in this model.

## Data and reproducibility

Run from the repository root:

```sh
python scripts/train_model.py --samples 10000 --epochs 140
```

The generator makes a new random spatial texture per seed, mixes fine and coarse structure with occasional edges, applies the known translation in the Fourier domain, crops boundaries, and adds independent Gaussian sensor noise and small brightness offsets. Ten percent of pairs have zero displacement. Textures use 40 × 40 patches, translation uniform within ±0.95 pixels per axis, and grayscale Gaussian noise sigma 0–0.055 on a 0–1 scale.

| Split | Texture seeds | Use |
| --- | --- | --- |
| Training | 10,000–19,999 | Fit weights and input normalization |
| Calibration | 300,000–300,999 | Error-scale estimate only |
| Held-out test | 600,000–601,499 | Report final accuracy |

All optimizer randomness is seeded with 20261025. Final measured values, exact parameters and the weights SHA-256 are recorded in [`noise_fusion.json`](../vibescope/models/noise_fusion.json). A small analytic-texture test set in `tests/test_learned.py` also changes the generator family from the training generator.

## Observed held-out results

The delivered checkpoint was trained with the default command above in 257 seconds on this Mac. On 1,500 untouched synthetic texture seeds:

| Method | Vector RMSE (px) | Median endpoint error (px) | 95th percentile endpoint error (px) |
| --- | ---: | ---: | ---: |
| Noise Fusion MLP | 0.07705 | 0.04177 | 0.15581 |
| Without learned calibration | 0.08428 | 0.04859 | 0.17182 |

The overall RMSE reduction is **8.58% for this synthetic distribution**. Low-noise RMSE is 0.02714 px versus 0.03399 px; high-noise RMSE is 0.10766 px versus 0.11647 px. The weaker result at high noise and the 0.156-pixel tail error are material limits, especially for signals much smaller than these errors. These pairwise numbers are not accuracy guarantees for full video waveforms.

Ten estimator tests passed in the project environment, including independent analytic textures, noisy motion, lighting offsets, input limits and rejection of flat/one-dimensional patches. Inference was checked with the project environment's NumPy 2.5.4 / SciPy 1.18.1; weights were trained with the installed NumPy 2.3.5 / SciPy 1.18.1 runtime.

The baseline ablation removes the learned correction while retaining the same middle-scale physical estimator. This comparison measures the value of the fitted correction; it does not establish superiority over larger neural magnification models.

## Inference and evidence limits

```python
from vibescope.learned import estimate_pair
result = estimate_pair(reference_gray, current_gray)
```

Inputs are matching finite grayscale arrays normalized to 0–1 and at least 16 × 16 pixels. Positive x is image-right and positive y is image-down. The included `.npz` checkpoint contains ordinary arrays loaded with `allow_pickle=False`. NumPy inference needs no GPU, network connection, API key or billed service.

A low-texture patch, nearly one-dimensional edge (the aperture problem), or inferred translation above 1.3 pixels in either axis sets `quality=0` and `within_training_domain=False`. A caller must reject that sample or use a suitable independently checked tracking method. These gates do not detect every invalid scene. The model assumes approximate translation, stable lighting and visible texture; rotation, perspective change, occlusion, deformation inside one patch and strong compression can break that assumption.

`quality` is a heuristic based on photometric residual and estimated noise. `sigma_px` is the calibration-split 90th percentile endpoint error scaled by residual/noise diagnostics. **It is an error scale, not a standard deviation, confidence interval or demonstrated coverage guarantee.** It excludes calibration error, camera motion, rolling shutter, exposure blur, sampling aliases and real-world domain shift. Invalid patches receive a deliberately large 10-pixel scale and zero quality.

The model cannot detect a frequency above camera Nyquist, infer millimeters without a valid in-plane calibration, identify a mechanical resonance without a controlled excitation experiment, or diagnose a fault. Real measurements require separately documented capture, independent reference measurements and validation. Portable inference currently supports CPU only.
