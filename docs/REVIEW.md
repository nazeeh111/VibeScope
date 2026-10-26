# Instrument verification audit

Reviewed 2026-10-25. **Evidence is controlled synthetic motion and exact mathematical signals. No independently captured physical experiment or reference displacement instrument was available.** These checks establish the behavior tested here, not real-world measurement accuracy or diagnostic reliability.

## Independently verified

- **48 project tests passed** in the complete release check. Independent reruns covered **38 focused tests** across signal processing, translation measurement, the trained estimator and reference screening. They cover both motion axes, independent camera-translation subtraction, lighting changes, flat/aperture rejection, reference-model metadata, ringdown assumptions, sampling aliases and invalid controls.
- A separate adversarial probe recovered an exact 7.3 Hz, 0.25 px sine to numerical precision. Pure 1, 2, 3, 15 and 20 Hz motion did not become a frequency measurement in a selected 4–12 Hz band. None of 100 seeded white-noise signals produced a resolved tone. This finite test does not calibrate a false-alarm probability.
- Actual encoded videos containing a one-frame presentation-time gap and a duplicate timestamp were rejected. Unknown timing remains explicitly unverified, and real inputs require a capture-rate declaration.
- An input placed at the usual output filename remained byte-for-byte unchanged when the occupied destination was rejected. A failed analysis left neither a final destination nor its temporary stage. The decoded-frame cap rejected excess work without silently skipping frames.
- Seventeen occluded frames out of 180, concentrated near vibration peaks, were rejected before a measurement was issued. Retaining zero-displacement placeholders would underestimate the 0.25 px amplitude by about 19%; the instrument instead fails when any sample is unobservable.
- A reference trained at 60 fps rejected a 120 fps analysis. A different confirmed acquisition scope was rejected; matching scope was recorded as user-confirmed. Malformed reference metadata, insufficient calibration records and unsupported recording lengths are covered by tests.
- A complete encoded synthetic demo measured **7.29995966 Hz and 0.27989206 px**, against rendering parameters **7.3 Hz and 0.28 px**. The source hash matched the supplied recording, all seven result artifacts were present, and no frame was unobservable. This validates this controlled clip through decoding and export, not a physical recording.
- The saved controlled and external-neural comparison reports were complete, with recorded source/checkpoint hashes matching the audited files. Every compared sequence had zero unobservable frames.
- Two additional API checks passed, including excessive declared video length and an actual chunked multipart body exceeding the receive limit before route processing.
- All 1,500 reserved synthetic model-test pairs were regenerated independently. The supplied weights match their recorded SHA-256, and the model card's metrics reproduce to floating-point precision: vector RMSE **0.07704698 px**, versus **0.08427644 px** without learned calibration. This ablation supports the learned correction on that distribution only.

The main regression checks are reproducible with:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q
```

The model generator, seed splits, checkpoint hash and training command are documented in [MODEL_CARD.md](MODEL_CARD.md). Broader controlled comparisons and their limitations are described in the [project README](../README.md).

## Correctness policies checked

Measurements use decoded original frames. Optional reference subtraction removes image-plane camera translation. Optional mm/px conversion scales the original displacement; it does not supply physical calibration by itself. Display gain, magnified frames and the intensity-activity heatmap do not feed back into measurement.

A resolved frequency requires an interior spectral peak, at least 15 dB peak-to-median PSD and a sine fit explaining at least 15% of full-signal detrended variance. These conservative engineering gates prevent the tested leakage and noise counterexamples. They can leave genuine weak modes or band-edge modes unresolved. Widening the band can help with a boundary peak; it cannot recover sampling aliases or poor observations.

The fitted compact model contains 362 trained parameters and runs locally with NumPy. Its reported error scale comes from a separate synthetic calibration split. It is not a physical standard deviation or a confidence interval. The approximate phase method is a baseline with measurable amplitude bias, not a reproduction of a full steerable-pyramid magnification method.

Analysis publishes a complete fresh output directory only after successful processing. Existing destinations and source recordings are preserved. Upload bytes are bounded before multipart parsing, including requests without a declared length. Decoded frames, dimensions and aggregate pixel count are also bounded.

## Material limits

- Regular presentation timestamps cannot prove that every physical exposure was retained. Uniform frame decimation, duplicated exposures, slow-motion playback rates and motion above Nyquist can remain ambiguous. Capture timing needs independent evidence.
- Translation models assume a textured rigid patch and small image-plane motion. Rotation, parallax, rolling shutter, exposure blur, changing focus, occlusion and internal deformation can invalidate the estimate. Quality gates do not detect every model violation.
- Ringdown damping is conditional on an isolated linear free response declared by the user. A strong exponential-envelope fit does not prove absence of forcing or multiple modes. The reported natural frequency and force-to-displacement resonance estimate are conditional single-degree-of-freedom model quantities, checked against an analytic ringdown; they are not an experimentally measured resonance. A response peak alone does not establish resonance.
- A baseline scope declaration cannot prove matching camera, region, axis, scale, exposure or processing method. An outside-reference score is novelty relative to supplied recordings, not damage identification or a diagnostic probability. Small calibration sets do not resolve rare false-alarm rates.
- Browser MP4s are display exports. CRF0 synthetic-source coding preserves converted YUV, not exact RGB pixels; RGB/YUV conversion and chroma subsampling remain. Browser exports use additional encoding. Quantization and compression can materially alter subpixel signals.
- No claim is made of generic real-structure accuracy, universal noise rejection, superiority over larger neural models, or readiness for safety-critical decisions.

## Reviewed source fingerprints

These fingerprints identify the audited source. The spectral resolver and conditional ringdown fields received separate analytic regression checks; the formatted backend retains the reviewed behavior:

| File | SHA-256 |
|---|---|
| `vibescope/signals.py` | `9d0071e36367ae5548be56eb1e8ab7e4d12fd40fea4d3971dccc6db2f21960b3` |
| `vibescope/motion.py` | `c71d088fe7f0c3d9fc5871369dad8d03b68c5f890b71385ca60607d3e7ad02de` |
| `vibescope/learned.py` | `6995980c8fb348561681018bd2413244d07b3661a8ef717d53b4caac4e12630b` |
| `vibescope/pipeline.py` | `667a7aac4011ec4044f7d3bcde5bbb0a76be5cf3489f9e9574f65ff585d1069b` |
| `vibescope/video.py` | `92d03ed1711786887e921464eabc800f9e3212b2b195b4ebd33b4db2d65e2159` |
| `vibescope/anomaly.py` | `043b7732311f817b4edc27f20c38e78ae403c9737929734a5cb33f3348a6b22a` |
| `vibescope/server.py` | `5082683657b3728e884d0eea8e1436da4f5d33a67d550a92cfa0174a301ee9fe` |
| `vibescope/models/noise_fusion.npz` | `194fb84f67d2b52ef94a207cf655f8c09a7114bae0f1e926c6fd144b407589d8` |

Later source changes require checks appropriate to the behavior changed; this audit does not automatically transfer to a different revision.

## Final release regression

After the original audit, complete learned/phase inference-and-export checks were added. The current full local suite passed **50 tests**. The complete browser workflow, baseline screening, mobile rendering, actual H264 playback and installed wheel resources were also verified. These checks establish software behavior within the stated controlled-synthetic scope.
