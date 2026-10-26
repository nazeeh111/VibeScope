# VibeScope 0.1.0

A local video vibration workspace with original-frame measurement, frequency-selective visualization and a trained compact motion estimator.

## Included

- Classical iterative, quadrature-phase and 362-parameter learned motion estimation.
- Independent stationary-region translation subtraction, capture-rate controls and optional declared mm/px calibration.
- Conservative tone resolution, isolated-ringdown damping, conditional natural-frequency/SDOF estimates and same-setup reference novelty screening.
- Synchronized video, selectable regions, displacement/spectrum inspection and CSV/PNG/JSON/video exports.
- Portable CPU inference, deterministic synthetic training, pinned optional EulerMormer CPU/MPS/CUDA runner, benchmark data, tests and CI.
- Actual browser demonstration, screenshots, packaged controlled video and original-versus-estimated truth.

## Evidence

Local release suite: 50 tests passed, including complete classical, learned and phase inference/export paths. Browser smoke exercises actual analysis, playback/seek, exports, mobile layout and baseline screening. A clean wheel-resource smoke checks the installed interface, compact model and packaged demo. Repository CI runs the CPU suite and a controlled benchmark across Python 3.11–3.14.

The controlled decoded-video demo recovers 7.29996 Hz and 0.27989 px from a 7.3 Hz / 0.28 px synthetic translation. The model's separate 1,500-pair test improves its own uncorrected baseline by 8.58%; the stronger classical tracker remains the default. Full results, conditions, intervals, failure cases and upstream notices are included in the repository.

## Scope

This is a research release validated on controlled synthetic data. No independently measured physical experiment or fault-diagnostic validation is claimed. Derived natural/resonance values require the stated model assumptions. Display magnification and intensity activity maps are visualizations, never calibrated physical measurements.
