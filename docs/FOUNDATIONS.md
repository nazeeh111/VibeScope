# Research foundations and reproducible baseline

Source review: 2026-10-25. These repositories are research foundations, not evidence that VibeScope meets their reported results. No upstream app, model source, or weights are bundled into the core application. VibeScope's displacement estimator is original code. The optional EulerMormer runner invokes the unmodified external model.

## Inspected revisions

| Foundation | Pinned source revision | License at that revision | What informs VibeScope |
|---|---|---|---|
| [STB-VMM](https://github.com/RLado/STB-VMM/tree/af2712392a948fc188f36d8429082b759d27f548) | `af2712392a948fc188f36d8429082b759d27f548` | MIT, Ricard Lado, 2022 | Noise is a primary obstacle in subpixel video motion. Swin Transformer magnification is a useful visual baseline, but its image-quality claims do not establish displacement metrology. |
| [EulerMormer, AAAI 2024](https://github.com/VUT-HFUT/EulerMormer/tree/821385981c25e7ed89a082be2cc48e65be1a0870) | `821385981c25e7ed89a082be2cc48e65be1a0870` | MIT, Fei Wang, 2024 | Dynamic filtering suppresses noise in learned motion representations. Selected external neural baseline because this revision includes a compact pretrained checkpoint and an explicit license. |
| [Axial-MM, ECCV 2024](https://github.com/kaist-ami/Axial-mm/tree/f6d2ce387d6f7f4e4840905d35019cc809c6ce00) | `f6d2ce387d6f7f4e4840905d35019cc809c6ce00` | No repository LICENSE file found in inspected revision | Directional control and temporal filters motivate explicit axis and frequency controls. Source and included weights are not redistributed or incorporated. Public availability is not a license grant. |
| [ViMag](https://github.com/RLado/ViMag/tree/fa1503be8ed56639dd10d6ab7b5d62a1c7d3ef9b) | `fa1503be8ed56639dd10d6ab7b5d62a1c7d3ef9b` | MIT, Ricard Lado, 2022 | A practical workflow connects selected video regions, displacement histories, spectra, and CSV. Its lab damage-detection claims belong to its own study, not VibeScope. |

### Source-level findings

- STB-VMM `run.py` has CPU/CUDA selection. `models/model.py` manipulates feature differences with `(amp - 1)` and adds them to the second frame's representation. Its `mag` is intended as total output gain. The README requires a checkpoint, but no checkpoint is tracked in this inspected source. Pinned requirements include torch 1.13.1 and torchvision 0.14.1, so blindly installing them into a current Python environment is inappropriate.
- EulerMormer `models/magnet.py` exposes `MagNet(a, b, amp, x_c, mode='evaluate')`. Inputs are RGB tensors in `[-1,1]`, shaped `N,C,H,W`. `data.py` confirms normalization and RGB conversion. `PEM` uses `(amp - 1)`, and `Manipulator` adds this to the current motion encoding. An output at `amp=1` still passes through learned reconstruction, so pixel-perfect identity is not guaranteed. The published `test_video.py` assumes CUDA and uses `strict=False`; VibeScope's external runner calls the same model directly on CPU and loads the checkpoint with `strict=True`. No upstream code is patched.
- Axial-MM `module.py` projects learned x/y features onto an angle `theta`, multiplies differences by separate maps, and projects back. Its raw feature gain convention differs from EulerMormer's `(amp-1)` convention. The README's examples use `alpha_x`, `alpha_y`, `theta`, and temporal filters; do not equate these numbers with verified physical output displacement. The README explicitly leaves its physical-accuracy experiment code as a TODO at this revision.
- ViMag is an Electron/Python toolbox. Its `python/csv-fft.py` exposes sample rate, frequency range, window choice and CSV output. It does not make video displacement uncertainty or diagnostic performance transferable to another program.

## Classical phase foundation

[Wadhwa et al., *Phase-Based Video Motion Processing*, SIGGRAPH 2013](https://people.csail.mit.edu/nwadhwa/phase-video/) associates motion with phase changes of complex steerable-pyramid coefficients, temporally filters their phase, and reconstructs magnified video. The authors also provide MATLAB code and source videos. This establishes the method, not permission to silently relabel a different algorithm as its reproduction.

A global Fourier phase registration estimator and a local oriented phase filter are useful classical baselines, but they are not the complete Wadhwa steerable-pyramid implementation. VibeScope labels its actual algorithm and scope. Magnification must be evaluated separately from the original-video displacement estimate. A convincing rendering cannot establish true physical gain.

## Optional EulerMormer reproduction

The ordinary application needs no PyTorch. Use a separate research environment, or install the optional dependencies into the project environment:

```bash
python -m pip install torch einops
mkdir -p /tmp/vibescope-baselines
git clone https://github.com/VUT-HFUT/EulerMormer.git /tmp/vibescope-baselines/EulerMormer
git -C /tmp/vibescope-baselines/EulerMormer checkout 821385981c25e7ed89a082be2cc48e65be1a0870
python scripts/run_neural_baseline.py \
  --source /tmp/vibescope-baselines/EulerMormer \
  --input examples/controlled.mp4 \
  --output /tmp/vibescope-eulermormer.avi \
  --roi 96 64 64 64 --alpha 10 --max-frames 120 --device cpu
```

The ROI above is an example, not an asserted best ROI. Choose a textured region inside the moving specimen. The runner verifies both model and checkpoint bytes against the inspected source, calls `torch.load(..., weights_only=True)`, and makes no network requests. It writes an MJPEG video and adjacent JSON containing input/output hashes, model revision, parameters, frame count, frame rate, ROI, target gain and measured runtime. It can use CPU, MPS or CUDA if available. The CPU path is the tested baseline; optional accelerators do not change the measurement contract.

Pinned external files:

| File | SHA-256 |
|---|---|
| `models/magnet.py` | `ac7d5691698f0f04c75ab88a54226d672e9caf1bf226d4464eed3224472ca76c` |
| `weights_dateEuler/magnet_epoch99_loss5.38e-02.pth` | `0e789923b3d59cf8d64c65ed68072c6b152d6f158716aa1c322cf36b42dd5bf6` |

The model uses stride-two encoding. Odd input dimensions receive reflected padding to even sizes and are cropped back afterward. No resizing is performed, so pixel scale is preserved. The first frame remains the reference throughout. The output is a learned rendering at the requested total gain, without temporal filtering. This is a deliberately bounded baseline, not reproduction of the authors' complete evaluation suite. For comparable displacement assessment, estimate motion between its reconstructed initial output and later outputs, then divide by requested gain; report reconstruction error, failed tracking and requested-versus-measured gain, rather than treating target gain as truth.

## Comparison contract

Use identical source frames and ROIs, fixed seeds, stated frame rate, and a held-out synthetic sequence for all compared estimators. Keep original-video estimation metrics separate from magnified-output image metrics. Record:

1. Frequency error in Hz and relative error, with observation length and FFT bin spacing.
2. Displacement RMSE and amplitude bias in source pixels; millimetres only with in-plane calibration.
3. Output gain and reconstruction artifacts for magnification baselines, explicitly distinct from sensor measurements.
4. Noise level, texture, camera translation, exposure/blur, compression, frame count, and processing dimensions.
5. Wall-clock inference throughput, hardware, thread count, device and warm-up policy.
6. Held-out anomaly labels and seed/recording separation. Synthetic anomaly scores support only the specified simulation family.

Do not borrow paper accuracy numbers into VibeScope tables. An unavailable or failed baseline must remain `not run`, with the reason. None of these source projects supplies the independent calibration experiment, camera/exposure characterization or diagnostic validation required for a professional physical measurement claim.

## Bibliography

- Lado-Roigé, R. and Pérez, M. A. (2023). [STB-VMM: Swin Transformer Based Video Motion Magnification](https://doi.org/10.1016/j.knosys.2023.110493). *Knowledge-Based Systems*.
- Wang, F., Guo, D., Li, K. and Wang, M. (2024). [EulerMormer: Robust Eulerian Motion Magnification via Dynamic Filtering within Transformer](https://arxiv.org/abs/2312.04152). *AAAI* 38(6), 5345–5353.
- Kwon, B.-K., Oh, H.-B., Kim, J.-S., Ha, H. and Oh, T.-H. (2024). [Learning-based Axial Video Motion Magnification](https://arxiv.org/abs/2312.09551). *ECCV*.
- Wadhwa, N., Rubinstein, M., Durand, F. and Freeman, W. T. (2013). [Phase-Based Video Motion Processing](https://people.csail.mit.edu/nwadhwa/phase-video/). *ACM Transactions on Graphics* 32(4).
