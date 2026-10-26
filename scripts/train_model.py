#!/usr/bin/env python3
"""Reproducible synthetic supervision for the compact noise-aware estimator."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.ndimage import gaussian_filter, fourier_shift
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vibescope.learned import pair_features, _forward, FEATURE_NAMES


def sample_pair(seed, size=40, max_shift=0.95, noise_max=0.055):
    """Independent texture per seed; Fourier translation precedes independent noise."""
    rng = np.random.default_rng(seed)
    side = size + 24
    texture = gaussian_filter(rng.normal(size=(side, side)), rng.uniform(0.55, 2.6))
    # Distinct mixtures include fine/coarse texture and discontinuous boundaries.
    texture += gaussian_filter(rng.normal(size=(side, side)), rng.uniform(1, 5)) * rng.uniform(0, 1)
    if seed % 3 == 0:
        yy, xx = np.mgrid[:side, :side]
        angle = rng.uniform(0, 2 * np.pi)
        texture += (xx * np.cos(angle) + yy * np.sin(angle) > rng.uniform(-side, side)) * rng.uniform(0.1, 0.6)
    texture = (texture - texture.mean()) / max(texture.std(), 1e-8)
    texture = 0.5 + texture * rng.uniform(0.045, 0.15)
    delta = rng.uniform(-max_shift, max_shift, 2)
    if seed % 10 == 0:
        delta[:] = 0  # Static clips are supervised too.
    shifted = np.fft.ifftn(fourier_shift(np.fft.fftn(texture), [delta[1], delta[0]])).real
    noise = rng.uniform(0, noise_max)
    sl = (slice(12, -12), slice(12, -12))
    a = np.clip(texture[sl] + rng.normal(0, noise, (size, size)), 0, 1)
    b = np.clip(shifted[sl] + rng.normal(0, noise, (size, size)) + rng.uniform(-0.008, 0.008), 0, 1)
    return a.astype(np.float32), b.astype(np.float32), delta, noise


def make_dataset(start_seed, count):
    x, y, bases, noises = [], [], [], []
    for seed in range(start_seed, start_seed + count):
        a, b, delta, noise = sample_pair(seed)
        features, base, _ = pair_features(a, b)
        x.append(features); y.append(delta - base); bases.append(base); noises.append(noise)
    return np.array(x), np.array(y), np.array(bases), np.array(noises)


def fit(x, y, epochs, seed=20261025):
    rng = np.random.default_rng(seed)
    mean, std = x.mean(0), np.maximum(x.std(0), 1e-4)
    x = np.clip((x - mean) / std, -8, 8)
    model = {"mean": mean, "std": std,
             "w1": rng.normal(0, 0.15, (x.shape[1], 24)), "b1": np.zeros(24),
             "w2": rng.normal(0, 0.004, (24, 2)), "b2": np.zeros(2)}
    keys = ("w1", "b1", "w2", "b2")
    moments = {k: [np.zeros_like(model[k]), np.zeros_like(model[k])] for k in keys}
    step = 0
    for epoch in range(epochs):
        for indices in np.array_split(rng.permutation(len(x)), max(1, len(x) // 256)):
            xb, yb = x[indices], y[indices]
            h = np.tanh(xb @ model["w1"] + model["b1"])
            pred = h @ model["w2"] + model["b2"]
            # Huber clipping limits occasional aperture/degenerate examples.
            d = np.clip(pred - yb, -0.15, 0.15) / len(xb)
            dh = (d @ model["w2"].T) * (1 - h**2)
            grads = {"w2": h.T @ d + 2e-5 * model["w2"], "b2": d.sum(0),
                     "w1": xb.T @ dh + 2e-5 * model["w1"], "b1": dh.sum(0)}
            step += 1
            lr = 0.002 * (0.3 + 0.7 * (1 - epoch / epochs))
            for key in keys:
                m, v = moments[key]
                m[:] = 0.9 * m + 0.1 * grads[key]
                v[:] = 0.999 * v + 0.001 * grads[key]**2
                model[key] -= lr * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
    return model


def evaluate(x, y, bases, model, noises):
    error = np.clip(_forward(x, model), -0.3, 0.3) - y
    baseline = -y
    def stats(e):
        return {"vector_rmse_px": float(np.sqrt(np.mean(np.sum(e**2, axis=1)))),
                "median_endpoint_error_px": float(np.median(np.linalg.norm(e, axis=1))),
                "p95_endpoint_error_px": float(np.quantile(np.linalg.norm(e, axis=1), .95)),
                "bias_x_px": float(e[:, 0].mean()), "bias_y_px": float(e[:, 1].mean())}
    return {"noise_fusion": stats(error), "without_learned_calibration": stats(baseline),
            "by_noise": [{"range": [lo, hi], "n": int(((noises >= lo) & (noises < hi)).sum()),
                          "noise_fusion": stats(error[(noises >= lo) & (noises < hi)]),
                          "without_learned_calibration": stats(baseline[(noises >= lo) & (noises < hi)])}
                         for lo, hi in [(0, .015), (.015, .035), (.035, .056)]]}, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--output", type=Path, default=Path("experiments/retrained/noise_fusion.npz"))
    args = parser.parse_args()
    if args.samples < 100 or args.epochs < 1:
        parser.error("Use at least 100 training samples and one epoch.")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("Choose fresh checkpoint and metadata paths; existing models are never overwritten.")
    started = time.perf_counter()
    x, y, bases, noises = make_dataset(10000, args.samples)
    model = fit(x, y, args.epochs)
    # Calibration seeds are not reused in fit or final test.
    cx, cy, cb, cn = make_dataset(300000, 1000)
    calibration, errors = evaluate(cx, cy, cb, model, cn)
    model["error_scale_px"] = np.array(np.quantile(np.linalg.norm(errors, axis=1), .90))
    tx, ty, tb, tn = make_dataset(600000, 1500)
    test, errors = evaluate(tx, ty, tb, model, tn)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **model)
    report = {"model": "noise-fusion-mlp-v1", "architecture": "12-24-2 tanh residual MLP",
              "trainable_parameters": int(sum(model[k].size for k in ["w1", "b1", "w2", "b2"])),
              "feature_names": list(FEATURE_NAMES), "train": {"seed_start": 10000, "count": args.samples},
              "calibration": {"seed_start": 300000, "count": 1000, "metrics": calibration},
              "test": {"seed_start": 600000, "count": 1500, "metrics": test},
              "optimizer": "NumPy Adam; robust displacement loss; L2 2e-5", "epochs": args.epochs,
              "optimizer_seed": 20261025, "max_shift_per_axis_px": .95, "noise_sigma_range": [0, .055],
              "error_scale_px": float(model["error_scale_px"]),
              "elapsed_seconds": time.perf_counter() - started,
              "weights_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
              "evidence_scope": "Controlled synthetic translated patches only; no real-world validation.",
              "limitations": ["Rigid translation only", "No out-of-plane/rolling-shutter/occlusion correction", "Synthetic error scale is not a confidence interval", "No claim of publication-level novelty"]}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
