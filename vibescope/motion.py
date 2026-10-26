"""Translation estimators. Coordinates: x right, y down; displacement in pixels.

These assume a rigid textured patch and brightness constancy. The physical solver
jointly fits brightness gain/offset, then subpixel shift using cubic interpolation.
The phase baseline is an original narrowband quadrature implementation, not an
implementation of a cited paper's complete steerable-pyramid algorithm.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def _validate(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or min(a.shape) < 16:
        raise ValueError("Matching grayscale patches of at least 16 × 16 pixels are required.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Frames contain invalid values.")
    return a, b


def estimate_pair(reference, current, **_):
    a, b = _validate(reference, current)
    # A modest blur suppresses pixel noise; both observations use the same kernel.
    a, b = gaussian_filter(a, .8), gaussian_filter(b, .8)
    yy, xx = np.mgrid[5:a.shape[0]-5, 5:a.shape[1]-5]
    # Bound computation on large ROIs without changing displacement units.
    stride = max(1, int(np.sqrt(xx.size / 6000)))
    yy, xx = yy[::stride, ::stride], xx[::stride, ::stride]
    target = a[yy, xx].ravel()
    gy, gx = np.gradient(b)
    shift = np.zeros(2)
    condition = 0.
    for _ in range(15):
        coords = [yy + shift[1], xx + shift[0]]
        sampled = map_coordinates(b, coords, order=3, mode='reflect').ravel()
        design_light = np.column_stack([sampled, np.ones(sampled.size)])
        gain, offset = np.linalg.lstsq(design_light, target, rcond=None)[0]
        residual = gain * sampled + offset - target
        j = gain * np.column_stack([
            map_coordinates(gx, coords, order=1, mode='reflect').ravel(),
            map_coordinates(gy, coords, order=1, mode='reflect').ravel()])
        gram = j.T @ j
        ev = np.linalg.eigvalsh(gram)
        condition = float(ev[0] / max(ev[-1], 1e-15))
        if ev[0] < 1e-8 or condition < 1e-5:
            return dict(dx=0., dy=0., sigma_px=10., quality=0., condition=condition)
        step = np.linalg.solve(gram + np.eye(2)*1e-12, j.T @ residual)
        shift -= np.clip(step, -1., 1.)
        if np.linalg.norm(step) < 1e-5:
            break
    sigma = np.sqrt(np.mean(residual**2))
    cov = np.linalg.pinv(gram) * sigma**2
    # Residual/texture quality detects flat patches and serious model mismatch.
    texture = np.std(target)
    quality = float(np.clip(1 - sigma / max(texture, 1e-9), 0, 1))
    if np.max(np.abs(shift)) > 4:
        quality *= .1
    return dict(dx=float(shift[0]), dy=float(shift[1]),
                sigma_px=float(np.sqrt(np.max(np.diag(cov)))),
                quality=quality, condition=condition)


def analytic_band(image, axis='x', wavelength=12.):
    """Complex one-sided oriented Fourier band; frequencies in cycles/pixel."""
    image = np.asarray(image, float)
    fy = np.fft.fftfreq(image.shape[0])[:, None]
    fx = np.fft.fftfreq(image.shape[1])[None, :]
    k, orth = (fx, fy) if axis == 'x' else (fy, fx)
    center = 1. / wavelength
    filt = 2 * np.exp(-.5*((k-center)/(center*.45))**2) * np.exp(-.5*(orth/(center*.55))**2) * (k > 0)
    return np.fft.ifft2(np.fft.fft2(image) * filt)


def phase_pair(reference, current, axis='x'):
    a, b = _validate(reference, current)
    estimates = []
    for wavelength in (8., 12., 20.):
        ar, ac = analytic_band(a, axis, wavelength), analytic_band(b, axis, wavelength)
        dim = 1 if axis == 'x' else 0
        derivative = np.gradient(ar, axis=dim)
        k = np.imag(np.conj(ar) * derivative) / (np.abs(ar)**2 + 1e-12)
        phase = np.angle(ac * np.conj(ar))
        weight = np.minimum(np.abs(ar), np.abs(ac))**2
        valid = np.zeros(a.shape, bool); valid[5:-5, 5:-5] = True
        valid &= (np.abs(k) > .03) & (np.abs(k) < 1.5)
        den = np.sum(weight[valid] * k[valid]**2)
        shift = -np.sum(weight[valid] * k[valid] * phase[valid]) / max(den, 1e-15)
        err = np.sqrt(np.sum(weight[valid]*(phase[valid]+k[valid]*shift)**2)/max(np.sum(weight[valid]), 1e-15))
        estimates.append((den, float(shift), float(err)))
    den, shift, err = max(estimates)
    quality = float(np.clip(1-err, 0, 1)) if den > 1e-7 else 0.
    return dict(dx=shift if axis == 'x' else 0., dy=shift if axis == 'y' else 0.,
                sigma_px=float(err / .4), quality=quality)


def validate_roi(roi, shape):
    if len(roi) != 4 or any(not np.isfinite(v) or int(v) != v for v in roi):
        raise ValueError("Region must contain four integer pixel coordinates.")
    x,y,w,h = map(int, roi)
    if x < 0 or y < 0 or min(w,h) < 24 or x+w > shape[1] or y+h > shape[0]:
        raise ValueError("Region must be inside the image and at least 24 × 24 pixels.")
    return x,y,w,h


def track(frames, roi, method='classical', axis='x', device='cpu'):
    x,y,w,h = validate_roi(roi, frames.shape[1:])
    a = frames[0,y:y+h,x:x+w]
    values, qualities, sigmas = [], [], []
    for image in frames:
        b = image[y:y+h,x:x+w]
        if method == 'learned':
            from .learned import estimate_pair as learned_pair
            result = learned_pair(a, b, device=device)
        elif method == 'phase':
            result = phase_pair(a,b,axis)
        elif method == 'classical':
            result = estimate_pair(a,b)
        else:
            raise ValueError("Unknown measurement method.")
        values.append(result['dx' if axis == 'x' else 'dy'])
        qualities.append(result['quality']); sigmas.append(result['sigma_px'])
    return np.array(values), np.array(qualities), np.array(sigmas)
