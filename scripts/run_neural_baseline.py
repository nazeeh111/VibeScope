#!/usr/bin/env python3
"""Run the unmodified, pinned EulerMormer model on a video/ROI.

Optional research dependency: pip install torch einops
Source and checkpoint stay external to the VibeScope distribution. No network calls.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time

COMMIT = '821385981c25e7ed89a082be2cc48e65be1a0870'
SOURCE_SHA256 = 'ac7d5691698f0f04c75ab88a54226d672e9caf1bf226d4464eed3224472ca76c'
WEIGHT_SHA256 = '0e789923b3d59cf8d64c65ed68072c6b152d6f158716aa1c322cf36b42dd5bf6'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(source: Path, device: str = 'cpu'):
    import torch
    model_file = source / 'models/magnet.py'
    weight_file = source / 'weights_dateEuler/magnet_epoch99_loss5.38e-02.pth'
    for path, expected in [(model_file, SOURCE_SHA256), (weight_file, WEIGHT_SHA256)]:
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'Expected pinned EulerMormer file/hash: {path}. See docs/FOUNDATIONS.md.')
    spec = importlib.util.spec_from_file_location('vibescope_external_eulermormer', model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.MagNet().to(device)
    state = torch.load(weight_file, map_location='cpu', weights_only=True)
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def infer_pair(model, first_bgr, current_bgr, alpha: float = 10, device: str = 'cpu'):
    """Return original-sized BGR float32 image in [0,1]. Alpha is total target gain."""
    import cv2
    import numpy as np
    import torch
    if first_bgr.shape != current_bgr.shape or first_bgr.ndim != 3:
        raise ValueError('Expected equal H,W,3 images')
    height, width = current_bgr.shape[:2]
    def tensor(frame):
        frame = cv2.copyMakeBorder(frame, 0, height % 2, 0, width % 2, cv2.BORDER_REFLECT_101)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1
        return torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None].to(device)
    a, b = tensor(first_bgr), tensor(current_bgr)
    amp = torch.tensor(alpha, dtype=torch.float32, device=device).reshape(1, 1, 1, 1)
    with torch.inference_mode():
        result = model(a, b, amp, None, mode='evaluate')[0].detach().cpu().numpy()
    result = np.clip((result.transpose(1, 2, 0) + 1) * .5, 0, 1)
    return result[:height, :width, ::-1].copy()


def run(args):
    import cv2
    import numpy as np
    import torch
    if not 1 <= args.alpha <= 100:
        raise ValueError('alpha must be in [1,100]')
    if not 2 <= args.max_frames <= 10000 or args.threads < 1:
        raise ValueError('Invalid frame/thread limit')
    torch.set_num_threads(args.threads)
    model = load_model(args.source, args.device)
    cap = cv2.VideoCapture(str(args.input))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not cap.isOpened() or not 0 < fps < 10000:
        raise ValueError('Could not decode video with valid frame rate')
    if args.fps is not None:
        if not 0 < args.fps < 10000:
            raise ValueError('fps must be positive and below 10000')
        fps = args.fps
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or args.output.with_suffix('.json').exists() or args.output.is_symlink():
        raise ValueError('Use a fresh output video and metadata path. Existing files are never overwritten.')
    if args.input.resolve() == args.output.resolve():
        raise ValueError('Input and output must be different files.')
    if args.output.suffix.lower() != '.avi':
        raise ValueError('Use .avi for MJPEG output; transcode separately for browser playback')
    writer = None
    times = []
    first = None
    start = time.perf_counter()
    try:
        for _ in range(args.max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            if args.roi:
                x, y, width, height = args.roi
                if min(x,y) < 0 or min(width,height) < 16 or x+width > frame.shape[1] or y+height > frame.shape[0]:
                    raise ValueError('ROI must be in bounds with width and height >=16')
                frame = frame[y:y+height, x:x+width]
            if min(frame.shape[:2]) < 16:
                raise ValueError('Frames must be at least 16x16')
            if first is None:
                if frame.shape[0] % 2 or frame.shape[1] % 2:
                    raise ValueError('MJPEG video export requires even ROI/frame dimensions; no silent pixel cropping is allowed')
                first = frame.copy()
                writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'MJPG'), fps, frame.shape[1::-1])
                if not writer.isOpened():
                    raise ValueError('Could not open MJPEG writer')
            tick = time.perf_counter()
            output = infer_pair(model, first, frame, args.alpha, args.device)
            times.append(time.perf_counter() - tick)
            writer.write(np.round(output*255).astype(np.uint8))
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if len(times) < 2:
        raise ValueError('Need at least two decoded video frames')
    metadata = {
        'baseline': 'EulerMormer (AAAI 2024), unmodified pinned model',
        'source_url': f'https://github.com/VUT-HFUT/EulerMormer/tree/{COMMIT}',
        'source_commit': COMMIT, 'model_sha256': SOURCE_SHA256, 'weights_sha256': WEIGHT_SHA256,
        'checkpoint': 'magnet_epoch99_loss5.38e-02.pth',
        'input_sha256': digest(args.input), 'output_sha256': digest(args.output),
        'frames': len(times), 'fps': fps, 'roi_xywh': args.roi,
        'alpha_total_target_gain': args.alpha, 'mode': 'static first-frame reference; no temporal filtering',
        'device': args.device, 'threads': args.threads, 'torch_version': torch.__version__,
        'platform': platform.platform(), 'parameters': sum(p.numel() for p in model.parameters()),
        'mean_inference_seconds': float(np.mean(times)), 'median_inference_seconds': float(np.median(times)),
        'inference_fps': float(1/np.mean(times)), 'total_run_seconds': time.perf_counter()-start,
        'limits': ['Rendered baseline is visualization, not independent physical evidence.',
                   'No calibrated real experiment or diagnostic validation.',
                   'Use original frames for measurement; output may contain learned reconstruction artifacts.',
                   'This runner does not reproduce authors\' full dataset evaluation or frequency filtering.'],
    }
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--roi', nargs=4, type=int, metavar=('X','Y','WIDTH','HEIGHT'))
    parser.add_argument('--alpha', type=float, default=10)
    parser.add_argument('--max-frames', type=int, default=120)
    parser.add_argument('--device', choices=['cpu','mps','cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--fps', type=float, help='Explicit frame-rate override if metadata is wrong')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
