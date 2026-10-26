from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="vibescope",
        description="Local experimental video vibration instrument. No cloud API or diagnostic claims.",
    )
    sub = parser.add_subparsers(dest="command")
    serve = sub.add_parser("serve", help="Launch local workspace (default)")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--data-dir", type=Path, default=Path("runs"))
    demo = sub.add_parser(
        "demo", help="Generate a labeled controlled synthetic recording"
    )
    demo.add_argument("--output", type=Path, default=Path("examples"))
    measure = sub.add_parser("analyze", help="Measure a local original video")
    measure.add_argument("video", type=Path)
    measure.add_argument("--output", type=Path, required=True)
    measure.add_argument("--roi", nargs=4, type=int, required=True)
    measure.add_argument("--reference-roi", nargs=4, type=int)
    measure.add_argument("--axis", choices=["x", "y"], default="x")
    measure.add_argument("--band", nargs=2, type=float, default=[4, 12])
    measure.add_argument("--gain", type=float, default=20)
    measure.add_argument(
        "--method", choices=["classical", "phase", "learned"], default="classical"
    )
    measure.add_argument("--mm-per-px", type=float)
    measure.add_argument("--fps", type=float)
    measure.add_argument("--confirm-cfr", action="store_true")
    measure.add_argument(
        "--synthetic",
        action="store_true",
        help="Label input synthetic; never use for a real recording",
    )
    measure.add_argument("--ringdown", action="store_true")
    measure.add_argument(
        "--baseline",
        type=Path,
        help="Same-setup healthy reference model from baseline command",
    )
    measure.add_argument(
        "--baseline-scope",
        help="Exact scope string confirming identical acquisition setup",
    )
    baseline = sub.add_parser(
        "baseline",
        help="Train healthy-reference novelty model from >=15 original measurement CSVs",
    )
    baseline.add_argument("csvs", nargs="+", type=Path)
    baseline.add_argument("--fps", type=float, required=True)
    baseline.add_argument("--scope", required=True)
    baseline.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.command in (None, "serve")
        and not 1 <= getattr(args, "port", 8765) <= 65535
    ):
        parser.error("Port must be between1 and65535.")
    if args.command in (None, "serve"):
        import uvicorn
        from .server import create_app

        port = getattr(args, "port", 8765)
        print(
            f"VibeScope → http://127.0.0.1:{port}\nLocal processing. Ctrl+C to stop. Data stays in runs/ unless --data-dir is supplied."
        )
        uvicorn.run(
            create_app(getattr(args, "data_dir", Path("runs"))),
            host="127.0.0.1",
            port=port,
        )
    elif args.command == "demo":
        from .synthetic import make_demo

        path, _ = make_demo(args.output)
        print(path)
    elif args.command == "analyze":
        from .pipeline import analyze

        if args.baseline and not args.baseline_scope:
            parser.error(
                "Confirm identical acquisition using --baseline-scope with the exact reference scope."
            )
        result = analyze(
            args.video,
            args.output,
            args.roi,
            args.reference_roi,
            args.axis,
            *args.band,
            args.gain,
            args.mm_per_px,
            args.method,
            args.ringdown,
            args.fps,
            args.confirm_cfr,
            "synthetic" if args.synthetic else "user_recording",
            args.baseline,
            baseline_scope=args.baseline_scope,
        )
        print(json.dumps(result["metrics"], indent=2))
        print(f"Artifacts: {args.output.resolve()}")
    else:
        import numpy as np
        from .anomaly import train_reference

        if len(args.csvs) < 15:
            parser.error(
                "Provide at least 15 independent same-setup healthy recordings, not windows from a single recording."
            )
        if args.output.exists() or args.output.is_symlink():
            parser.error("Reference output exists; choose a fresh output path.")
        records = [
            np.genfromtxt(p, delimiter=",", names=True)["raw_displacement_px"]
            for p in args.csvs
        ]
        split = max(10, int(len(records) * 0.7))
        model = train_reference(records[:split], records[split:], args.fps, args.scope)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(model, indent=2))
        print(args.output)


if __name__ == "__main__":
    main()
