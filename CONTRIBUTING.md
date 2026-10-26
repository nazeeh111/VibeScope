# Contributing

Keep measurement and rendering separate. Use original frames for quantitative estimates; display gain must never change measured displacement. Do not add diagnostic, physical-accuracy or superiority claims without corresponding evidence.

1. Install with `./setup.sh`.
2. Run `.venv/bin/python -m pytest -q`.
3. For estimator/signal changes, run the controlled benchmark and keep disjoint test seeds. Report rejected measurements as well as successful cases.
4. For UI changes, run the browser smoke procedure in `tests/ui/README.md`, inspect desktop/mobile screenshots and retain source provenance.
5. Describe the concrete behavior change and relevant checks in a pull request. Preserve license notices and avoid committing recordings without consent and provenance.

Use a fresh results directory for each run. Keep source recordings out of Git by default. Baseline checkpoints stay external; the original compact model may be regenerated using the documented training command.
