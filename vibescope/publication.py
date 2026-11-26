"""Publish a complete local output without replacing a previous result."""
import logging
import os
from pathlib import Path
import tempfile


def write_new(path: Path, data: bytes) -> None:
    fd, stage = tempfile.mkstemp(prefix=".vibescope-", dir=path.parent)
    published = False
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.link(stage, path)
        published = True
    finally:
        try:
            os.unlink(stage)
        except FileNotFoundError:
            pass
        except OSError as error:
            logging.getLogger(__name__).warning(
                "Could not remove staging file %s (%s): %s",
                stage, "output published" if published else "output unpublished", error,
            )
