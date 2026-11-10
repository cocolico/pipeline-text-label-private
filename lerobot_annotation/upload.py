"""Append-only transfer to explicitly configured mounted receiver directories.

No SSH commands, remote deletions, overwrites, or renames. Receiver must ignore
datasets until meta/info.json exists; it is uploaded LAST after verification.
"""
from __future__ import annotations

from pathlib import Path
import shutil

from .dataset import atomic_json, inspect, read_json, sha256
from .pipeline import job_lock


def upload(dataset, destination, state_dir):
    source = Path(dataset).resolve(strict=True)
    inspect(source)
    destination = Path(destination).absolute()
    state = Path(state_dir).resolve()
    if destination.is_symlink() or any(p.is_symlink() for p in destination.parents):
        raise ValueError("Upload destination must not contain symlinks")
    target = destination.resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Upload source and destination must be disjoint")
    if state == source or source in state.parents or state == target or target in state.parents:
        raise ValueError("Upload state must be outside source and destination")
    if not destination.parent.is_dir():
        raise ValueError("Receiver parent must already exist; no guessed remote path creation")
    files = sorted(p for p in source.rglob("*") if p.is_file())
    if any(p.is_symlink() for p in source.rglob("*")):
        raise ValueError("Upload source contains symlinks")
    expected = {str(p.relative_to(source)): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files}
    config = {"source": str(source), "destination": str(target), "files": expected}
    with job_lock(state):
        journal = state / "transfer.json"
        if journal.exists():
            old = read_json(journal)
            if old["config"] != config:
                raise ValueError("Upload config or local dataset changed")
        else:
            if destination.exists():
                raise FileExistsError("Refuse an existing receiver dataset without this job's journal")
            # This record authorizes resuming only this exact new destination.
            atomic_json(journal, {"config": config, "status": "initializing"})
        destination.mkdir(exist_ok=True)
        # The standard info file is the completion gate, not an extra sidecar.
        files.sort(key=lambda p: p.relative_to(source).as_posix() == "meta/info.json")
        try:
            for file in files:
                rel = file.relative_to(source)
                target_file = destination / rel
                if target_file.is_symlink() or any(p.is_symlink() for p in target_file.parents):
                    raise ValueError("Receiver symlink detected")
                target_file.parent.mkdir(parents=True, exist_ok=True)
                proof = expected[str(rel)]
                if target_file.exists():
                    if not target_file.is_file() or target_file.stat().st_size != proof["bytes"] or sha256(target_file) != proof["sha256"]:
                        raise ValueError(f"Receiver conflict; never overwrite: {rel}")
                    continue
                # Exclusive creation. A truncated file after interruption remains
                # a conflict; choose a new destination rather than overwriting it.
                with file.open("rb") as src, target_file.open("xb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                if sha256(target_file) != proof["sha256"]:
                    raise ValueError(f"Upload checksum mismatch: {rel}")
            atomic_json(journal, {"config": config, "status": "completed"})
        except BaseException as exc:
            atomic_json(journal, {"config": config, "status": "failed", "error": str(exc)})
            raise
    return destination
