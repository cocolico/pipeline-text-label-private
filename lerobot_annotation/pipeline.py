"""Conversion-completion hook: immutable input -> complete annotated dataset."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import ctypes
import json
import os
from pathlib import Path
import shutil

from .dataset import (artifact, atomic_json, export_dataset, inspect, path_for,
                      read_json, sha256, validate_output, validate_segments)
from .dataset import select_prefix
from .model import ArkAnnotator, PROMPT, windows
import hashlib


@contextmanager
def job_lock(work):
    work.mkdir(parents=True, exist_ok=True)
    with (work / "job.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def preflight(input_root, output_root, work_root):
    source = inspect(input_root)
    if Path(output_root).is_symlink():
        raise ValueError("Output must not be a symlink")
    output, work = Path(output_root).resolve(), Path(work_root).resolve()
    root = source["root"]
    paths = [root, output, work]
    if any(a == b or a in b.parents or b in a.parents for i, a in enumerate(paths) for b in paths[i+1:]):
        raise ValueError("Input, output and work must be disjoint directories")
    if output.is_symlink():
        raise ValueError("Output must not be a symlink")
    return source


def publish_new_directory(source, destination):
    """Linux atomic RENAME_NOREPLACE: even an empty existing output is protected."""
    library = ctypes.CDLL(None, use_errno=True)
    function = getattr(library, "renameat2", None)
    if function is None:
        raise RuntimeError("Atomic no-replace publication requires Linux renameat2")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def require_space(source, output):
    ancestor = Path(output).parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    # Parquet recompression can change size; this is a conservative estimate.
    required = int(source["bytes"] * 1.25) + 256 * 1024 * 1024
    free = shutil.disk_usage(ancestor).free
    if free < required:
        raise OSError(f"Insufficient disk space: need approximately {required} bytes, free {free}")


def run(input_root, output_root, work_root, *, model, video_fps=5.0,
        window_seconds=30.0, attempts=2, max_calls=300, annotator=None,
        episode_limit=None):
    work = Path(work_root).resolve()
    source = preflight(input_root, output_root, work)
    source = select_prefix(source, episode_limit)
    output = Path(output_root).resolve()
    with job_lock(work):
        config = {"source_signature": source["signature"], "source_root": str(source["root"]),
                  "output_root": str(output), "model": model, "camera": "observation.images.cam0",
                  "video_fps": video_fps, "window_seconds": window_seconds,
                  "episode_limit": episode_limit,
                  "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                  "api_base": os.environ.get("API_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")}
        if (work / "job_config.json").exists() and read_json(work / "job_config.json") != config:
            raise ValueError("Input or configuration changed; use a new work directory")
        atomic_json(work / "job_config.json", config)
        if output.exists():
            proof_path = work / "completed.json"
            if not proof_path.exists():
                proof_path = work / "ready_to_publish.json"
            if not proof_path.exists():
                raise FileExistsError("Refuse existing output without this job's completion proof")
            proof = read_json(proof_path)
            if proof.get("config") != config:
                raise ValueError("Completion proof differs")
            labels = {int(k): v for k, v in read_json(work / "all_labels.json").items()}
            report = validate_output(source, output, labels)
            if report != proof["validation"]:
                raise ValueError("Completed output differs from publication proof")
            atomic_json(work / "completed.json", proof)
            atomic_json(work / "status.json", {"status": "completed", "output_root": str(output)})
            return output
        require_space(source, output)
        if config["camera"] not in source["cameras"]:
            raise ValueError("cam0 primary video is required")
        engine = annotator or ArkAnnotator(model=model, video_fps=video_fps,
            attempts=attempts, max_calls=max_calls, api_base=config["api_base"])
        if annotator is None:
            engine.calls = sum(len(read_json(p)) for p in work.glob("episode_*/frames_*/attempts.json"))
        summary = {"status": "annotating", "completed_episodes": 0,
                   "total_episodes": len(source["episodes"])}
        atomic_json(work / "status.json", summary)
        # Original source task/provenance stays in work, never mixed into delivery.
        atomic_json(work / "source_metadata.json", {"info": source["info"], "tasks": source["tasks"],
                    "episodes": source["episodes"], "metadata_hashes": source["metadata_hashes"],
                    "original_info": source.get("original_info", source["info"])})
        labels = {}
        job_id = hashlib.sha256(str(work).encode()).hexdigest()[:12]
        building = output.with_name("." + output.name + ".annotation-building-" + job_id)
        try:
            if hasattr(engine, "verify_video"):
                summary["status"] = "validating_source_videos"
                atomic_json(work / "status.json", summary)
                for episode in source["episodes"]:
                    for camera in source["cameras"]:
                        engine.verify_video(artifact(source["root"], path_for(source["info"], episode["episode_index"], video_key=camera)),
                            length=episode["length"], fps=source["info"]["fps"],
                            shape=source["info"]["features"][camera]["shape"])
                summary["status"] = "annotating"
                atomic_json(work / "status.json", summary)
            for episode in source["episodes"]:
                ep = episode["episode_index"]
                video = artifact(source["root"], path_for(source["info"], ep, video_key=config["camera"]))
                source_hash = sha256(video)
                steps = []
                for start, end in windows(episode["length"], source["info"]["fps"], window_seconds):
                    steps.extend(engine.annotate(video, start=start, end=end, fps=source["info"]["fps"],
                        task=episode["tasks"], source_hash=source_hash,
                        work=work / f"episode_{ep:06d}" / f"frames_{start}_{end}"))
                validate_segments(steps, episode["length"])
                labels[ep] = steps
                summary["completed_episodes"] += 1
                atomic_json(work / "status.json", summary)
            atomic_json(work / "all_labels.json", labels)
            if inspect(source["root"])["signature"] != source["signature"]:
                raise ValueError("Source dataset changed during annotation")
            summary["status"] = "exporting"
            atomic_json(work / "status.json", summary)
            # A previous interrupted export is rebuildable; never delete input/output.
            if building.exists():
                owner = read_json(work / "building_owner.json")
                if building.is_symlink() or owner != {"path": str(building), "config": config}:
                    raise ValueError("Refuse unowned staging directory")
                shutil.rmtree(building)
            atomic_json(work / "building_owner.json", {"path": str(building), "config": config})
            export_dataset(source, building, labels)
            report = validate_output(source, building, labels)
            if inspect(source["root"])["signature"] != source["signature"]:
                raise ValueError("Source dataset changed during export")
            atomic_json(work / "validation.json", report)
            proof = {"config": config, "validation": report}
            # Allows automatic recovery after publish succeeds but the process
            # stops before completed.json is written.
            atomic_json(work / "ready_to_publish.json", proof)
            # Publish only after validation. Local lock serializes this job;
            # output must still be absent and no existing dataset is replaced.
            if output.exists() or output.is_symlink():
                raise FileExistsError(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            publish_new_directory(building, output)
            atomic_json(work / "completed.json", proof)
            atomic_json(work / "status.json", {**summary, "status": "completed", "output_root": str(output)})
            return output
        except BaseException as exc:
            atomic_json(work / "status.json", {**summary, "status": "failed",
                        "error_type": type(exc).__name__, "error": str(exc)})
            raise
