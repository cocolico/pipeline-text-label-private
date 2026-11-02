"""Read-only inputs; only task_index and its standard metadata may change."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.parquet as pq


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    temp.replace(path)


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def jsonl(path, values):
    Path(path).write_text("".join(json.dumps(v, ensure_ascii=False, allow_nan=False) + "\n" for v in values))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe dataset path: {relative}")
    path = Path(root) / relative
    if path.resolve() != path.absolute():
        raise ValueError(f"Dataset artifact must not contain symlinks: {relative}")
    return path


def path_for(info, episode, *, video_key=None):
    fields = {"episode_index": episode, "episode_chunk": episode // info["chunks_size"]}
    if video_key is not None:
        fields["video_key"] = video_key
    return info["video_path" if video_key is not None else "data_path"].format(**fields)


def inspect(root):
    root = Path(root).resolve(strict=True)
    info = read_json(root / "meta/info.json")
    if info.get("codebase_version") != "v2.1":
        raise ValueError("This adapter supports LeRobot v2.1 only")
    if type(info.get("chunks_size")) is not int or info["chunks_size"] <= 0:
        raise ValueError("Invalid chunks_size")
    if not isinstance(info.get("fps"), (int, float)) or not math.isfinite(info["fps"]) or info["fps"] <= 0:
        raise ValueError("Invalid FPS")
    episodes = rows(root / "meta/episodes.jsonl")
    tasks = rows(root / "meta/tasks.jsonl")
    stats = rows(root / "meta/episodes_stats.jsonl")
    if not episodes or [ep["episode_index"] for ep in episodes] != list(range(len(episodes))):
        raise ValueError("Episodes must be nonempty, contiguous and ordered from zero")
    if [r["episode_index"] for r in stats] != list(range(len(episodes))):
        raise ValueError("Episode statistics must cover exactly the episodes in order")
    if [t["task_index"] for t in tasks] != list(range(len(tasks))) or any(not isinstance(t.get("task"), str) or not t["task"].strip() for t in tasks):
        raise ValueError("Invalid task vocabulary")
    if info["total_episodes"] != len(episodes) or info["total_frames"] != sum(ep["length"] for ep in episodes) or info["total_tasks"] != len(tasks):
        raise ValueError("Dataset metadata counts disagree")
    cameras = [name for name, value in info["features"].items() if value["dtype"] == "video"]
    if not cameras:
        raise ValueError("Dataset has no video features")
    artifacts = []
    total = 0
    for ep in episodes:
        if type(ep["length"]) is not int or ep["length"] <= 0:
            raise ValueError("Empty/invalid episode")
        relative = path_for(info, ep["episode_index"])
        file = artifact(root, relative)
        with pq.ParquetFile(file) as parquet:
            if parquet.metadata.num_rows != ep["length"]:
                raise ValueError(f"Row count differs: {relative}")
            for name in ("frame_index", "episode_index", "index", "task_index"):
                if not pa.types.is_int64(parquet.schema_arrow.field(name).type):
                    raise ValueError(f"Expected scalar int64 column: {name}")
            expected = set(info["features"]) - set(cameras)
            if set(parquet.schema_arrow.names) != expected:
                raise ValueError(f"Feature/schema mismatch: {relative}")
        for name in [relative] + [path_for(info, ep["episode_index"], video_key=c) for c in cameras]:
            file = artifact(root, name)
            if not file.is_file():
                raise ValueError(f"Missing artifact: {name}")
            stat = file.stat()
            artifacts.append({"path": name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
            total += stat.st_size
    if info.get("total_videos") != len(episodes) * len(cameras):
        raise ValueError("Video count differs")
    metadata_hashes = {name: sha256(root / "meta" / name) for name in
                       ("info.json", "episodes.jsonl", "tasks.jsonl", "episodes_stats.jsonl")}
    identity = {"metadata": metadata_hashes, "artifacts": artifacts}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return {"root": root, "info": info, "episodes": episodes, "tasks": tasks,
            "stats": stats, "cameras": cameras, "artifacts": artifacts,
            "bytes": total, "signature": signature, "metadata_hashes": metadata_hashes}


def validate_segments(segments, length):
    if not isinstance(segments, list) or not segments:
        raise ValueError("A complete nonempty step list is required")
    cursor = 0
    for segment in segments:
        start, end = segment.get("start_frame"), segment.get("end_frame")
        if type(start) is not int or type(end) is not int or not (start == cursor < end <= length):
            raise ValueError("Steps must cover every frame exactly once, using half-open intervals")
        if not isinstance(segment.get("text"), str) or not segment["text"].strip():
            raise ValueError("Step text is empty")
        if segment.get("evidence_status") != "observed":
            raise ValueError("Uncertain model annotation cannot be automatically delivered")
        cursor = end
    if cursor != length:
        raise ValueError("Unannotated episode tail")


def select_prefix(source, limit):
    """Read a leading subset directly from source; no input files are copied.

    A prefix preserves episode_index/global index without reindexing sensor rows.
    Output metadata explicitly describes the selected subset, not the full input.
    """
    if limit is None:
        return source
    if type(limit) is not int or not 1 <= limit <= len(source["episodes"]):
        raise ValueError("episode-limit must be between 1 and the source episode count")
    result = dict(source)
    result["original_info"] = source["info"]
    result["episodes"] = source["episodes"][:limit]
    result["stats"] = source["stats"][:limit]
    info = deepcopy(source["info"])
    info.update(total_episodes=limit, total_frames=sum(ep["length"] for ep in result["episodes"]),
                total_videos=limit * len(source["cameras"]),
                total_chunks=(limit + info["chunks_size"] - 1) // info["chunks_size"])
    splits = {}
    for name, span in info["splits"].items():
        a, b = map(int, span.split(":"))
        if min(a, limit) < min(b, limit):
            splits[name] = f"{min(a,limit)}:{min(b,limit)}"
    info["splits"] = splits
    result["info"] = info
    paths = set()
    for ep in result["episodes"]:
        paths.add(path_for(info, ep["episode_index"]))
        paths.update(path_for(info, ep["episode_index"], video_key=c) for c in source["cameras"])
    result["artifacts"] = [row for row in source["artifacts"] if row["path"] in paths]
    result["bytes"] = sum(row["bytes"] for row in result["artifacts"])
    return result


def task_stats(indices):
    n = len(indices)
    mean = sum(indices) / n
    return {"min": [min(indices)], "max": [max(indices)], "mean": [mean],
            "std": [math.sqrt(sum((v - mean) ** 2 for v in indices) / n)], "count": [n]}


def export_dataset(source, destination, labels):
    """Build a NEW directory. Existing datasets are never overwritten."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "meta").mkdir()
    info = deepcopy(source["info"])
    tasks, vocabulary, episodes_out, stats_out = [], {}, [], []
    for episode, stats in zip(source["episodes"], source["stats"]):
        ep = episode["episode_index"]
        segments = labels[ep]
        validate_segments(segments, episode["length"])
        indices, names = [], []
        for segment in segments:
            text = segment["text"].strip()
            if text not in vocabulary:
                vocabulary[text] = len(tasks)
                tasks.append({"task_index": vocabulary[text], "task": text})
            index = vocabulary[text]
            indices.extend([index] * (segment["end_frame"] - segment["start_frame"]))
            if text not in names:
                names.append(text)
        relative = path_for(info, ep)
        output = artifact(destination, relative)
        output.parent.mkdir(parents=True, exist_ok=True)
        with pq.ParquetFile(artifact(source["root"], relative)) as parquet:
            with pq.ParquetWriter(output, parquet.schema_arrow, compression="zstd") as writer:
                offset = 0
                for batch in parquet.iter_batches(batch_size=8, use_threads=False):
                    table = pa.Table.from_batches([batch])
                    field = table.schema.field("task_index")
                    array = pa.array(indices[offset:offset + len(batch)], type=field.type)
                    table = table.set_column(table.schema.get_field_index("task_index"), field, array)
                    writer.write_table(table)
                    offset += len(batch)
        for camera in source["cameras"]:
            relative = path_for(info, ep, video_key=camera)
            target = artifact(destination, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            # No hardlinks: modifying a delivered video must never modify input.
            shutil.copyfile(artifact(source["root"], relative), target)
        row = deepcopy(episode)
        row["tasks"] = names
        # This source-specific scalar cannot describe multi-task episodes.
        row.pop("task_index", None)
        if len(names) == 1:
            row["task_index"] = vocabulary[names[0]]
        episodes_out.append(row)
        stat = deepcopy(stats)
        stat["stats"]["task_index"] = task_stats(indices)
        stats_out.append(stat)
    info["total_tasks"] = len(tasks)
    jsonl(destination / "meta/tasks.jsonl", tasks)
    jsonl(destination / "meta/episodes.jsonl", episodes_out)
    jsonl(destination / "meta/episodes_stats.jsonl", stats_out)
    atomic_json(destination / "meta/info.json", info)


def validate_output(source, output, labels):
    """Compare ALL non-task values, Arrow schemas, and video bytes."""
    delivered = inspect(output)
    expected_info = deepcopy(source["info"])
    expected_info["total_tasks"] = delivered["info"]["total_tasks"]
    if delivered["info"] != expected_info:
        raise ValueError("Unexpected info.json changes")
    tasks = {row["task_index"]: row["task"] for row in delivered["tasks"]}
    global_index = 0
    for old_ep, new_ep, old_stats, new_stats in zip(source["episodes"], delivered["episodes"], source["stats"], delivered["stats"]):
        ep = old_ep["episode_index"]
        expected_text = [s["text"].strip() for s in labels[ep] for _ in range(s["end_frame"] - s["start_frame"])]
        old_extra = {k: v for k, v in old_ep.items() if k not in {"task_index", "tasks"}}
        new_extra = {k: v for k, v in new_ep.items() if k not in {"task_index", "tasks"}}
        if old_extra != new_extra or new_ep["tasks"] != list(dict.fromkeys(expected_text)):
            raise ValueError("Episode metadata mismatch")
        relative = path_for(source["info"], ep)
        with pq.ParquetFile(artifact(source["root"], relative)) as before, pq.ParquetFile(artifact(output, relative)) as after:
            if not before.schema_arrow.equals(after.schema_arrow, check_metadata=True):
                raise ValueError("Arrow schema changed")
            old_batches = before.iter_batches(batch_size=8, use_threads=False)
            new_batches = after.iter_batches(batch_size=8, use_threads=False)
            offset, actual_indices = 0, []
            for a, b in zip(old_batches, new_batches):
                if len(a) != len(b):
                    raise ValueError("Batch length mismatch")
                for name in a.schema.names:
                    if name != "task_index" and not a.column(name).equals(b.column(name)):
                        raise ValueError(f"Non-task data changed: episode={ep} field={name}")
                if b.column("frame_index").to_pylist() != list(range(offset, offset + len(b))):
                    raise ValueError("Non-contiguous frame_index")
                if b.column("episode_index").to_pylist() != [ep] * len(b):
                    raise ValueError("Wrong episode_index")
                if b.column("index").to_pylist() != list(range(global_index + offset, global_index + offset + len(b))):
                    raise ValueError("Non-contiguous dataset index")
                values = b.column("task_index").to_pylist()
                if [tasks.get(index) for index in values] != expected_text[offset:offset + len(b)]:
                    raise ValueError("Task labels do not match frames")
                actual_indices.extend(values)
                offset += len(b)
            if offset != old_ep["length"]:
                raise ValueError("Missing rows")
        expected_stats = deepcopy(old_stats)
        expected_stats["stats"]["task_index"] = task_stats(actual_indices)
        if expected_stats != new_stats:
            raise ValueError("Unexpected statistics changes")
        for camera in source["cameras"]:
            relative = path_for(source["info"], ep, video_key=camera)
            if sha256(artifact(source["root"], relative)) != sha256(artifact(output, relative)):
                raise ValueError("Video bytes changed")
        global_index += old_ep["length"]
    hashes = {str(path.relative_to(output)): sha256(path) for path in Path(output).rglob("*") if path.is_file()}
    return {"status": "passed", "episodes": len(source["episodes"]), "frames": global_index,
            "all_non_task_values_equal": True, "video_checksums_equal": True,
            "arrow_schemas_equal": True, "model_semantic_correctness_certified": False,
            "output_sha256": hashes}
