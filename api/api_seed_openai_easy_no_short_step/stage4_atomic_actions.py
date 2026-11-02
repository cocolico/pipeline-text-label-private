#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stage 4: Decompose each step into fine-grained atomic actions with temporal boundaries and cut atomic clips."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from common import (
    add_api_cli_args,
    add_sampling_cli_args,
    VIDEO_EXTS,
    api_config_from_args,
    build_video_api_content,
    build_retry_prefix,
    build_source_video_meta,
    can_open_video,
    call_chat_completion,
    collect_videos,
    cut_video_segment_ffmpeg,
    default_output_root,
    ensure_frame_index_fps,
    ensure_video_out_dir_safe,
    extract_json_from_response,
    format_duration,
    get_video_metadata,
    initialize_api_client,
    guard_schema_fingerprint,
    logger,
    now_utc_iso,
    read_json,
    sample_video_to_frame_index,
    sampling_config_from_args,
    sanitize_filename,
    stage4_atomic_clip_relpath,
    update_run_summary,
    video_id_from_path,
    write_frame_manifest,
    write_json,
    write_text,
)
from prompts import (
    SYSTEM_PROMPT_ATOMIC_ACTION_DECOMPOSER,
    SYSTEM_PROMPT_OBJECT_INVENTORY,
    build_stage4_object_inventory_user_prompt,
    build_stage4_user_prompt,
)

if TYPE_CHECKING:
    from common import ApiConfig, SamplingConfig


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

MIN_STANDALONE_ATOMIC_FRAME_SPAN = 5
MIN_STANDALONE_ATOMIC_DURATION_SEC = 1.0
TRANSITIONAL_MAIN_ACTION_RE = re.compile(
    r"^\s*(?:release|withdraw|pause|hold\s+steady|stabili[sz]e|"
    r"minor(?:ly)?\s+adjust|adjust\s+slightly|move\s+(?:the\s+)?(?:hand|hands|finger|fingers)\s+away|"
    r"remove\s+(?:the\s+)?(?:hand|hands|finger|fingers)|let\s+go)\b",
    re.IGNORECASE,
)


def validate_stage4_atomic_actions(
    obj: Any,
    step_id: int,
    num_frames: int,
    *,
    local_timestamps: Optional[List[float]] = None,
    clip_duration: Optional[float] = None,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Validate and normalize a Stage 4 atomic actions response.

    Returns ``(normalized_obj, errors)`` where *normalized_obj* is ``None``
    when validation fails (errors will be non-empty).
    """
    errors: List[str] = []

    if not isinstance(obj, dict):
        return None, ["Response is not a JSON object."]

    # -- step_id -----------------------------------------------------------
    raw_sid = obj.get("step_id")
    if raw_sid is None:
        errors.append("Missing 'step_id'.")
    else:
        try:
            got_sid = int(raw_sid)
        except (ValueError, TypeError):
            errors.append(f"'step_id' is not an integer: {raw_sid!r}")
            got_sid = None
        if got_sid is not None and got_sid != int(step_id):
            errors.append(f"'step_id' mismatch: expected {int(step_id)}, got {got_sid}.")

    # -- atomic_actions list -----------------------------------------------
    actions = obj.get("atomic_actions")
    if not isinstance(actions, list) or len(actions) == 0:
        errors.append("'atomic_actions' must be a non-empty list.")
        return None, errors

    REQUIRED_KEYS = {
        "atomic_action_id", "start_frame_index", "end_frame_index",
        "actor", "action", "patient", "caption",
    }

    # Frame/time reference pattern (must NOT appear in free-form text fields)
    _FRAME_REF_RE = re.compile(
        r"(?:frame[_\s]*(?:index|#|num(?:ber)?)?[_\s]*[:=]?\s*\d)|"
        r"(?:\b(?:at|from|to|around|between)\s+\d+(?:\.\d+)?\s*(?:s|sec|seconds?|ms)\b)|"
        r"(?:timestamp\s*[:=]?\s*\d)",
        re.IGNORECASE,
    )

    normalized_actions: List[Dict[str, Any]] = []
    prev_end: Optional[int] = None
    enforce_min_span = len(actions) > 1 and int(num_frames) > MIN_STANDALONE_ATOMIC_FRAME_SPAN + 1
    timing_available = (
        isinstance(local_timestamps, list)
        and len(local_timestamps) > 0
        and clip_duration is not None
    )

    for i, act in enumerate(actions):
        prefix = f"atomic_actions[{i}]"
        if not isinstance(act, dict):
            errors.append(f"{prefix}: not a JSON object.")
            continue

        # Required keys
        missing = REQUIRED_KEYS - set(act.keys())
        if missing:
            errors.append(f"{prefix}: missing keys {sorted(missing)}.")
            continue

        # atomic_action_id
        try:
            aid = int(act["atomic_action_id"])
        except (ValueError, TypeError):
            errors.append(f"{prefix}: 'atomic_action_id' is not an integer.")
            continue
        if aid != i + 1:
            errors.append(f"{prefix}: 'atomic_action_id' must be {i + 1} (sequential), got {aid}.")

        # Frame indices
        try:
            s = int(act["start_frame_index"])
            e = int(act["end_frame_index"])
        except (ValueError, TypeError):
            errors.append(f"{prefix}: start/end frame_index must be integers.")
            continue

        if not (1 <= s <= num_frames):
            errors.append(f"{prefix}: start_frame_index={s} out of range [1, {num_frames}].")
        if not (2 <= e <= num_frames + 1):
            errors.append(f"{prefix}: end_frame_index={e} out of range [2, {num_frames + 1}].")
        if s >= e:
            errors.append(f"{prefix}: start_frame_index ({s}) must be < end_frame_index ({e}).")
        span = e - s
        if enforce_min_span and span < MIN_STANDALONE_ATOMIC_FRAME_SPAN:
            errors.append(
                f"{prefix}: spans only {span} frame-index intervals. "
                f"For no_short_step runs, standalone atomic actions shorter than "
                f"{MIN_STANDALONE_ATOMIC_FRAME_SPAN} intervals (~1s at 5fps) are too fragmented; "
                "merge this brief fragment with the adjacent atomic action unless it is the only action in the step."
            )
        if enforce_min_span and timing_available and 1 <= s <= len(local_timestamps):
            start_t = float(local_timestamps[s - 1])
            if e - 1 < len(local_timestamps):
                end_t = float(local_timestamps[e - 1])
            else:
                end_t = float(clip_duration)
            duration_sec = max(0.0, end_t - start_t)
            if duration_sec < MIN_STANDALONE_ATOMIC_DURATION_SEC:
                errors.append(
                    f"{prefix}: duration is only {duration_sec:.3f}s. "
                    f"For no_short_step runs, standalone atomic actions shorter than "
                    f"{MIN_STANDALONE_ATOMIC_DURATION_SEC:.1f}s are too fragmented; "
                    "merge this brief fragment with the adjacent atomic action unless it is the only action in the step."
                )

        # Contiguity
        if i == 0 and s != 1:
            errors.append(f"{prefix}: first atomic action must start at 1, got {s}.")
        if prev_end is not None and s != prev_end:
            errors.append(f"{prefix}: contiguity violation: expected start={prev_end}, got {s}.")
        prev_end = e

        # Text fields non-empty
        for key in ("actor", "action", "patient", "caption"):
            val = act.get(key)
            if not isinstance(val, str) or not val.strip():
                errors.append(f"{prefix}: '{key}' must be a non-empty string.")

        # Frame/time reference ban in free-form text
        for key in ("action", "caption", "patient"):
            val = str(act.get(key, "")).strip()
            if val and _FRAME_REF_RE.search(val):
                errors.append(f"{prefix}: '{key}' must not reference frame indices or timestamps.")
        raw_action_text = str(act.get("action", "")).strip()
        if enforce_min_span and TRANSITIONAL_MAIN_ACTION_RE.search(raw_action_text):
            errors.append(
                f"{prefix}: action {raw_action_text!r} is a standalone transitional fragment. "
                "Merge release/withdraw/hold-steady/pause/minor-adjust/stabilization fragments with the adjacent task action."
            )

        # patient snake_case check (allow colon separator for multi-entity)
        raw_patient = str(act.get("patient", "")).strip()
        if raw_patient and not re.fullmatch(r"[a-z][a-z0-9_]*(?::[a-z][a-z0-9_]*)*", raw_patient):
            errors.append(
                f"{prefix}: 'patient' should be snake_case (got {raw_patient!r}). "
                "Use underscores between words, colons between entities."
            )

        # Build normalized action
        normalized_actions.append({
            "atomic_action_id": i + 1,
            "start_frame_index": s,
            "end_frame_index": e,
            "actor": str(act.get("actor", "")).strip(),
            "action": str(act.get("action", "")).strip(),
            "patient": str(act.get("patient", "")).strip(),
            "caption": str(act.get("caption", "")).strip(),
        })

    # Last action must end at num_frames + 1
    if normalized_actions:
        last_end = normalized_actions[-1]["end_frame_index"]
        if last_end != num_frames + 1:
            errors.append(
                f"Last atomic action end_frame_index must be {num_frames + 1} (full coverage), got {last_end}."
            )

    if errors:
        return None, errors

    normalized = {
        "step_id": int(step_id),
        "atomic_actions": normalized_actions,
    }
    return normalized, []


def _public_atomic_action_entry(action: Dict[str, Any]) -> Dict[str, Any]:
    """Return the dataset-facing atomic action schema.

    The model/debug schema uses dense step-clip boundary indices internally.
    Public outputs expose only original-video 0-based frame numbers plus times.
    """
    ordered_keys = [
        "atomic_action_id",
        "actor",
        "action",
        "patient",
        "entity_ref",
        "caption",
        "start_sec",
        "end_sec",
        "start_frame_0based",
        "end_frame_0based_exclusive",
        "end_frame_0based_inclusive",
        "clip_relpath",
    ]
    return {key: action[key] for key in ordered_keys if key in action}


# ---------------------------------------------------------------------------
# Object inventory (recording-level entity_ref resolution)
# ---------------------------------------------------------------------------

_ALLOWED_IDENTITY_SCOPES = {"instance", "group", "unresolved"}


def _split_patient_mentions(patient: str) -> List[str]:
    """Split a (usually single-entity) patient string on ':' for the rare
    validator-tolerated multi-entity form. Returns non-empty trimmed parts."""
    raw = str(patient or "").strip()
    if not raw:
        return []
    return [p.strip() for p in raw.split(":") if p.strip()]


def _collect_object_mentions(final_steps: List[Dict[str, Any]]) -> List[str]:
    """Collect every distinct object-mention string across the whole recording.

    Sources: hand_object_relations[].object and atomic_actions[].patient
    (colon-multi-entity patients are split into their parts). Order is
    chronological (step order, then relation/action order); de-duplicated
    by exact string.
    """
    seen: Dict[str, None] = {}
    for step in final_steps:
        for rel in step.get("hand_object_relations", []) or []:
            if not isinstance(rel, dict):
                continue
            obj_name = str(rel.get("object", "")).strip()
            if obj_name and obj_name not in seen:
                seen[obj_name] = None
        for act in step.get("atomic_actions", []) or []:
            if not isinstance(act, dict):
                continue
            for part in _split_patient_mentions(act.get("patient", "")):
                if part not in seen:
                    seen[part] = None
    return list(seen.keys())


def _build_object_inventory_context(high_level_goal: str, final_steps: List[Dict[str, Any]]) -> str:
    """Build a compact, read-only text context of the whole recording's plan
    (step goals, hand_object_relations, atomic actions) for the inventory prompt."""
    lines: List[str] = []
    for step in final_steps:
        sid = step.get("step_id")
        lines.append(f"Step {sid}: {step.get('step_goal', '')}")
        rels = step.get("hand_object_relations", []) or []
        if rels:
            lines.append("  hand_object_relations:")
            for rel in rels:
                if not isinstance(rel, dict):
                    continue
                lines.append(
                    f"    - object={rel.get('object', '')!r} hand={rel.get('hand', '')!r} "
                    f"relation={rel.get('relation', '')!r}"
                )
        acts = step.get("atomic_actions", []) or []
        if acts:
            lines.append("  atomic_actions:")
            for act in acts:
                if not isinstance(act, dict):
                    continue
                lines.append(
                    f"    - actor={act.get('actor', '')!r} action={act.get('action', '')!r} "
                    f"patient={act.get('patient', '')!r} caption={act.get('caption', '')!r}"
                )
    return "\n".join(lines)


def validate_object_inventory(
    obj: Any,
    mentions: List[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, str]], List[str]]:
    """Validate an object-inventory VLM response.

    Returns ``(inventory_list, mention_to_ref, errors)``. On failure,
    the first two are ``None`` and *errors* is non-empty.
    """
    errors: List[str] = []
    if not isinstance(obj, dict):
        return None, None, ["Response is not a JSON object."]

    inventory = obj.get("object_inventory")
    if not isinstance(inventory, list) or len(inventory) == 0:
        errors.append("'object_inventory' must be a non-empty list.")
        return None, None, errors

    seen_refs: Dict[str, None] = {}
    normalized_inventory: List[Dict[str, Any]] = []
    for i, entry in enumerate(inventory):
        prefix = f"object_inventory[{i}]"
        if not isinstance(entry, dict):
            errors.append(f"{prefix}: not a JSON object.")
            continue
        entity_ref = str(entry.get("entity_ref", "")).strip()
        canonical_name = str(entry.get("canonical_name", "")).strip()
        identity_scope = str(entry.get("identity_scope", "")).strip()
        if not entity_ref:
            errors.append(f"{prefix}: 'entity_ref' must be a non-empty string.")
            continue
        if entity_ref in seen_refs:
            errors.append(f"{prefix}: duplicate entity_ref {entity_ref!r}.")
            continue
        if not canonical_name:
            errors.append(f"{prefix}: 'canonical_name' must be a non-empty string.")
        if identity_scope not in _ALLOWED_IDENTITY_SCOPES:
            errors.append(
                f"{prefix}: 'identity_scope' must be one of {sorted(_ALLOWED_IDENTITY_SCOPES)}, got {identity_scope!r}."
            )
        seen_refs[entity_ref] = None
        normalized_inventory.append(
            {"entity_ref": entity_ref, "canonical_name": canonical_name, "identity_scope": identity_scope}
        )

    if errors:
        return None, None, errors

    mentions_map = obj.get("mentions")
    if not isinstance(mentions_map, dict):
        errors.append("'mentions' must be a JSON object mapping mention strings to entity_ref.")
        return None, None, errors

    mention_to_ref: Dict[str, str] = {}
    valid_refs = set(seen_refs.keys())
    missing_mentions: List[str] = []
    for mention in mentions:
        if mention not in mentions_map:
            missing_mentions.append(mention)
            continue
        ref = str(mentions_map.get(mention, "")).strip()
        if ref not in valid_refs:
            errors.append(f"mentions[{mention!r}]: entity_ref {ref!r} not found in 'object_inventory'.")
            continue
        mention_to_ref[mention] = ref

    if missing_mentions:
        errors.append(
            f"'mentions' is missing {len(missing_mentions)} required mention(s): {missing_mentions[:10]!r}"
            + (" (truncated)" if len(missing_mentions) > 10 else "")
        )

    if errors:
        return None, None, errors

    return normalized_inventory, mention_to_ref, []


def _apply_entity_refs(final_steps: List[Dict[str, Any]], mention_to_ref: Dict[str, str]) -> None:
    """Mutate final_steps in place, injecting stable entity_ref alongside every
    object-referring field. No text is rewritten; only the sibling key is added."""
    for step in final_steps:
        for rel in step.get("hand_object_relations", []) or []:
            if not isinstance(rel, dict):
                continue
            obj_name = str(rel.get("object", "")).strip()
            ref = mention_to_ref.get(obj_name)
            if ref:
                rel["entity_ref"] = ref
            else:
                rel.pop("entity_ref", None)
        for act in step.get("atomic_actions", []) or []:
            if not isinstance(act, dict):
                continue
            parts = _split_patient_mentions(act.get("patient", ""))
            refs = [mention_to_ref[p] for p in parts if p in mention_to_ref]
            if refs and len(refs) == len(parts):
                act["entity_ref"] = ":".join(refs)
            else:
                act.pop("entity_ref", None)


def _validate_applied_entity_refs(
    final_steps: List[Dict[str, Any]],
    mention_to_ref: Dict[str, str],
    object_inventory: List[Dict[str, Any]],
) -> List[str]:
    """Verify that every object mention in every step resolves through the
    recording-level inventory. This prevents stale per-step refs from silently
    surviving resume or partial reruns."""
    errors: List[str] = []
    valid_refs = {str(e.get("entity_ref", "")).strip() for e in object_inventory if isinstance(e, dict)}
    for step in final_steps:
        sid = step.get("step_id", "?")
        for i, rel in enumerate(step.get("hand_object_relations", []) or []):
            if not isinstance(rel, dict):
                continue
            mention = str(rel.get("object", "")).strip()
            expected = mention_to_ref.get(mention)
            actual = str(rel.get("entity_ref", "")).strip()
            if not expected or actual != expected or actual not in valid_refs:
                errors.append(f"Step {sid} relation[{i}] object={mention!r}: expected {expected!r}, got {actual!r}")
        for i, act in enumerate(step.get("atomic_actions", []) or []):
            if not isinstance(act, dict):
                continue
            parts = _split_patient_mentions(act.get("patient", ""))
            expected_refs = [mention_to_ref.get(p) for p in parts]
            expected = ":".join(expected_refs) if parts and all(expected_refs) else ""
            actual = str(act.get("entity_ref", "")).strip()
            if not expected or actual != expected or any(ref not in valid_refs for ref in actual.split(":")):
                errors.append(f"Step {sid} atomic_action[{i}] patient={act.get('patient', '')!r}: expected {expected!r}, got {actual!r}")
    return errors


def _sync_entity_refs_to_step_caches(
    final_steps: List[Dict[str, Any]], step_output_paths: Dict[int, str]
) -> None:
    """Make per-step atomic_actions.json files agree with the merged plan."""
    for step in final_steps:
        sid = int(step["step_id"])
        path = step_output_paths.get(sid)
        if not path or not os.path.exists(path):
            continue
        data = read_json(path)
        actions = data.get("atomic_actions") if isinstance(data, dict) else None
        final_actions = step.get("atomic_actions", []) or []
        if not isinstance(actions, list) or len(actions) != len(final_actions):
            raise RuntimeError(f"Cannot synchronize entity_ref for Step {sid}: atomic action cache differs")
        for cached, resolved in zip(actions, final_actions):
            if not isinstance(cached, dict) or not isinstance(resolved, dict):
                raise RuntimeError(f"Cannot synchronize entity_ref for Step {sid}: invalid atomic action")
            ref = str(resolved.get("entity_ref", "")).strip()
            if not ref:
                raise RuntimeError(f"Cannot synchronize entity_ref for Step {sid}: unresolved patient")
            cached["entity_ref"] = ref
        write_json(path, data)


def _can_resume_object_inventory(
    inventory_path: str,
    mentions: List[str],
) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, str]]]:
    """Return cached (inventory, mention_to_ref) if the cached file is valid and
    still covers every current mention string, else None."""
    if not os.path.exists(inventory_path):
        return None
    try:
        data = read_json(inventory_path)
    except Exception:
        return None
    inventory, mention_to_ref, errs = validate_object_inventory(data, mentions)
    if inventory is None or mention_to_ref is None:
        return None
    return inventory, mention_to_ref


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _can_resume_stage4_step(
    step_out_path: str,
    step_id: int,
    num_frames: int,
    atomic_clips_dir: str,
) -> Optional[Dict[str, Any]]:
    """Return cached atomic_actions.json if valid and all clips exist, else None."""
    if not os.path.exists(step_out_path):
        return None
    try:
        data = read_json(step_out_path)
    except Exception:
        return None

    normalized, errs = validate_stage4_atomic_actions(data, step_id, num_frames)
    if normalized is None:
        return None

    # Ensure all atomic clips exist
    step_dir_name = os.path.basename(os.path.dirname(atomic_clips_dir))
    stage4_dir = os.path.dirname(os.path.dirname(atomic_clips_dir))
    for act in normalized.get("atomic_actions", []):
        aid = int(act["atomic_action_id"])
        clip_relpath = stage4_atomic_clip_relpath(step_dir_name, aid, str(act.get("action", "")))
        clip_path = os.path.join(stage4_dir, clip_relpath)
        if not os.path.exists(clip_path) or os.path.getsize(clip_path) <= 0:
            return None

    return normalized


def _can_resume_stage4_final(
    final_path: str,
    segments_path: str,
    stage3_plan_path: str,
    inventory_path: str,
) -> bool:
    """Return True if the merged final plan exists and is structurally valid."""
    if not os.path.exists(final_path):
        return False
    try:
        data = read_json(final_path)
    except Exception:
        return False

    if not isinstance(data, dict):
        return False

    steps = data.get("steps")
    if not isinstance(steps, list):
        return False
    if not steps:
        # Upstream screening may legitimately produce no valid hand steps.
        try:
            plan = read_json(stage3_plan_path)
            segments = read_json(segments_path)
            return (
                plan.get("steps") == []
                and segments.get("segments") == []
                and data.get("high_level_goal") == plan.get("high_level_goal", "")
                and data.get("object_inventory") == []
                and read_json(inventory_path) == {"object_inventory": [], "mentions": {}}
            )
        except Exception:
            return False

    extra_top = sorted(set(data.keys()) - {"source_video", "high_level_goal", "object_inventory", "steps"})
    if extra_top:
        return False

    # The recording-level object inventory must exist and be structurally valid
    # whenever the cached final plan references entity_ref at all.
    inventory = data.get("object_inventory")
    if inventory is not None:
        if not isinstance(inventory, list) or not inventory:
            return False
        if not os.path.exists(inventory_path):
            return False
        seen_refs: set = set()
        for entry in inventory:
            if not isinstance(entry, dict):
                return False
            if set(entry.keys()) != {"entity_ref", "canonical_name", "identity_scope"}:
                return False
            ref = entry.get("entity_ref")
            if not isinstance(ref, str) or not ref.strip() or ref in seen_refs:
                return False
            seen_refs.add(ref)
            if entry.get("identity_scope") not in _ALLOWED_IDENTITY_SCOPES:
                return False

    # Quick structural check on each step
    for st in steps:
        if not isinstance(st, dict):
            return False
        extra = sorted(set(st.keys()) - {
            "step_id",
            "step_goal",
            "hand_object_relations",
            "start_sec",
            "end_sec",
            "start_frame_0based",
            "end_frame_0based_exclusive",
            "end_frame_0based_inclusive",
            "clip_relpath",
            "atomic_actions",
        })
        if extra:
            return False
        if st.get("step_id") is None or not st.get("atomic_actions"):
            return False
        relations = st.get("hand_object_relations")
        if not isinstance(relations, list):
            return False
        for relation in relations:
            if not isinstance(relation, dict):
                return False
            if set(relation.keys()) - {"object", "hand", "relation", "entity_ref"}:
                return False
            if not {"object", "hand", "relation"}.issubset(relation.keys()):
                return False
            if not all(
                isinstance(relation.get(key), str) and relation.get(key, "").strip()
                for key in ("object", "hand", "relation")
            ):
                return False
        acts = st["atomic_actions"]
        if not isinstance(acts, list) or not acts:
            return False
        for act in acts:
            if not isinstance(act, dict):
                return False
            if "start_frame_index" in act or "end_frame_index" in act:
                return False
            required_public_frame_keys = {
                "start_frame_0based",
                "end_frame_0based_exclusive",
                "end_frame_0based_inclusive",
            }
            if not required_public_frame_keys.issubset(act.keys()):
                return False

    # Ensure stage3 and stage2 inputs still exist (not stale)
    if not os.path.exists(segments_path):
        return False
    if not os.path.exists(stage3_plan_path):
        return False

    # Check that step IDs and cached Stage-2 clip paths align with current inputs.
    try:
        plan = read_json(stage3_plan_path)
        segments = read_json(segments_path)
        if data.get("high_level_goal") != plan.get("high_level_goal", ""):
            return False
        segment_clip_by_id: Dict[int, str] = {}
        for seg in segments.get("segments", []):
            if isinstance(seg, dict) and seg.get("step_id") is not None:
                rel = seg.get("clip_relpath")
                if isinstance(rel, str) and rel:
                    segment_clip_by_id[int(seg["step_id"])] = rel
        plan_steps_by_id: Dict[int, Dict[str, Any]] = {}
        for s in plan.get("steps", []):
            if isinstance(s, dict) and s.get("step_id") is not None:
                plan_steps_by_id[int(s["step_id"])] = s
        plan_sids = set(plan_steps_by_id)
        final_sids = set()
        for s in steps:
            if isinstance(s, dict) and s.get("step_id") is not None:
                sid = int(s["step_id"])
                final_sids.add(sid)
                plan_step = plan_steps_by_id.get(sid)
                if not isinstance(plan_step, dict):
                    return False
                if s.get("step_goal") != plan_step.get("step_goal"):
                    return False
                # Stage 3 refs are step-local; Stage 4 resolves recording-level refs.
                # Compare relation semantics here; validate resolved refs against inventory below.
                cached_relations = [
                    {key: value for key, value in rel.items() if key != "entity_ref"}
                    for rel in s.get("hand_object_relations", [])
                ]
                input_relations = [
                    {key: value for key, value in rel.items() if key != "entity_ref"}
                    for rel in plan_step.get("hand_object_relations", [])
                ]
                if cached_relations != input_relations:
                    return False
                expected_clip = segment_clip_by_id.get(sid)
                got_clip = s.get("clip_relpath")
                if expected_clip and isinstance(got_clip, str):
                    stage4_dir = os.path.dirname(final_path)
                    video_out = os.path.dirname(stage4_dir)
                    expected_abs = os.path.normpath(os.path.join(video_out, expected_clip))
                    got_abs = os.path.normpath(os.path.join(stage4_dir, got_clip))
                    if got_abs != expected_abs:
                        return False
        if plan_sids != final_sids:
            return False
    except Exception:
        return False

    # A structurally valid legacy final plan may still contain refs produced by
    # independent step runs. Resume only when every mention resolves to the
    # exact recording-level ref stored in object_inventory.json.
    try:
        mentions = _collect_object_mentions(steps)
        inventory_data = read_json(inventory_path)
        cached_inventory, mention_to_ref, inventory_errors = validate_object_inventory(
            inventory_data, mentions
        )
        if cached_inventory is None or mention_to_ref is None or inventory_errors:
            return False
        if data.get("object_inventory") != cached_inventory:
            return False
        if _validate_applied_entity_refs(steps, mention_to_ref, cached_inventory):
            return False
    except Exception:
        return False

    return True


# ---------------------------------------------------------------------------
# Main Stage 4 logic
# ---------------------------------------------------------------------------

def run_stage4_for_video(
    video_path: str,
    output_root: str,
    api_cfg: "ApiConfig",
    sampling_cfg: "SamplingConfig",
    overwrite: bool,
    max_retries: int,
    *,
    ffmpeg_bin: str = "ffmpeg",
    cut_mode: str = "reencode",
    seek_slop_sec: float = 1.0,
    crf: int = 18,
    preset: str = "veryfast",
    keep_audio: bool = False,
    allow_legacy_resume: bool = False,
) -> str:
    """Run Stage 4 (atomic action decomposition) for a single video.

    Reads stage3 outputs, decomposes each step into atomic actions via VLM,
    cuts atomic clips, and merges into a final plan JSON.

    Returns the video output directory path.
    """
    t_start = time.perf_counter()

    ensure_frame_index_fps(sampling_cfg, getattr(api_cfg, "video_fps", 10.0))

    vid = video_id_from_path(video_path)
    video_out = os.path.join(output_root, vid)
    ensure_video_out_dir_safe(video_out, video_path)

    stage2_dir = os.path.join(video_out, "stage2")
    stage3_dir = os.path.join(video_out, "stage3")
    stage4_dir = os.path.join(video_out, "stage4")
    os.makedirs(stage4_dir, exist_ok=True)
    source_meta = get_video_metadata(video_path)
    source_fps = float(source_meta.get("fps", 0.0) or 0.0)
    source_total_frames = int(source_meta.get("total_frames", 0) or 0)
    source_duration_sec = float(source_meta.get("duration_sec", 0.0) or 0.0)

    # Input paths
    stage3_plan_path = os.path.join(stage3_dir, "step_plan.json")
    segments_path = os.path.join(stage2_dir, "step_segments.json")
    final_path = os.path.join(stage4_dir, "atomic_plan_with_clips.json")
    inventory_path = os.path.join(stage4_dir, "object_inventory.json")
    run_summary_path = os.path.join(video_out, "run_summary.json")

    # Validate inputs exist
    if not os.path.exists(stage3_plan_path):
        raise FileNotFoundError(f"Stage 3 plan not found: {stage3_plan_path}")
    if not os.path.exists(segments_path):
        raise FileNotFoundError(f"Stage 2 segments not found: {segments_path}")

    # Full-plan resume check
    will_resume = not overwrite and _can_resume_stage4_final(
        final_path, segments_path, stage3_plan_path, inventory_path,
    )
    schema_fp = guard_schema_fingerprint(
        run_summary_path,
        video_out,
        stage="Stage 4",
        overwrite=overwrite,
        allow_legacy_resume=allow_legacy_resume,
        will_resume=will_resume,
    )
    if will_resume:
        logger.info(f"[stage4] video_id={vid} resume: {os.path.relpath(video_out, output_root)}")
        # Ensure run_summary reflects completion
        stage4_obj: Dict[str, Any] = {}
        if os.path.exists(run_summary_path):
            try:
                rs = read_json(run_summary_path)
            except Exception:
                rs = {}
            if isinstance(rs, dict) and isinstance(rs.get("stage4"), dict):
                stage4_obj = dict(rs.get("stage4") or {})
        stage4_obj.pop("error", None)
        stage4_obj.pop("failed_at_utc", None)
        stage4_obj.pop("started_at_utc", None)
        stage4_obj["status"] = "completed"
        stage4_obj["final_plan_path"] = os.path.relpath(final_path, video_out)
        stage4_obj["resumed_at_utc"] = now_utc_iso()
        update_run_summary(
            run_summary_path,
            {
                "source_video": build_source_video_meta(video_path),
                "output_root": os.path.abspath(output_root),
                "schema_fingerprint": schema_fp,
                "updated_at_utc": now_utc_iso(),
                "stage4": stage4_obj,
            },
        )
        return video_out

    logger.info(
        f"[stage4] video_id={vid} start: overwrite={bool(overwrite)} "
        f"frame_index_fps={float(getattr(sampling_cfg, 'frame_index_fps', 0.0) or 0.0)} "
        f"src={os.path.abspath(video_path)}"
    )

    # Fail fast before any model call.
    if shutil.which(ffmpeg_bin) is None:
        raise FileNotFoundError(
            f"ffmpeg binary not found: '{ffmpeg_bin}'. Install ffmpeg or pass a valid path via --ffmpeg-bin."
        )

    # Load stage3 plan and stage2 segments
    stage3_plan = read_json(stage3_plan_path)
    high_level_goal = str(stage3_plan.get("high_level_goal", "")).strip()
    plan_steps = stage3_plan.get("steps")
    if not isinstance(plan_steps, list):
        raise RuntimeError("Stage 3 plan must contain a steps list.")

    segments_data = read_json(segments_path)
    segs = segments_data.get("segments")
    if not isinstance(segs, list):
        raise RuntimeError("Stage 2 segments must contain a segments list.")
    if not plan_steps:
        if segs:
            raise RuntimeError("Stage 3 has no steps but Stage 2 has segments.")
        empty_plan = {
            "source_video": build_source_video_meta(video_path),
            "high_level_goal": high_level_goal,
            "object_inventory": [],
            "steps": [],
        }
        write_json(final_path, empty_plan)
        write_json(os.path.join(video_out, "final_plan.json"), empty_plan)
        write_json(inventory_path, {"object_inventory": [], "mentions": {}})
        update_run_summary(run_summary_path, {
            "source_video": build_source_video_meta(video_path),
            "output_root": os.path.abspath(output_root),
            "schema_fingerprint": schema_fp,
            "updated_at_utc": now_utc_iso(),
            "stage4": {
                "status": "completed",
                "reason": "no_valid_upstream_steps",
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_calls": 0},
                "final_plan_path": os.path.relpath(final_path, video_out),
                "merged_final_plan_path": "final_plan.json",
                "object_inventory_path": os.path.relpath(inventory_path, video_out),
            },
        })
        logger.info(f"[stage4] video_id={vid} completed: no valid upstream steps; no model calls")
        return video_out
    if not segs:
        raise RuntimeError("Stage 2 segments missing/empty.")
    seg_by_id: Dict[int, Dict[str, Any]] = {}
    for seg in segs:
        if isinstance(seg, dict) and seg.get("step_id") is not None:
            seg_by_id[int(seg["step_id"])] = seg

    # Sort steps
    ordered_steps = sorted(
        [s for s in plan_steps if isinstance(s, dict)],
        key=lambda x: int(x.get("step_id", 0)),
    )
    total_steps = len(ordered_steps)
    logger.info(f"[stage4] video_id={vid} steps={total_steps}")

    client = initialize_api_client(api_cfg)
    if not client:
        raise SystemExit("Failed to initialize API client.")

    final_steps: List[Dict[str, Any]] = []
    step_output_paths: Dict[int, str] = {}
    stage_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_calls": 0}

    for i_step, step in enumerate(ordered_steps, start=1):
        sid = int(step.get("step_id", 0))
        step_goal = str(step.get("step_goal", "")).strip()
        step_goal_short = " ".join(step_goal.split())

        if sid not in seg_by_id:
            raise RuntimeError(f"Missing Stage 2 segment for step_id={sid}")
        seg = seg_by_id[sid]
        try:
            clip_start_sec = float(seg.get("start_sec", 0.0))
            clip_end_sec = float(seg.get("end_sec", 0.0))
        except Exception:
            clip_start_sec = 0.0
            clip_end_sec = 0.0
        clip_rel = seg.get("clip_relpath")
        if not clip_rel:
            raise RuntimeError(f"Missing clip_relpath for step_id={sid}")
        clip_path = clip_rel if os.path.isabs(clip_rel) else os.path.join(video_out, clip_rel)
        if not os.path.exists(clip_path):
            raise FileNotFoundError(f"Clip not found: {clip_path}")

        # Create per-step output directory
        slug = sanitize_filename(step_goal)
        step_dir_name = f"step{sid:02d}_{slug}"
        step_out_dir = os.path.join(stage4_dir, step_dir_name)
        atomic_clips_dir = os.path.join(step_out_dir, "atomic_clips")
        step_out_path = os.path.join(step_out_dir, "atomic_actions.json")
        step_output_paths[sid] = step_out_path
        manifest_path = os.path.join(step_out_dir, "frame_manifest.json")
        raw_path = os.path.join(step_out_dir, "stage4_raw_response.txt")
        sys_prompt_path = os.path.join(step_out_dir, "stage4_system_prompt.txt")
        user_prompt_path = os.path.join(step_out_dir, "stage4_user_prompt.txt")

        # Step-level resume: check if cached atomic_actions.json and clips exist
        if not overwrite:
            # We need num_frames to validate; sample first to count, or use default 50
            cached_num_frames = int(sampling_cfg.max_frames)
            # If manifest exists, use actual num_frames from it
            if os.path.exists(manifest_path):
                try:
                    cached_manifest = read_json(manifest_path)
                    cached_num_frames = int(cached_manifest.get("num_frames", 0) or 0)
                    if cached_num_frames <= 0:
                        cached_num_frames = len(cached_manifest.get("frames", []) or []) or int(sampling_cfg.max_frames)
                except Exception:
                    cached_num_frames = int(sampling_cfg.max_frames)

            cached = _can_resume_stage4_step(
                step_out_path, sid, cached_num_frames, atomic_clips_dir,
            )
            if cached is not None:
                logger.info(
                    f"[stage4] video_id={vid} step {i_step}/{total_steps} step_id={sid}: "
                    f"reuse cached atomic_actions goal='{step_goal_short}'"
                )
                # Build final step entry from cached data
                final_entry = _build_final_step_entry(
                    cached, step, seg, stage2_dir, step_dir_name, atomic_clips_dir,
                    manifest_path, clip_start_sec, clip_end_sec,
                )
                final_steps.append(final_entry)
                continue

        step_started = time.perf_counter()
        logger.info(
            f"[stage4] video_id={vid} step {i_step}/{total_steps} step_id={sid}: "
            f"goal='{step_goal_short}' clip={os.path.relpath(clip_path, video_out)} "
            f"({clip_start_sec:.2f}s..{clip_end_sec:.2f}s)"
        )

        os.makedirs(step_out_dir, exist_ok=True)
        os.makedirs(atomic_clips_dir, exist_ok=True)

        # Build the dense frame-index timeline for the step clip without saving local sampled frames.
        sampled_frames, _ = sample_video_to_frame_index(clip_path, sampling_cfg)
        num_frames = len(sampled_frames)

        # Adjust timestamps to original-video time (same pattern as stage 3)
        for fr in sampled_frames:
            try:
                fr["timestamp_sec"] = float(fr.get("timestamp_sec", 0.0)) + clip_start_sec
            except Exception:
                fr["timestamp_sec"] = clip_start_sec
            if source_fps > 0 and source_total_frames > 0:
                src_frame = int(round(float(fr.get("timestamp_sec", 0.0)) * source_fps))
                src_frame = min(max(0, src_frame), source_total_frames - 1)
                fr["source_frame_0based"] = src_frame
                fr["original_frame_index"] = src_frame
                fr["source_video_fps"] = source_fps
                fr["source_total_frames"] = source_total_frames
                fr["source_video_duration_sec"] = source_duration_sec

        # Keep clip-local timestamps for cutting atomic clips.
        # These are computed BEFORE the clip_start_sec offset was added,
        # i.e., relative to the step clip's own timeline (0-based from clip start).
        clip_duration = clip_end_sec - clip_start_sec
        local_timestamps: List[float] = []
        for fr in sampled_frames:
            local_t = float(fr.get("timestamp_sec", 0.0)) - clip_start_sec
            # Clamp to [0, clip_duration] to handle floating-point drift
            local_t = max(0.0, min(local_t, clip_duration))
            local_timestamps.append(local_t)

        write_frame_manifest(sampled_frames, None, manifest_path)

        # Build stage4 prompt
        step_annotation_json = json.dumps(
            {key: value for key, value in step.items() if key not in {"frame_ids", "bbox_tracks"}},
            ensure_ascii=False,
        )
        base_prompt = build_stage4_user_prompt(
            high_level_goal=high_level_goal,
            step_goal=step_goal,
            step_annotation_json=step_annotation_json,
            num_frames=num_frames,
        )

        write_text(sys_prompt_path, SYSTEM_PROMPT_ATOMIC_ACTION_DECOMPOSER)
        write_text(user_prompt_path, base_prompt)

        frames_content = build_video_api_content(
            clip_path,
            sampled_frames,
            video_fps=api_cfg.video_fps,
            include_manifest=True,
        )
        base_user_content = [{"type": "text", "text": base_prompt}] + frames_content
        system_msg = {"role": "system", "content": SYSTEM_PROMPT_ATOMIC_ACTION_DECOMPOSER}

        # Retry loop for VLM call
        last_content = ""
        last_errors: List[str] = []
        normalized_result: Optional[Dict[str, Any]] = None

        for attempt in range(1, max_retries + 1):
            logger.info(
                f"[stage4] video_id={vid} step {i_step}/{total_steps} step_id={sid} "
                f"model_call attempt={attempt}/{max_retries}"
            )
            if attempt == 1:
                user_content = base_user_content
            else:
                prefix = build_retry_prefix(last_errors, last_content)
                user_content = [{"type": "text", "text": prefix + base_prompt}] + frames_content

            messages = [system_msg, {"role": "user", "content": user_content}]
            content, usage = call_chat_completion(client, api_cfg, messages, max_tokens=api_cfg.max_tokens)
            stage_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            stage_usage["completion_tokens"] += usage.get("completion_tokens", 0)
            stage_usage["total_tokens"] += usage.get("total_tokens", 0)
            stage_usage["api_calls"] += 1
            last_content = content

            try:
                clean = extract_json_from_response(content)
                obj = json.loads(clean)
            except Exception as e:
                last_errors = [f"JSON parse error: {e}"]
                continue

            normalized_result, errs = validate_stage4_atomic_actions(
                obj,
                sid,
                num_frames,
                local_timestamps=local_timestamps,
                clip_duration=clip_duration,
            )
            if normalized_result is not None:
                break
            last_errors = errs

        if normalized_result is None:
            write_text(raw_path, last_content)
            raise RuntimeError(
                f"Stage 4 failed for step_id={sid} after {max_retries} attempts: "
                + " | ".join(last_errors[:10])
            )

        write_text(raw_path, last_content)
        write_json(step_out_path, normalized_result)

        # Cut atomic action clips from the step clip
        MIN_CLIP_DURATION = 0.04  # minimum 40ms (~1 frame at 25fps)
        for act in normalized_result["atomic_actions"]:
            aid = int(act["atomic_action_id"])
            s_idx = int(act["start_frame_index"])
            e_idx = int(act["end_frame_index"])
            clip_relpath = stage4_atomic_clip_relpath(step_dir_name, aid, str(act.get("action", "")))
            dst_path = os.path.join(stage4_dir, clip_relpath)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            # Compute LOCAL timestamps (relative to step clip start)
            local_start = local_timestamps[s_idx - 1]  # 1-based to 0-based
            if e_idx - 1 < len(local_timestamps):
                local_end = local_timestamps[e_idx - 1]
            else:
                # Last atomic action extends to end of clip
                local_end = clip_duration

            # Ensure positive and sufficient duration
            duration = local_end - local_start
            if duration < MIN_CLIP_DURATION:
                # Extend to at least MIN_CLIP_DURATION, but don't exceed clip boundary
                local_end = min(local_start + MIN_CLIP_DURATION, clip_duration)
                if local_end - local_start < MIN_CLIP_DURATION:
                    # If still too short (start is near clip end), shift start back
                    local_start = max(0.0, local_end - MIN_CLIP_DURATION)
                logger.warning(
                    f"[stage4] video_id={vid} step_id={sid} atomic_action_id={aid}: "
                    f"adjusted short duration ({duration:.3f}s) -> {local_end - local_start:.3f}s "
                    f"(local_start={local_start:.3f} local_end={local_end:.3f})"
                )

            logger.info(
                f"[stage4] video_id={vid} step_id={sid} atomic_action_id={aid}: "
                f"cutting {local_start:.2f}s..{local_end:.2f}s (clip-local) -> {clip_relpath}"
            )

            cut_video_segment_ffmpeg(
                ffmpeg_bin,
                clip_path,
                local_start,
                local_end,
                dst_path,
                overwrite=True,
                mode=cut_mode,
                seek_slop_sec=seek_slop_sec,
                crf=crf,
                preset=preset,
                keep_audio=keep_audio,
            )

            if not os.path.exists(dst_path) or os.path.getsize(dst_path) <= 0:
                logger.warning(
                    f"[stage4] video_id={vid} step_id={sid} atomic_action_id={aid}: "
                    f"ffmpeg produced an empty clip: {dst_path}"
                )

            # Store clip relpath in the action for the final merged output
            act["clip_relpath"] = clip_relpath

        # Also compute original-video timestamps for each atomic action
        orig_timestamps = [float(fr.get("timestamp_sec", 0.0)) for fr in sampled_frames]
        orig_frames_0based = [
            int(fr.get("source_frame_0based", fr.get("original_frame_index", -1))) for fr in sampled_frames
        ]
        try:
            clip_end_frame_0based_exclusive = int(seg.get("end_frame_0based_exclusive"))
        except Exception:
            clip_end_frame_0based_exclusive = int(round(float(clip_end_sec) * source_fps)) if source_fps > 0 else 0
        if source_total_frames > 0:
            clip_end_frame_0based_exclusive = min(max(0, clip_end_frame_0based_exclusive), source_total_frames)
        for act in normalized_result["atomic_actions"]:
            s_idx = int(act["start_frame_index"])
            e_idx = int(act["end_frame_index"])
            act["start_sec"] = orig_timestamps[s_idx - 1]
            if e_idx - 1 < len(orig_timestamps):
                act["end_sec"] = orig_timestamps[e_idx - 1]
            else:
                act["end_sec"] = clip_end_sec
            start_frame_0based = orig_frames_0based[s_idx - 1] if s_idx - 1 < len(orig_frames_0based) else -1
            if e_idx - 1 < len(orig_frames_0based):
                end_frame_0based_exclusive = orig_frames_0based[e_idx - 1]
            else:
                end_frame_0based_exclusive = clip_end_frame_0based_exclusive
            if start_frame_0based >= 0:
                end_frame_0based_exclusive = max(start_frame_0based + 1, end_frame_0based_exclusive)
            if source_total_frames > 0:
                end_frame_0based_exclusive = min(end_frame_0based_exclusive, source_total_frames)
            act["start_frame_0based"] = start_frame_0based
            act["end_frame_0based_exclusive"] = end_frame_0based_exclusive
            act["end_frame_0based_inclusive"] = max(start_frame_0based, end_frame_0based_exclusive - 1)

        # Write updated step output with clip_relpath and timestamps
        write_json(step_out_path, normalized_result)

        # Build final step entry
        final_entry = _build_final_step_entry(
            normalized_result, step, seg, stage2_dir, step_dir_name, atomic_clips_dir,
            manifest_path, clip_start_sec, clip_end_sec,
        )
        final_steps.append(final_entry)

        logger.info(
            f"[stage4] video_id={vid} step {i_step}/{total_steps} step_id={sid} done: "
            f"atomic_actions={len(normalized_result['atomic_actions'])} "
            f"elapsed={format_duration(time.perf_counter() - step_started)}"
        )

    # Recording-level object inventory: resolve stable entity_ref for every
    # object mention across all steps, then inject entity_ref everywhere.
    mentions = _collect_object_mentions(final_steps)
    object_inventory: List[Dict[str, Any]] = []
    mention_to_ref: Dict[str, str] = {}

    if mentions:
        cached_inventory = None if overwrite else _can_resume_object_inventory(inventory_path, mentions)
        if cached_inventory is not None:
            object_inventory, mention_to_ref = cached_inventory
            logger.info(f"[stage4] video_id={vid} object_inventory: reuse cached ({len(object_inventory)} entities)")
        else:
            steps_context = _build_object_inventory_context(high_level_goal, final_steps)
            base_prompt = build_stage4_object_inventory_user_prompt(
                high_level_goal=high_level_goal,
                steps_context=steps_context,
                mention_list=mentions,
            )
            system_msg = {"role": "system", "content": SYSTEM_PROMPT_OBJECT_INVENTORY}

            last_content = ""
            last_errors: List[str] = []
            for attempt in range(1, max_retries + 1):
                logger.info(
                    f"[stage4] video_id={vid} object_inventory model_call attempt={attempt}/{max_retries}"
                )
                if attempt == 1:
                    user_text = base_prompt
                else:
                    user_text = build_retry_prefix(last_errors, last_content) + base_prompt

                messages = [system_msg, {"role": "user", "content": [{"type": "text", "text": user_text}]}]
                content, usage = call_chat_completion(client, api_cfg, messages, max_tokens=api_cfg.max_tokens)
                stage_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                stage_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                stage_usage["total_tokens"] += usage.get("total_tokens", 0)
                stage_usage["api_calls"] += 1
                last_content = content

                try:
                    clean = extract_json_from_response(content)
                    obj = json.loads(clean)
                except Exception as e:
                    last_errors = [f"JSON parse error: {e}"]
                    continue

                object_inventory, mention_to_ref, errs = validate_object_inventory(obj, mentions)
                if object_inventory is not None and mention_to_ref is not None:
                    break
                last_errors = errs

            if object_inventory is None or mention_to_ref is None:
                write_text(os.path.join(stage4_dir, "object_inventory_raw_response.txt"), last_content)
                raise RuntimeError(
                    f"Stage 4 object inventory failed for video_id={vid} after {max_retries} attempts: "
                    + " | ".join(last_errors[:10])
                )

            write_json(inventory_path, {"object_inventory": object_inventory, "mentions": mention_to_ref})
            logger.info(f"[stage4] video_id={vid} object_inventory: {len(object_inventory)} entities resolved")

    _apply_entity_refs(final_steps, mention_to_ref)
    consistency_errors = _validate_applied_entity_refs(final_steps, mention_to_ref, object_inventory)
    if consistency_errors:
        raise RuntimeError(
            f"Stage 4 entity_ref consistency check failed for video_id={vid}: "
            + " | ".join(consistency_errors[:10])
        )
    _sync_entity_refs_to_step_caches(final_steps, step_output_paths)

    # Merge all steps into final plan
    final_plan: Dict[str, Any] = {
        "source_video": build_source_video_meta(video_path),
        "high_level_goal": high_level_goal,
        "object_inventory": object_inventory,
        "steps": final_steps,
    }
    write_json(final_path, final_plan)

    # Build root final_plan.json with video_out-relative paths.
    merged_steps: List[Dict[str, Any]] = []
    for s4_step in final_steps:
        sid = int(s4_step["step_id"])

        # Rebase clip_relpath: from stage4/-relative ("../stage2/...") to video_out-relative ("stage2/...")
        s4_clip = s4_step.get("clip_relpath", "")
        if s4_clip.startswith("../"):
            s4_clip = s4_clip[len("../"):]

        # Rebase atomic action clip_relpath and expose only original-video 0-based frame fields.
        merged_atoms: List[Dict[str, Any]] = []
        for act in s4_step.get("atomic_actions", []):
            a = _public_atomic_action_entry(dict(act))
            if a.get("clip_relpath") and not a["clip_relpath"].startswith("stage4/"):
                a["clip_relpath"] = os.path.join("stage4", a["clip_relpath"])
            merged_atoms.append(a)

        merged_step: Dict[str, Any] = {
            "step_id": sid,
            "step_goal": s4_step.get("step_goal", ""),
            "hand_object_relations": s4_step.get("hand_object_relations", []),
            "start_sec": s4_step.get("start_sec"),
            "end_sec": s4_step.get("end_sec"),
            "start_frame_0based": s4_step.get("start_frame_0based"),
            "end_frame_0based_exclusive": s4_step.get("end_frame_0based_exclusive"),
            "end_frame_0based_inclusive": s4_step.get("end_frame_0based_inclusive"),
            "clip_relpath": s4_clip,
            "atomic_actions": merged_atoms,
        }
        merged_steps.append(merged_step)

    final_merged: Dict[str, Any] = {
        "source_video": build_source_video_meta(video_path),
        "high_level_goal": high_level_goal,
        "object_inventory": object_inventory,
        "steps": merged_steps,
    }
    final_merged_path = os.path.join(video_out, "final_plan.json")
    write_json(final_merged_path, final_merged)
    logger.info(f"[stage4] video_id={vid} final_plan.json written: {final_merged_path}")

    # Update run_summary.json
    update_run_summary(
        run_summary_path,
        {
            "source_video": build_source_video_meta(video_path),
            "output_root": os.path.abspath(output_root),
            "schema_fingerprint": schema_fp,
            "updated_at_utc": now_utc_iso(),
            "stage4": {
                "status": "completed",
                "token_usage": stage_usage,
                "final_plan_path": os.path.relpath(final_path, video_out),
                "merged_final_plan_path": "final_plan.json",
                "object_inventory_path": os.path.relpath(inventory_path, video_out),
                "frame_index_note": (
                    "Public final outputs expose only original-video 0-based frame fields: "
                    "start_frame_0based, end_frame_0based_exclusive, and end_frame_0based_inclusive. "
                    "Internal dense boundary indices are kept only in raw/debug stage files."
                ),
                "api_config": {
                    "api_base_url": api_cfg.api_base_url,
                    "model_provider_id": api_cfg.model_provider_id,
                    "model_name": api_cfg.model_name,
                    "reasoning_effort": api_cfg.reasoning_effort,
                    "max_tokens": int(api_cfg.max_tokens),
                    "temperature": float(getattr(api_cfg, "temperature", 0.2)),
                    "api_call_retries": int(getattr(api_cfg, "api_call_retries", 1)),
                    "api_call_retry_backoff_sec": float(getattr(api_cfg, "api_call_retry_backoff_sec", 1.0)),
                },
            },
        },
    )

    logger.info(
        f"[stage4] video_id={vid} completed: steps={len(final_steps)} "
        f"elapsed={format_duration(time.perf_counter() - t_start)}"
    )
    return video_out


def _build_final_step_entry(
    normalized_result: Dict[str, Any],
    step: Dict[str, Any],
    seg: Dict[str, Any],
    stage2_dir: str,
    step_dir_name: str,
    atomic_clips_dir: str,
    manifest_path: str,
    clip_start_sec: float,
    clip_end_sec: float,
) -> Dict[str, Any]:
    """Build a merged final step entry for the atomic_plan_with_clips.json output.

    If the cached *normalized_result* already has ``start_sec``/``end_sec``/``clip_relpath``
    fields we use them; otherwise we recompute from the manifest.
    """
    sid = int(normalized_result["step_id"])
    step_goal = str(step.get("step_goal", "")).strip()
    clip_rel = seg.get("clip_relpath", "")

    # Stage 1 owns the reusable step clip; expose its path relative to stage4/.
    video_out = os.path.dirname(stage2_dir)
    stage4_dir = os.path.join(video_out, "stage4")
    source_clip_path = clip_rel if os.path.isabs(clip_rel) else os.path.join(video_out, clip_rel)
    clip_relpath_from_stage4 = os.path.relpath(source_clip_path, stage4_dir) if clip_rel else ""

    # Build atomic actions with timestamps and clip relpaths
    enriched_actions: List[Dict[str, Any]] = []
    actions = normalized_result.get("atomic_actions", [])

    # Try to load timestamps / original-video frame numbers from manifest if
    # not already in cached atomic actions.
    orig_timestamps: Optional[List[float]] = None
    orig_frames_0based: Optional[List[int]] = None
    needs_timestamps = actions and ("start_sec" not in actions[0])
    needs_original_frames = actions and ("start_frame_0based" not in actions[0])
    needs_clip_relpath = actions and ("clip_relpath" not in actions[0])

    if (needs_timestamps or needs_original_frames) and os.path.exists(manifest_path):
        try:
            manifest = read_json(manifest_path)
            frames = manifest.get("frames", [])
            if isinstance(frames, list) and frames:
                orig_timestamps = [
                    float(fr.get("timestamp_sec", 0.0)) if isinstance(fr, dict) else 0.0
                    for fr in frames
                ]
                orig_frames_0based = [
                    int(fr.get("source_frame_0based", fr.get("original_frame_index", -1))) if isinstance(fr, dict) else -1
                    for fr in frames
                ]
        except Exception:
            pass

    for act in actions:
        entry = dict(act)  # shallow copy

        # Add timestamps if missing
        if needs_timestamps and orig_timestamps:
            s_idx = int(act["start_frame_index"])
            e_idx = int(act["end_frame_index"])
            entry["start_sec"] = orig_timestamps[s_idx - 1] if s_idx - 1 < len(orig_timestamps) else clip_start_sec
            if e_idx - 1 < len(orig_timestamps):
                entry["end_sec"] = orig_timestamps[e_idx - 1]
            else:
                entry["end_sec"] = clip_end_sec

        # Add original-video 0-based frame numbers if missing.
        if needs_original_frames and orig_frames_0based:
            s_idx = int(act["start_frame_index"])
            e_idx = int(act["end_frame_index"])
            start_frame_0based = orig_frames_0based[s_idx - 1] if s_idx - 1 < len(orig_frames_0based) else -1
            if e_idx - 1 < len(orig_frames_0based):
                end_frame_0based_exclusive = orig_frames_0based[e_idx - 1]
            else:
                end_frame_0based_exclusive = int(seg.get("end_frame_0based_exclusive", start_frame_0based + 1) or start_frame_0based + 1)
            if start_frame_0based >= 0:
                end_frame_0based_exclusive = max(start_frame_0based + 1, end_frame_0based_exclusive)
            entry["start_frame_0based"] = start_frame_0based
            entry["end_frame_0based_exclusive"] = end_frame_0based_exclusive
            entry["end_frame_0based_inclusive"] = max(start_frame_0based, end_frame_0based_exclusive - 1)

        # Add clip relpath if missing
        if needs_clip_relpath:
            aid = int(act["atomic_action_id"])
            entry["clip_relpath"] = stage4_atomic_clip_relpath(step_dir_name, aid, str(act.get("action", "")))

        enriched_actions.append(_public_atomic_action_entry(entry))

    return {
        "step_id": sid,
        "step_goal": step_goal,
        "hand_object_relations": step.get("hand_object_relations", []),
        "start_sec": seg.get("start_sec"),
        "end_sec": seg.get("end_sec"),
        "start_frame_0based": seg.get("start_frame_0based"),
        "end_frame_0based_exclusive": seg.get("end_frame_0based_exclusive"),
        "end_frame_0based_inclusive": seg.get("end_frame_0based_inclusive"),
        "clip_relpath": clip_relpath_from_stage4,
        "atomic_actions": enriched_actions,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4: decompose each step into fine-grained atomic actions and cut atomic clips."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-video", help="Path to one video file.")
    src.add_argument("--input-video-dir", help="Directory of videos to process.")
    parser.add_argument(
        "--output-root",
        default=default_output_root(),
        help="Output root under ECCV/four_stage/...",
    )

    add_api_cli_args(parser, include_no_embed_index=True)
    add_sampling_cli_args(parser, default_max_frames=100, default_jpeg_quality=95)

    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument(
        "--cut-mode",
        choices=["copy", "reencode"],
        default="reencode",
        help="ffmpeg cut mode (copy is fast but may snap to keyframes).",
    )
    parser.add_argument(
        "--seek-slop-sec",
        type=float,
        default=1.0,
        help="Re-encode mode: pre-seek slop (seconds) for hybrid accurate seek.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="Re-encode mode: x264 CRF (lower=better quality).",
    )
    parser.add_argument(
        "--preset",
        default="veryfast",
        help="Re-encode mode: x264 preset.",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep/re-encode audio into the clip (default: drop audio).",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-legacy-resume",
        action="store_true",
        help="Allow resuming cached outputs whose run_summary.json lacks schema_fingerprint (legacy outputs).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Don't stop batch processing on single video failure.",
    )
    args = parser.parse_args()

    api_cfg = api_config_from_args(args)
    sampling_cfg = sampling_config_from_args(args)

    videos: List[str] = []
    if args.input_video:
        videos = [args.input_video]
    else:
        videos = collect_videos(args.input_video_dir, VIDEO_EXTS)

    if not videos:
        raise SystemExit("No videos found.")

    succeeded = 0
    failed = 0
    for vp in videos:
        try:
            run_stage4_for_video(
                vp,
                args.output_root,
                api_cfg,
                sampling_cfg,
                overwrite=args.overwrite,
                max_retries=args.max_retries,
                ffmpeg_bin=args.ffmpeg_bin,
                cut_mode=args.cut_mode,
                seek_slop_sec=args.seek_slop_sec,
                crf=args.crf,
                preset=args.preset,
                keep_audio=args.keep_audio,
                allow_legacy_resume=args.allow_legacy_resume,
            )
            succeeded += 1
        except Exception as e:
            failed += 1
            logger.error(f"[stage4] Failed for video {vp}: {e}")
            if not args.continue_on_error:
                raise

    if len(videos) > 1:
        logger.info(f"[stage4] Batch complete: {succeeded} succeeded, {failed} failed out of {len(videos)} videos.")


if __name__ == "__main__":
    main()
