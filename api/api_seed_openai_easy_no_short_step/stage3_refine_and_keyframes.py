#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from common import (
    _contains_frame_ref,
    _contains_time_ref,
    add_api_cli_args,
    add_sampling_cli_args,
    VIDEO_EXTS,
    api_config_from_args,
    build_video_api_content,
    build_retry_prefix,
    build_source_video_meta,
    call_chat_completion,
    collect_videos,
    default_output_root,
    ensure_frame_index_fps,
    ensure_video_out_dir_safe,
    extract_json_from_response,
    format_duration,
    get_video_metadata,
    guard_schema_fingerprint,
    initialize_api_client,
    logger,
    now_utc_iso,
    read_json,
    sample_video_to_frame_index,
    sampling_config_from_args,
    sanitize_filename,
    stage2_step_clip_relpath,
    update_run_summary,
    video_id_from_path,
    write_frame_manifest,
    write_json,
    write_text,
)
from prompts import (
    SYSTEM_PROMPT_OBJECT_INVENTORY,
    SYSTEM_PROMPT_STEP_GOAL_REFINER,
    build_stage3_step_goal_refinement_user_prompt,
    build_stage4_object_inventory_user_prompt,
)
from hand_step_pipeline import refine_hand_steps_text_only
from text_annotation_goal import NO_SUBTASK_GOAL, compose_video_goal

if TYPE_CHECKING:
    from common import ApiConfig, SamplingConfig


_STAGE3_ALLOWED_STEP_KEYS = {"step_id", "step_goal", "hand_object_relations"}
_STAGE3_ALLOWED_RELATION_KEYS = {"object", "hand", "relation", "entity_ref"}
_STAGE3_REQUIRED_RELATION_KEYS = {"object", "hand", "relation"}  # entity_ref is optional
_STAGE3_ALLOWED_HANDS = {"left hand", "right hand", "both hands", "unknown hand"}
_PLACEHOLDER_GOALS = {"", "unknown", "n/a", "na", "none", "null", "tbd", "todo", "...", "-"}
_STAGE3_OBJECT_PLACEHOLDERS = _PLACEHOLDER_GOALS | {
    "unknown object",
    "unidentified object",
    "the object",
    "object",
    "something",
}
_STAGE3_PLAN_STEP_KEYS = {
    "step_id",
    "step_goal",
    "hand_object_relations",
    "start_sec",
    "end_sec",
    "start_frame_0based",
    "end_frame_0based_exclusive",
    "end_frame_0based_inclusive",
    "clip_relpath",
}
_STAGE3_FORBIDDEN_KEYS = {
    "rationale",
    "counterfactual_challenge_question",
    "expected_challenge_outcome",
    "failure_reflecting",
    "critical_frames",
    "frame_index",
    "interaction",
    "keyframe_image_path",
    "atomic_actions",
}


def _step_plan_path(video_out: str) -> str:
    return os.path.join(video_out, "stage3", "step_plan.json")


def _step_refine_dir(stage3_dir: str, step_id: int) -> str:
    return os.path.join(stage3_dir, f"step_{int(step_id):03d}")


def _stage2_clip_relpath_for_final_step(step_id: int, final_step_goal: str) -> str:
    return stage2_step_clip_relpath(step_id, final_step_goal)


def _move_existing_path_aside(path: str) -> str:
    base = f"{path}.legacy_{int(time.time())}"
    candidate = base
    j = 0
    while os.path.exists(candidate):
        j += 1
        candidate = f"{base}_{j}"
    os.rename(path, candidate)
    return candidate


def _sync_stage2_clip_names_with_final_steps(*, video_out: str, final_plan: Dict[str, Any]) -> Dict[str, Any]:
    """Rename Stage-2 step clips to match final Stage-3/Stage-4 step goals."""
    stage2_dir = os.path.join(video_out, "stage2")
    segments_path = os.path.join(stage2_dir, "step_segments.json")
    if not os.path.exists(segments_path):
        return final_plan

    segments_data = read_json(segments_path)
    segs = segments_data.get("segments", [])
    steps = final_plan.get("steps", [])
    if not isinstance(segs, list) or not isinstance(steps, list):
        return final_plan

    seg_by_id: Dict[int, Dict[str, Any]] = {}
    for seg in segs:
        if not isinstance(seg, dict) or seg.get("step_id") is None:
            continue
        try:
            seg_by_id[int(seg.get("step_id"))] = seg
        except Exception:
            continue

    segments_changed = False
    for st in steps:
        if not isinstance(st, dict) or st.get("step_id") is None:
            continue
        try:
            sid = int(st.get("step_id"))
        except Exception:
            continue
        final_goal = str(st.get("step_goal", "")).strip()
        seg = seg_by_id.get(sid)
        if not final_goal or not isinstance(seg, dict):
            continue

        old_rel = str(seg.get("clip_relpath") or st.get("clip_relpath") or "").strip()
        if not old_rel:
            continue
        new_rel = _stage2_clip_relpath_for_final_step(sid, final_goal)
        old_abs = os.path.abspath(os.path.join(stage2_dir, old_rel))
        new_abs = os.path.abspath(os.path.join(stage2_dir, new_rel))

        if old_abs != new_abs:
            os.makedirs(os.path.dirname(new_abs), exist_ok=True)
            if os.path.exists(old_abs):
                if os.path.exists(new_abs):
                    same_file = False
                    try:
                        same_file = os.path.samefile(old_abs, new_abs)
                    except Exception:
                        same_file = False
                    if not same_file:
                        legacy = _move_existing_path_aside(new_abs)
                        logger.warning(
                            f"[stage3] step_id={sid} existing final-named Stage-2 clip moved aside: "
                            f"{os.path.relpath(legacy, video_out)}"
                        )
                os.rename(old_abs, new_abs)
                logger.info(
                    f"[stage3] step_id={sid} Stage-2 clip renamed to final step_goal: "
                    f"{os.path.relpath(new_abs, video_out)}"
                )
            elif not os.path.exists(new_abs):
                raise FileNotFoundError(f"Stage-2 clip not found for step_id={sid}: {old_abs}")

        if seg.get("clip_relpath") != new_rel:
            seg["clip_relpath"] = new_rel
            segments_changed = True
        if st.get("clip_relpath") != new_rel:
            st["clip_relpath"] = new_rel

    if segments_changed:
        write_json(segments_path, segments_data)

    return final_plan


_ALLOWED_IDENTITY_SCOPES = {"instance", "group", "unresolved"}


def _collect_object_mentions(steps: List[Dict[str, Any]]) -> List[str]:
    """Collect every distinct object-mention string across the WHOLE recording's
    hand_object_relations (all steps), in chronological order (step order, then
    relation order), de-duplicated by exact string. This is the input to the
    recording-level object-identity-resolution VLM call, which is what lets the
    same physical object keep the same entity_ref across steps even when the
    per-step VLM call used slightly different wording for it."""
    seen: Dict[str, None] = {}
    for step in steps:
        for rel in step.get("hand_object_relations", []) or []:
            if not isinstance(rel, dict):
                continue
            obj_name = str(rel.get("object", "")).strip()
            if obj_name and obj_name not in seen:
                seen[obj_name] = None
    return list(seen.keys())


def _build_object_inventory_context(high_level_goal: str, steps: List[Dict[str, Any]]) -> str:
    """Build a compact, read-only text context of the whole recording's plan
    (step goals + hand_object_relations) for the recording-level inventory prompt."""
    lines: List[str] = []
    for step in steps:
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
    return "\n".join(lines)


def validate_object_inventory(
    obj: Any,
    mentions: List[str],
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, str]], List[str]]:
    """Validate a recording-level object-inventory VLM response.

    Returns ``(inventory_list, mention_to_ref, errors)``. On failure, the first
    two are ``None`` and *errors* is non-empty. Mirrors Stage 4's validator so
    the same schema/guarantees apply here (every mention mapped to exactly one
    entity_ref; every entity_ref used in the mapping present in the inventory;
    no duplicate entity_ref values).
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


def _apply_entity_refs(steps: List[Dict[str, Any]], mention_to_ref: Dict[str, str]) -> None:
    """Mutate steps in place, injecting the recording-level canonical entity_ref
    into every hand_object_relations item. No text is rewritten; only the
    sibling `entity_ref` key is added/overwritten."""
    for step in steps:
        for rel in step.get("hand_object_relations", []) or []:
            if not isinstance(rel, dict):
                continue
            obj_name = str(rel.get("object", "")).strip()
            ref = mention_to_ref.get(obj_name)
            if ref:
                rel["entity_ref"] = ref


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


def _build_deterministic_inventory_fallback(
    steps: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, str]], Dict[str, str], List[str]]:
    """Build a conservative inventory when no API client is available.

    This path exists for legacy cached plans and unit/offline recovery only.
    The normal Stage-3 path uses the recording-level VLM resolver so that
    differently worded mentions can be reconciled across hand_steps.
    """
    mention_to_ref: Dict[str, str] = {}
    inventory_by_ref: Dict[str, Dict[str, str]] = {}
    used_refs: set[str] = set()
    warnings: List[str] = []

    for step in steps:
        for relation in step.get("hand_object_relations", []) or []:
            if not isinstance(relation, dict):
                continue
            mention = str(relation.get("object", "")).strip()
            if not mention:
                continue
            existing_ref = str(relation.get("entity_ref", "")).strip()
            if mention in mention_to_ref:
                continue
            if existing_ref:
                ref = existing_ref
            else:
                slug = re.sub(r"[^a-z0-9]+", "_", mention.lower()).strip("_") or "object"
                suffix = 1
                ref = f"obj_{slug}_{suffix}"
                while ref in used_refs:
                    suffix += 1
                    ref = f"obj_{slug}_{suffix}"
                warnings.append(
                    f"Generated deterministic entity_ref {ref!r} for {mention!r}; "
                    "recording-level VLM identity resolution was unavailable."
                )
            used_refs.add(ref)
            mention_to_ref[mention] = ref
            inventory_by_ref.setdefault(
                ref,
                {
                    "entity_ref": ref,
                    "canonical_name": mention,
                    "identity_scope": "instance",
                },
            )
    return list(inventory_by_ref.values()), mention_to_ref, warnings


def _resolve_object_inventory_for_recording(
    *,
    client: Any,
    api_cfg: "ApiConfig",
    high_level_goal: str,
    steps: List[Dict[str, Any]],
    stage3_dir: str,
    overwrite: bool,
    max_retries: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, int]]:
    """Recording-level object-identity resolution: resolve a stable `entity_ref`
    for every object mention across ALL steps in a SINGLE VLM call that sees the
    whole recording's plan at once.

    This replaces the old per-step string-matching heuristic (exact-match on
    lowercased `object` text, decided independently per Stage-3 step call),
    which could not:
      - recognize that two different-looking mentions in different steps (e.g.
        "the spoon" in step 2 vs "plastic spoon" in step 4) refer to the SAME
        physical object, since each step's VLM call has no visibility into any
        other step's wording; or
      - avoid collapsing two genuinely DIFFERENT objects that happen to share a
        generic mention string (e.g. two different cloths both called "cloth")
        into one entity_ref.

    Because this call sees every step's `step_goal` and `hand_object_relations`
    text together, it can make a single globally-consistent identity decision,
    the same mechanism already used and validated in Stage 4.
    """
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_calls": 0}
    inventory_path = os.path.join(stage3_dir, "object_inventory.json")

    mentions = _collect_object_mentions(steps)
    if not mentions:
        write_json(inventory_path, {"object_inventory": [], "mentions": {}})
        return [], {}, usage

    cached = None if overwrite else _can_resume_object_inventory(inventory_path, mentions)
    if cached is not None:
        inventory, mention_to_ref = cached
        logger.info(f"[stage3] object_inventory: reuse cached ({len(inventory)} entities)")
        return inventory, mention_to_ref, usage

    if client is None or api_cfg is None:
        inventory, mention_to_ref, warnings = _build_deterministic_inventory_fallback(steps)
        write_json(
            inventory_path,
            {
                "object_inventory": inventory,
                "mentions": mention_to_ref,
                "warnings": warnings,
            },
        )
        logger.warning(
            "[stage3] object_inventory: API unavailable; used deterministic legacy fallback"
        )
        return inventory, mention_to_ref, usage

    steps_context = _build_object_inventory_context(high_level_goal, steps)
    base_prompt = build_stage4_object_inventory_user_prompt(
        high_level_goal=high_level_goal,
        steps_context=steps_context,
        mention_list=mentions,
    )
    system_msg = {"role": "system", "content": SYSTEM_PROMPT_OBJECT_INVENTORY}

    last_content = ""
    last_errors: List[str] = []
    inventory: Optional[List[Dict[str, Any]]] = None
    mention_to_ref: Optional[Dict[str, str]] = None
    retries = max(1, int(max_retries))

    for attempt in range(1, retries + 1):
        logger.info(f"[stage3] object_inventory model_call attempt={attempt}/{retries}")
        if attempt == 1:
            user_text = base_prompt
        else:
            user_text = build_retry_prefix(last_errors, last_content) + base_prompt

        messages = [system_msg, {"role": "user", "content": [{"type": "text", "text": user_text}]}]
        content, call_usage = call_chat_completion(client, api_cfg, messages, max_tokens=api_cfg.max_tokens)
        usage["prompt_tokens"] += call_usage.get("prompt_tokens", 0)
        usage["completion_tokens"] += call_usage.get("completion_tokens", 0)
        usage["total_tokens"] += call_usage.get("total_tokens", 0)
        usage["api_calls"] += 1
        last_content = content

        try:
            clean = extract_json_from_response(content)
            obj = json.loads(clean)
        except Exception as e:
            last_errors = [f"JSON parse error: {e}"]
            continue

        inventory, mention_to_ref, errs = validate_object_inventory(obj, mentions)
        if inventory is not None and mention_to_ref is not None:
            break
        last_errors = errs

    if inventory is None or mention_to_ref is None:
        write_text(os.path.join(stage3_dir, "object_inventory_raw_response.txt"), last_content)
        raise RuntimeError(
            f"Stage 3 recording-level object_inventory resolution failed after {retries} attempts: "
            + " | ".join(last_errors[:10])
        )

    write_json(inventory_path, {"object_inventory": inventory, "mentions": mention_to_ref})
    logger.info(f"[stage3] object_inventory: {len(inventory)} entities resolved")
    return inventory, mention_to_ref, usage


def _find_forbidden_keys(obj: Any, path: str) -> List[str]:
    errors: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            next_path = f"{path}.{k}" if path else str(k)
            if k in _STAGE3_FORBIDDEN_KEYS:
                errors.append(f"Forbidden key '{k}' found at: {next_path}")
            errors.extend(_find_forbidden_keys(v, next_path))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            errors.extend(_find_forbidden_keys(v, f"{path}[{i}]"))
    return errors


def _validate_hand_object_relations(value: Any) -> Tuple[Optional[List[Dict[str, str]]], List[str]]:
    errors: List[str] = []
    if not isinstance(value, list):
        return None, ["hand_object_relations must be a JSON array."]
    if not value:
        return [], []

    normalized: List[Dict[str, str]] = []
    for i, item in enumerate(value):
        prefix = f"hand_object_relations[{i}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix} must be a JSON object.")
            continue

        missing_required = sorted(_STAGE3_REQUIRED_RELATION_KEYS - set(item.keys()))
        extra = sorted(set(item.keys()) - _STAGE3_ALLOWED_RELATION_KEYS)
        if missing_required:
            errors.append(f"{prefix} is missing required keys: {missing_required}")
        if extra:
            errors.append(f"{prefix} contains extra keys: {extra}")
        if missing_required or extra:
            continue

        raw_object = item.get("object")
        raw_hand = item.get("hand")
        raw_relation = item.get("relation")
        raw_entity_ref = item.get("entity_ref")
        if not isinstance(raw_object, str):
            errors.append(f"{prefix}.object must be a string.")
        if not isinstance(raw_hand, str):
            errors.append(f"{prefix}.hand must be a string.")
        if not isinstance(raw_relation, str):
            errors.append(f"{prefix}.relation must be a string.")
        if raw_entity_ref is not None and not isinstance(raw_entity_ref, str):
            errors.append(f"{prefix}.entity_ref must be a string when provided.")

        object_name = raw_object.strip() if isinstance(raw_object, str) else ""
        hand = raw_hand.strip().lower() if isinstance(raw_hand, str) else ""
        relation = raw_relation.strip().lower() if isinstance(raw_relation, str) else ""
        entity_ref = raw_entity_ref.strip() if isinstance(raw_entity_ref, str) else ""

        if object_name.lower() in _STAGE3_OBJECT_PLACEHOLDERS:
            errors.append(f"{prefix}.object is missing, empty, or a placeholder.")
        if relation.lower() in _PLACEHOLDER_GOALS:
            errors.append(f"{prefix}.relation is missing, empty, or a placeholder.")
        if hand not in _STAGE3_ALLOWED_HANDS:
            errors.append(
                f"{prefix}.hand must be one of {sorted(_STAGE3_ALLOWED_HANDS)}, got {hand!r}."
            )
        for key, text in (("object", object_name), ("relation", relation)):
            if _contains_frame_ref(text):
                errors.append(f"{prefix}.{key} must not reference frame/image indices.")
            if _contains_time_ref(text):
                errors.append(f"{prefix}.{key} must not reference timestamps, durations, or timecodes.")

        normalized.append({"object": object_name, "hand": hand, "relation": relation, "entity_ref": entity_ref})

    if errors:
        return None, errors
    return normalized, []


def validate_stage3_step_goal_output(obj: Any, expected_step_id: int) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    if not isinstance(obj, dict):
        return None, ["Stage 3 step-goal output must be a JSON object."]

    errors.extend(_find_forbidden_keys(obj, ""))
    extra = sorted(set(obj.keys()) - _STAGE3_ALLOWED_STEP_KEYS)
    if extra:
        errors.append(f"Stage 3 output contains extra keys (not allowed): {extra}")

    try:
        sid = int(obj.get("step_id"))
    except Exception:
        sid = -1
        errors.append(f"Invalid or missing step_id: {obj.get('step_id')!r}")
    if sid != int(expected_step_id):
        errors.append(f"step_id mismatch: expected {int(expected_step_id)}, got {sid}.")

    step_goal = str(obj.get("step_goal", "")).strip()
    if step_goal.lower() in _PLACEHOLDER_GOALS:
        errors.append("step_goal is missing, empty, or a placeholder.")
    if _contains_frame_ref(step_goal):
        errors.append("step_goal must not reference frame/image indices.")
    if _contains_time_ref(step_goal):
        errors.append("step_goal must not reference timestamps, durations, or timecodes.")

    hand_object_relations, relation_errors = _validate_hand_object_relations(
        obj.get("hand_object_relations")
    )
    errors.extend(relation_errors)

    if errors:
        return None, errors
    return {
        "step_id": int(expected_step_id),
        "step_goal": step_goal,
        "hand_object_relations": hand_object_relations,
    }, []


def _load_valid_cached_step_goal(step_out_path: str, expected_step_id: int) -> Optional[Dict[str, Any]]:
    if not os.path.exists(step_out_path):
        return None
    try:
        data = read_json(step_out_path)
    except Exception:
        return None
    normalized, errors = validate_stage3_step_goal_output(data, expected_step_id)
    if errors:
        return None
    return normalized


def _load_stage_inputs(draft_path: str, segments_path: str) -> Tuple[str, List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    draft = read_json(draft_path)
    segments_data = read_json(segments_path)

    high_level_goal = str(draft.get("high_level_goal", "")).strip()
    if not high_level_goal:
        raise RuntimeError("Stage 1 draft has an empty high_level_goal.")

    draft_steps = sorted(
        [s for s in draft.get("steps", []) if isinstance(s, dict)],
        key=lambda x: int(x.get("step_id", 0)),
    )
    if not draft_steps:
        raise RuntimeError("Stage 1 draft has no steps.")

    seg_by_id: Dict[int, Dict[str, Any]] = {}
    for seg in segments_data.get("segments", []):
        if not isinstance(seg, dict) or seg.get("step_id") is None:
            continue
        seg_by_id[int(seg["step_id"])] = seg
    if not seg_by_id:
        raise RuntimeError("Stage 2 segments missing/empty.")

    return high_level_goal, draft_steps, seg_by_id


def _can_resume_stage3_step_plan(final_path: str, draft_path: str, segments_path: str, video_out: str) -> bool:
    if not os.path.exists(final_path):
        return False
    if not os.path.exists(draft_path) or not os.path.exists(segments_path):
        return False
    try:
        plan = read_json(final_path)
        _, draft_steps, seg_by_id = _load_stage_inputs(draft_path, segments_path)
    except Exception:
        return False

    if not isinstance(plan, dict):
        return False
    if not isinstance(plan.get("high_level_goal"), str) or not plan.get("high_level_goal", "").strip():
        return False
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        return False

    draft_ids = {int(st.get("step_id", 0)) for st in draft_steps}
    if not draft_ids:
        return False

    stage2_dir = os.path.join(video_out, "stage2")
    stage3_dir = os.path.join(video_out, "stage3")
    plan_ids: set[int] = set()
    for st in steps:
        if not isinstance(st, dict) or st.get("step_id") is None:
            return False
        extra = sorted(set(st.keys()) - _STAGE3_PLAN_STEP_KEYS)
        if extra:
            return False
        try:
            sid = int(st.get("step_id"))
        except Exception:
            return False
        if sid in plan_ids:
            return False
        plan_ids.add(sid)
        if not isinstance(st.get("step_goal"), str) or not st.get("step_goal", "").strip():
            return False
        relations, relation_errors = _validate_hand_object_relations(st.get("hand_object_relations"))
        if relation_errors:
            return False
        clip_rel = st.get("clip_relpath")
        if not isinstance(clip_rel, str) or not clip_rel:
            return False
        if not os.path.exists(os.path.join(stage2_dir, clip_rel)):
            return False

        step_out_path = os.path.join(_step_refine_dir(stage3_dir, sid), "step_goal_refined.json")
        cached = _load_valid_cached_step_goal(step_out_path, sid)
        if (
            cached is None
            or cached.get("step_goal") != st.get("step_goal")
            or cached.get("hand_object_relations") != relations
        ):
            return False

    return plan_ids == draft_ids == set(seg_by_id.keys())


def _refine_one_step_goal(
    *,
    client: Any,
    api_cfg: "ApiConfig",
    sampling_cfg: "SamplingConfig",
    step_id: int,
    draft_step_goal: str,
    clip_path: str,
    clip_start_sec: float,
    source_fps: float,
    source_total_frames: int,
    source_duration_sec: float,
    step_out_dir: str,
    max_retries: int,
) -> Tuple[Dict[str, Any], Dict[str, int], int]:
    os.makedirs(step_out_dir, exist_ok=True)
    manifest_path = os.path.join(step_out_dir, "frame_manifest.json")
    raw_path = os.path.join(step_out_dir, "stage3_raw_response.txt")
    sys_prompt_path = os.path.join(step_out_dir, "stage3_system_prompt.txt")
    user_prompt_path = os.path.join(step_out_dir, "stage3_user_prompt.txt")

    sampled_frames, _ = sample_video_to_frame_index(clip_path, sampling_cfg)
    for fr in sampled_frames:
        try:
            fr["timestamp_sec"] = float(fr.get("timestamp_sec", 0.0)) + float(clip_start_sec)
        except Exception:
            fr["timestamp_sec"] = float(clip_start_sec)
        if source_fps > 0 and source_total_frames > 0:
            src_frame = int(round(float(fr.get("timestamp_sec", 0.0)) * source_fps))
            src_frame = min(max(0, src_frame), source_total_frames - 1)
            fr["source_frame_0based"] = src_frame
            fr["original_frame_index"] = src_frame
            fr["source_video_fps"] = source_fps
            fr["source_total_frames"] = source_total_frames
            fr["source_video_duration_sec"] = source_duration_sec
    num_frames = len(sampled_frames)
    write_frame_manifest(sampled_frames, None, manifest_path)

    draft_step = {"step_id": int(step_id), "step_goal": str(draft_step_goal or "").strip()}
    draft_step_json = json.dumps(draft_step, ensure_ascii=False, indent=2)
    user_prompt = build_stage3_step_goal_refinement_user_prompt(draft_step_json, num_frames, step_id)
    write_text(sys_prompt_path, SYSTEM_PROMPT_STEP_GOAL_REFINER)
    write_text(user_prompt_path, user_prompt)

    # Stage 3 must only provide the current step clip frames; no labels or frame manifest are sent.
    frames_content = build_video_api_content(
        clip_path,
        sampled_frames,
        video_fps=api_cfg.video_fps,
        include_manifest=False,
    )
    base_user_content = [{"type": "text", "text": user_prompt}] + frames_content
    system_msg = {"role": "system", "content": SYSTEM_PROMPT_STEP_GOAL_REFINER}

    last_content = ""
    last_errors: List[str] = []
    normalized: Optional[Dict[str, Any]] = None
    stage_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    attempts = 0
    retries = max(1, int(max_retries))

    for attempt in range(1, retries + 1):
        attempts = attempt
        logger.info(f"[stage3] step_id={int(step_id)} model_call attempt={attempt}/{retries}")
        if attempt == 1:
            user_content = base_user_content
        else:
            prefix = build_retry_prefix(last_errors, last_content)
            user_content = [{"type": "text", "text": prefix + user_prompt}] + frames_content

        messages = [system_msg, {"role": "user", "content": user_content}]
        content, usage = call_chat_completion(client, api_cfg, messages, max_tokens=api_cfg.max_tokens)
        stage_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        stage_usage["completion_tokens"] += usage.get("completion_tokens", 0)
        stage_usage["total_tokens"] += usage.get("total_tokens", 0)
        last_content = content

        try:
            clean = extract_json_from_response(content)
            obj = json.loads(clean)
        except Exception as e:
            last_errors = [f"JSON parse error: {e}"]
            continue

        normalized, errors = validate_stage3_step_goal_output(obj, step_id)
        if normalized is not None:
            break
        last_errors = errors

    write_text(raw_path, last_content)
    if normalized is None:
        raise RuntimeError(
            f"Stage 3 failed for step_id={int(step_id)} after {attempts} attempts: "
            + " | ".join(last_errors[:10])
        )
    return normalized, stage_usage, attempts


def _build_step_plan(
    *,
    video_path: str,
    video_out: str,
    draft_path: str,
    segments_path: str,
    client: Any,
    api_cfg: "ApiConfig",
    sampling_cfg: "SamplingConfig",
    overwrite: bool,
    max_retries: int,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    high_level_goal, draft_steps, seg_by_id = _load_stage_inputs(draft_path, segments_path)
    stage2_dir = os.path.join(video_out, "stage2")
    stage3_dir = os.path.join(video_out, "stage3")
    source_meta = get_video_metadata(video_path)
    source_fps = float(source_meta.get("fps", 0.0) or 0.0)
    source_total_frames = int(source_meta.get("total_frames", 0) or 0)
    source_duration_sec = float(source_meta.get("duration_sec", 0.0) or 0.0)

    final_steps: List[Dict[str, Any]] = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_calls": 0}
    total_steps = len(draft_steps)

    for i_step, st in enumerate(draft_steps, start=1):
        sid = int(st.get("step_id", 0))
        draft_step_goal = str(st.get("step_goal", "")).strip()
        if sid <= 0 or not draft_step_goal:
            raise RuntimeError(f"Invalid Stage 1 step: step_id={sid}, step_goal={draft_step_goal!r}")

        seg = seg_by_id.get(sid)
        if not seg:
            raise RuntimeError(f"Missing Stage 2 segment for step_id={sid}")
        clip_rel = str(seg.get("clip_relpath") or "").strip()
        if not clip_rel:
            raise RuntimeError(f"Missing clip_relpath for step_id={sid}")
        clip_path = os.path.join(stage2_dir, clip_rel)
        if not os.path.exists(clip_path):
            raise FileNotFoundError(f"Clip not found for step_id={sid}: {clip_rel}")

        step_out_dir = _step_refine_dir(stage3_dir, sid)
        step_out_path = os.path.join(step_out_dir, "step_goal_refined.json")
        cached = None if overwrite else _load_valid_cached_step_goal(step_out_path, sid)
        if cached is not None:
            refined = cached
            logger.info(
                f"[stage3] step {i_step}/{total_steps} step_id={sid}: reuse cached refined step annotation"
            )
        else:
            logger.info(
                f"[stage3] step {i_step}/{total_steps} step_id={sid}: refine step annotation from current clip"
            )
            refined, usage, attempts = _refine_one_step_goal(
                client=client,
                api_cfg=api_cfg,
                sampling_cfg=sampling_cfg,
                step_id=sid,
                draft_step_goal=draft_step_goal,
                clip_path=clip_path,
                clip_start_sec=float(seg.get("start_sec", 0.0) or 0.0),
                source_fps=source_fps,
                source_total_frames=source_total_frames,
                source_duration_sec=source_duration_sec,
                step_out_dir=step_out_dir,
                max_retries=max_retries,
            )
            write_json(step_out_path, refined)
            total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
            total_usage["total_tokens"] += usage.get("total_tokens", 0)
            total_usage["api_calls"] += int(attempts)

        final_steps.append(
            {
                "step_id": sid,
                "step_goal": str(refined.get("step_goal", "")).strip(),
                "hand_object_relations": refined.get("hand_object_relations", []),
                "start_sec": seg.get("start_sec"),
                "end_sec": seg.get("end_sec"),
                "start_frame_0based": seg.get("start_frame_0based"),
                "end_frame_0based_exclusive": seg.get("end_frame_0based_exclusive"),
                "end_frame_0based_inclusive": seg.get("end_frame_0based_inclusive"),
                "clip_relpath": clip_rel,
            }
        )

    return (
        {
            "source_video": build_source_video_meta(video_path),
            "high_level_goal": compose_video_goal(step["step_goal"] for step in final_steps),
            "steps": final_steps,
        },
        total_usage,
    )


def _run_stage3_text_only_hand_steps(
    *,
    video_path: str,
    output_root: str,
    api_cfg: "ApiConfig",
    overwrite: bool,
    max_retries: int,
) -> str:
    """Record completion only after all terminal artifacts are written."""
    video_out = os.path.join(output_root, video_id_from_path(video_path))
    ensure_video_out_dir_safe(video_out, video_path)
    summary_path = os.path.join(video_out, "run_summary.json")
    update_run_summary(summary_path, {
        "updated_at_utc": now_utc_iso(),
        "stage3": {
            "status": "running", "mode": "text_only_hand_steps",
            "started_at_utc": now_utc_iso(),
        },
    })
    try:
        video_out, usage, step_count = _build_stage3_text_only_outputs(
            video_path=video_path, output_root=output_root, api_cfg=api_cfg,
            overwrite=overwrite, max_retries=max_retries,
        )
        artifacts = {
            "step_plan_path": "stage3/step_plan.json",
            "hand_steps_path": "stage3/hand_steps.json",
            "final_plan_path": "final_plan.json",
            "instruction_subtask_path": "stage3/instruction_subtask.json",
            "object_inventory_path": "stage3/object_inventory.json",
        }
        for relative_path in artifacts.values():
            # Missing or malformed outputs must not produce a completed status.
            read_json(os.path.join(video_out, relative_path))
        update_run_summary(summary_path, {
            "updated_at_utc": now_utc_iso(),
            "stage3": {
                "status": "completed", "mode": "text_only_hand_steps",
                "completed_at_utc": now_utc_iso(), "step_count": step_count,
                "token_usage": usage, **artifacts,
            },
        })
    except (Exception, SystemExit) as exc:
        update_run_summary(summary_path, {
            "updated_at_utc": now_utc_iso(),
            "stage3": {
                "status": "failed", "mode": "text_only_hand_steps",
                "failed_at_utc": now_utc_iso(), "error_type": type(exc).__name__,
            },
        })
        raise
    return video_out


def _build_stage3_text_only_outputs(
    *,
    video_path: str,
    output_root: str,
    api_cfg: "ApiConfig",
    overwrite: bool,
    max_retries: int,
) -> Tuple[str, Dict[str, int], int]:
    """Run ROI-level hand-step review and globally resolve object identity."""
    video_out, hand_step_plan, _hand_step_usage = refine_hand_steps_text_only(
        video_path=video_path,
        output_root=output_root,
        api_cfg=api_cfg,
        overwrite=overwrite,
        max_retries=max_retries,
    )
    stage3_dir = os.path.join(video_out, "stage3")
    _generate_final_plan_without_stage4(
        video_out,
        hand_step_plan,
        stage3_dir,
        client=None,
        api_cfg=api_cfg,
        overwrite=overwrite,
        max_retries=max_retries,
    )
    final_plan_path = os.path.join(video_out, "final_plan.json")
    hand_steps_path = os.path.join(stage3_dir, "hand_steps.json")
    if os.path.exists(final_plan_path) and os.path.exists(hand_steps_path):
        final_payload = read_json(final_plan_path)
        hand_payload = read_json(hand_steps_path)
        relations_by_id = {
            int(step.get("step_id", 0)): step.get("hand_object_relations", [])
            for step in final_payload.get("steps", [])
            if isinstance(step, dict)
        }
        for hand_step in hand_payload.get("hand_steps", []):
            if not isinstance(hand_step, dict):
                continue
            sid = int(hand_step.get("hand_step_id", 0))
            if sid in relations_by_id:
                hand_step["hand_object_relations"] = relations_by_id[sid]
        hand_payload["object_inventory"] = final_payload.get("object_inventory", [])
        write_json(hand_steps_path, hand_payload)
    return video_out, _hand_step_usage, len(hand_step_plan.get("steps", []))


def run_stage3_for_video(
    video_path: str,
    output_root: str,
    api_cfg: "ApiConfig",
    sampling_cfg: "SamplingConfig",
    overwrite: bool,
    max_retries: int,
    *,
    allow_legacy_resume: bool = False,
) -> str:
    # Prefer the new text-only path whenever Stage 2 produced its boundary
    # artifact. Legacy cached folders without that file still follow the old
    # resume path below, preserving backwards compatibility.
    vid_for_mode = video_id_from_path(video_path)
    new_boundaries = os.path.join(output_root, vid_for_mode, "stage2", "hand_step_boundaries.json")
    if os.path.exists(new_boundaries):
        return _run_stage3_text_only_hand_steps(
            video_path=video_path,
            output_root=output_root,
            api_cfg=api_cfg,
            overwrite=overwrite,
            max_retries=max_retries,
        )

    t_start = time.perf_counter()
    ensure_frame_index_fps(sampling_cfg, getattr(api_cfg, "video_fps", 10.0))

    vid = video_id_from_path(video_path)
    video_out = os.path.join(output_root, vid)
    ensure_video_out_dir_safe(video_out, video_path)
    stage1_dir = os.path.join(video_out, "stage1")
    stage2_dir = os.path.join(video_out, "stage2")
    stage3_dir = os.path.join(video_out, "stage3")
    os.makedirs(stage3_dir, exist_ok=True)

    draft_path = os.path.join(stage1_dir, "draft_plan.json")
    segments_path = os.path.join(stage2_dir, "step_segments.json")
    final_path = _step_plan_path(video_out)
    run_summary_path = os.path.join(video_out, "run_summary.json")

    if not os.path.exists(draft_path):
        raise FileNotFoundError(f"Stage 1 draft not found: {draft_path}")
    if not os.path.exists(segments_path):
        raise FileNotFoundError(f"Stage 2 segments not found: {segments_path}")

    will_resume = not overwrite and _can_resume_stage3_step_plan(final_path, draft_path, segments_path, video_out)
    schema_fp = guard_schema_fingerprint(
        run_summary_path,
        video_out,
        stage="Stage 3",
        overwrite=overwrite,
        allow_legacy_resume=allow_legacy_resume,
        will_resume=will_resume,
    )

    # The client is needed even on the resume path: object_inventory resolution
    # only reuses its own cache if that cache is still valid for the current
    # mention set, and lazily falls back to a fresh recording-level VLM call
    # otherwise (e.g. after a step's hand_object_relations changed upstream).
    client = None
    if not will_resume or getattr(api_cfg, "api_key", None):
        client = initialize_api_client(api_cfg)
        if not client:
            raise SystemExit("Failed to initialize API client.")

    if will_resume:
        logger.info(f"[stage3] video_id={vid} resume: {os.path.relpath(video_out, output_root)}")
        try:
            final_plan = read_json(final_path)
            final_plan = _sync_stage2_clip_names_with_final_steps(video_out=video_out, final_plan=final_plan)
            write_json(final_path, final_plan)
        except Exception as e:
            raise RuntimeError(f"Failed to ensure final Stage-2 clip names for video_id={vid}: {e}") from e
        update_run_summary(
            run_summary_path,
            {
                "source_video": build_source_video_meta(video_path),
                "output_root": os.path.abspath(output_root),
                "schema_fingerprint": schema_fp,
                "updated_at_utc": now_utc_iso(),
                "stage3": {
                    "status": "completed",
                    "final_plan_path": os.path.relpath(final_path, video_out),
                    "resumed_at_utc": now_utc_iso(),
                    "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_calls": 0},
                },
            },
        )
        _generate_final_plan_without_stage4(
            video_out,
            final_plan,
            stage3_dir,
            client=client,
            api_cfg=api_cfg,
            overwrite=overwrite,
            max_retries=max_retries,
        )
        return video_out

    logger.info(f"[stage3] video_id={vid} start step-goal and hand-object-relation annotation from Stage 2 clips")
    update_run_summary(
        run_summary_path,
        {
            "source_video": build_source_video_meta(video_path),
            "output_root": os.path.abspath(output_root),
            "schema_fingerprint": schema_fp,
            "updated_at_utc": now_utc_iso(),
            "stage3": {"status": "running", "started_at_utc": now_utc_iso()},
        },
    )

    final_plan, stage_usage = _build_step_plan(
        video_path=video_path,
        video_out=video_out,
        draft_path=draft_path,
        segments_path=segments_path,
        client=client,
        api_cfg=api_cfg,
        sampling_cfg=sampling_cfg,
        overwrite=overwrite,
        max_retries=max_retries,
    )
    final_plan = _sync_stage2_clip_names_with_final_steps(video_out=video_out, final_plan=final_plan)
    write_json(final_path, final_plan)

    update_run_summary(
        run_summary_path,
        {
            "source_video": build_source_video_meta(video_path),
            "output_root": os.path.abspath(output_root),
            "schema_fingerprint": schema_fp,
            "updated_at_utc": now_utc_iso(),
            "stage3": {
                "status": "completed",
                "final_plan_path": os.path.relpath(final_path, video_out),
                "token_usage": stage_usage,
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
                "sampling_config": {
                    "max_frames": sampling_cfg.max_frames,
                    "resize_dimension": sampling_cfg.resize_dimension,
                    "jpeg_quality": sampling_cfg.jpeg_quality,
                    "frame_index_fps": sampling_cfg.frame_index_fps,
                },
            },
        },
    )

    logger.info(
        f"[stage3] video_id={vid} completed: steps={len(final_plan.get('steps', []))} "
        f"api_calls={stage_usage.get('api_calls', 0)} elapsed={format_duration(time.perf_counter() - t_start)}"
    )

    _generate_final_plan_without_stage4(
        video_out,
        final_plan,
        stage3_dir,
        client=client,
        api_cfg=api_cfg,
        overwrite=overwrite,
        max_retries=max_retries,
    )

    return video_out


def _restore_upstream_geometry(video_out: str, steps: List[Dict[str, Any]]) -> None:
    """Overwrite final-plan frame IDs and bbox tracks from the upstream-derived Stage-2 artifact."""
    boundaries_path = os.path.join(video_out, "stage2", "hand_step_boundaries.json")
    if not os.path.exists(boundaries_path):
        return
    boundaries = read_json(boundaries_path)
    upstream_by_id = {
        int(item["hand_step_id"]): item
        for item in boundaries.get("hand_steps", [])
        if isinstance(item, dict) and "hand_step_id" in item
    }
    for step in steps:
        step_id = int(step.get("step_id", 0))
        upstream = upstream_by_id.get(step_id)
        if upstream is None:
            raise RuntimeError(f"Stage 3 final plan has no upstream geometry for step_id={step_id}")
        start = int(upstream["start_frame_id"])
        end = int(upstream["end_frame_id"])
        upstream_frame_ids = [int(x) for x in upstream.get("frame_ids", [])]
        upstream_bbox_tracks = json.loads(json.dumps(upstream.get("bbox_tracks", {})))
        step["start_frame_0based"] = start
        step["end_frame_0based_inclusive"] = end
        step["end_frame_0based_exclusive"] = end + 1
        step["frame_ids"] = upstream_frame_ids
        step["bbox_tracks"] = upstream_bbox_tracks
        step["geometry_source"] = "upstream_hand_screening_exact_copy"
        # Fail closed if any malformed bbox/frame record somehow entered the
        # upstream-derived boundary artifact.
        for hand, records in upstream_bbox_tracks.items():
            if hand not in {"left_hand", "right_hand"} or not isinstance(records, list):
                raise RuntimeError(f"Invalid upstream bbox track for step_id={step_id}: {hand}")
            for record in records:
                if set(record.keys()) != {"frame_id", "bbox_xyxy", "bbox_source"}:
                    raise RuntimeError(
                        f"Unexpected upstream bbox fields for step_id={step_id}: {sorted(record.keys())}"
                    )
                if not (start <= int(record["frame_id"]) <= end):
                    raise RuntimeError(f"Upstream bbox frame_id is outside step_id={step_id}")
                if record.get("bbox_source") != "upstream_sam3":
                    raise RuntimeError(f"Non-upstream bbox found in step_id={step_id} final geometry")


def _generate_final_plan_without_stage4(
    video_out: str,
    stage3_plan: Dict[str, Any],
    stage3_dir: str,
    *,
    client: Any = None,
    api_cfg: Optional["ApiConfig"] = None,
    overwrite: bool = False,
    max_retries: int = 1,
) -> None:
    steps = stage3_plan.get("steps", [])
    stage3_plan["high_level_goal"] = compose_video_goal(step["step_goal"] for step in steps) if steps else NO_SUBTASK_GOAL
    write_json(os.path.join(stage3_dir, "step_plan.json"), stage3_plan)
    _restore_upstream_geometry(video_out, steps)

    # Stage 2 already generated one independent object mapping per authoritative
    # keep=true step. Preserve those mappings exactly; Stage 3 must not perform
    # recording-level remapping or rewrite entity_ref values.
    object_inventory_by_step: List[Dict[str, Any]] = []
    object_inventory: List[Dict[str, Any]] = []
    inventory_warnings: List[str] = []
    for step in steps:
        step_id = int(step.get("step_id", 0))
        entries = list(step.get("object_inventory", []) or [])
        object_inventory_by_step.append({"step_id": step_id, "object_inventory": entries})
        for entry in entries:
            if isinstance(entry, dict):
                object_inventory.append({"step_id": step_id, **entry})
    # Legacy/offline plans may not contain the Stage-2 inventory. Preserve a
    # deterministic fallback only for that case; normal pipeline runs keep
    # Stage-2 step-local entity_ref values unchanged.
    if not object_inventory:
        fallback, mention_to_ref, inventory_warnings = _build_deterministic_inventory_fallback(steps)
        _apply_entity_refs(steps, mention_to_ref)
        object_inventory = fallback
    write_json(
        os.path.join(stage3_dir, "object_inventory.json"),
        {
            "schema": "step_object_inventory.v1",
            "steps": object_inventory_by_step,
            "object_inventory": object_inventory,
            "warnings": inventory_warnings,
        },
    )
    logger.info(f"[stage3] per-step object_inventory.json written: {len(object_inventory_by_step)} steps")

    final_plan: Dict[str, Any] = {
        "source_video": stage3_plan.get("source_video", {}),
        "high_level_goal": stage3_plan.get("high_level_goal", ""),
        "object_inventory": object_inventory,
        "object_inventory_by_step": object_inventory_by_step,
        "steps": steps,
    }

    final_path = os.path.join(video_out, "final_plan.json")
    write_json(final_path, final_plan)
    logger.info(f"[stage3] final_plan.json written")

    _write_instruction_subtask(stage3_plan, stage3_dir)


def _write_instruction_subtask(stage3_plan: Dict[str, Any], stage3_dir: str) -> None:
    steps = stage3_plan.get("steps", [])
    sub_tasks: List[str] = []

    for step in steps:
        step_goal = str(step.get("step_goal", "")).strip()

        # Format explicit primary-hand evidence into each sub-task. Preserve
        # left/right labels whenever the clip makes laterality reliable.
        relations = step.get("hand_object_relations", []) or []
        hand_phrases: List[str] = []
        for rel in relations:
            if not isinstance(rel, dict):
                continue
            hand = str(rel.get("hand", "")).strip()
            relation = str(rel.get("relation", "")).strip()
            obj = str(rel.get("object", "")).strip()

            if not hand or not relation or not obj:
                continue

            if hand.lower() == "unknown hand":
                phrase = f"hand (left/right not visually distinguishable) {relation} {obj}"
            else:
                phrase = f"{hand} {relation} {obj}"
            hand_phrases.append(phrase)

        # Append hand action phrases to step_goal if present
        if hand_phrases:
            combined = step_goal + "; " + "; ".join(hand_phrases)
            sub_tasks.append(combined)
        else:
            sub_tasks.append(step_goal + "; no reliable left/right hand-object action is visible")

    instruction_subtask = {
        # Every segment shares the paragraph copied from chronological subtask goals.
        "instruction": compose_video_goal(sub_tasks) if sub_tasks else NO_SUBTASK_GOAL,
        "sub_task": sub_tasks,
    }
    instruction_subtask_path = os.path.join(stage3_dir, "instruction_subtask.json")
    write_json(instruction_subtask_path, instruction_subtask)
    logger.info(
        f"[stage3] instruction_subtask.json written: {len(instruction_subtask['sub_task'])} sub_tasks"
    )

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 3: refine each step_goal and annotate hand-object relations using only "
            "its Stage 2 clip and Stage 1 step."
        )
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-video", help="Path to one video file.")
    src.add_argument("--input-video-dir", help="Directory of videos to process.")
    parser.add_argument("--output-root", default=default_output_root(), help="Output root under ECCV/four_stage/...")

    add_api_cli_args(parser, include_no_embed_index=True)
    add_sampling_cli_args(parser, default_max_frames=100, default_jpeg_quality=95)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-legacy-resume",
        action="store_true",
        help="Allow resuming cached outputs whose run_summary.json lacks schema_fingerprint (legacy outputs).",
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

    for vp in videos:
        run_stage3_for_video(
            vp,
            args.output_root,
            api_cfg,
            sampling_cfg,
            overwrite=args.overwrite,
            max_retries=args.max_retries,
            allow_legacy_resume=args.allow_legacy_resume,
        )


if __name__ == "__main__":
    main()
