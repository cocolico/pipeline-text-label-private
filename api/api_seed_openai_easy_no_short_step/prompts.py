from __future__ import annotations

import json
from typing import List, Tuple


PRIMARY_CENTER_OPERATOR_POLICY = """
PRIMARY OPERATOR SCOPE (NON-NEGOTIABLE):
- If hands from more than one person or more than one hand pair are visible, annotate exactly ONE primary operator: the hand pair whose sustained manipulation workspace is closest to the image center over time.
- Determine the primary operator from the pair/workspace centroid across the clip, not from one isolated frame, detection confidence, apparent hand size, or whichever hand is most visually salient.
- Maintain the same primary-operator identity through temporary occlusion, crossing, or brief motion. Do not switch to an edge/background person's hands merely because the primary hands are momentarily hidden.
- Ignore every action, intention, transition, contact, object interaction, and state change performed only by secondary hands near the image edges or in the background. Secondary-hand motion must never create a step, boundary, relation, or atomic action.
- `left hand` and `right hand` always mean the primary operator's anatomical hands only.
""".strip()


# Shared instruction block for Doubao-Seed-2.0-lite-260428 video understanding.
# The Ark adapter uploads the local MP4 as `input_video` and sets the Files API
# video preprocessing FPS; prompts must therefore refer to the attached video,
# its chronological visual evidence, and the requested JSON schema rather than
# incorrectly implying that a separate image list was supplied.
DOUBAO_VIDEO_INPUT_CONTRACT = """
INPUT MEDIA AND EVIDENCE:
- The attached input is one continuous video, processed by the API at the configured video-understanding FPS in chronological order.
- Inspect the video from beginning to end. Base every decision on visible video evidence; do not infer hidden/off-screen events.
- When deciding whether an action occurred, compare its visible start, manipulation, and result. Presence of a person, hand, or object alone is not evidence that the requested action occurred.
- Treat any supplied frame-index/timestamp manifest or detector cue as auxiliary alignment information. The attached video remains the source of truth.

OUTPUT CONTRACT:
- Return exactly the requested JSON object and no Markdown, prose outside JSON, comments, or extra keys.
- Use the exact field names, types, allowed values, and ordering constraints stated in the task schema.
""".strip()


SYSTEM_PROMPT_STEP_PLANNER = f"""
Your output MUST be a single, syntactically flawless JSON object. JSON validity is a critical, non-negotiable requirement.
Return JSON only: no markdown, no comments, no extra text.
Ensure outputs cover the entire video timeline from the first provided frame to the last provided frame.
{PRIMARY_CENTER_OPERATOR_POLICY}
""".strip()

SYSTEM_PROMPT_ATOMIC_ACTION_DECOMPOSER = f"""
Your output MUST be a single, syntactically flawless JSON object. JSON validity is a critical, non-negotiable requirement.
Return JSON only: no markdown, no comments, no extra text.
{PRIMARY_CENTER_OPERATOR_POLICY}
""".strip()

SYSTEM_PROMPT_STEP_GOAL_REFINER = f"""
You are an expert video step-goal and hand-object-relation annotation assistant.
Your output MUST be a single, syntactically flawless JSON object. JSON validity is a critical, non-negotiable requirement.
Return JSON only: no markdown, no comments, no extra text.
Use only the provided current step clip frames and the provided current Stage 1 draft step JSON.
Internally work in two ordered passes before you output anything: (1) refine `step_goal` around the primary operator's task-relevant hand action(s), including simultaneous actions by that operator's left and right hands when both are visibly meaningful, (2) draft and self-review only the `hand_object_relations` that directly ground those action(s). Ignore secondary/background people and unrelated simultaneous events. Only the final, self-reviewed JSON object is returned; never reveal or output the intermediate draft.
{PRIMARY_CENTER_OPERATOR_POLICY}
""".strip()

def build_stage1_user_prompt(num_frames: int, image_dimensions: Tuple[int, int]) -> str:
    w, h = image_dimensions
    return f"""
Analyze the attached continuous video. Its configured dense timeline contains {num_frames} chronological boundary points; use the attached video as the source of truth.

TASK:
Generate a step-by-step plan for the entire video.

{DOUBAO_VIDEO_INPUT_CONTRACT}

{PRIMARY_CENTER_OPERATOR_POLICY}

FULL-VIDEO COVERAGE (NON-NEGOTIABLE):
- The ordered `steps` MUST collectively cover the ENTIRE timeline from the FIRST frame to the LAST frame.
- The plan MUST NOT end early: the LAST step MUST include and reflect the last portion of the video (the last frames). Do NOT invent an "achieved final state" if the video ends mid-action; describe the last observed state and any visible ongoing action.
- Do NOT compress the whole plan into only the early/middle frames; later steps MUST reflect later-video events.
- Each step MUST correspond to a contiguous, localizable time interval in the video. Do NOT interleave events from different times inside the same step.

Language & grounding:
- Use objective, professional English.
- Do not hallucinate hidden states or off-screen objects.
- Use 3-8 steps total (prefer 4-7) at a MEDIUM granularity that is realistic to localize with {num_frames} sampled frames.
- Step segmentation guidance (complete but not fragmented):
  - Do NOT merge two clearly different subtasks into one step (e.g., different primary goal, different main object/patient, tool change, or location change).
  - Do NOT over-split one coherent interaction into tiny micro-steps (e.g., reach/grasp/lift) unless there is a clear intermediate world-state outcome that matters for later steps.
  - Each step should be long enough to show meaningful progress and a stable intermediate outcome that is visually anchorable in the frames.
  - For human operation videos, a very brief intention/transition action (typically <= 2.0 seconds) should NOT become its own step. Merge brief reach, release, withdrawal, reposition, setup, or transition motions into the adjacent coherent step unless that short interval creates a clear, independent, task-relevant stable world-state outcome.
- Use consistent object naming across all steps (do not rename the same object with different synonyms).
- All required fields MUST be present and non-empty (no empty strings, empty arrays, empty objects, or null). In any string field (including list elements), do NOT reference frame/image indices, timestamps, durations, or timecodes. Avoid placeholders like "unknown", "N/A", "...".

Output format (strict JSON only; no extra text):
{{
  "high_level_goal": "One comprehensive English sentence describing the overall goal and intended final outcome of the entire video (focus on the final world state; do NOT list steps).",
  "steps": [
    {{
      "step_id": 1,
      "step_goal": "One concise English sentence centered on the primary operator's hands: state the main hand-performed action, the directly manipulated object, and the visible result when needed."
    }}
  ]
}}

Additional constraints:
- Step ordering MUST follow the chronological order implied by the frames.
- `step_id` MUST start at 1 and increase by 1.
- Each `step_goal` must be specific and not duplicated across steps.
- Keep each `step_goal` focused on a single sub-goal (avoid multi-action conjunctions).
- Make the primary operator's hand action the grammatical and semantic center of every `step_goal` (for example: "The right hand tilts the pitcher to pour liquid into the cup").
- Describe what the primary hands physically do to the directly manipulated object; include a resulting object state only when visibly necessary to distinguish the action.
- Do NOT summarize everything that happens in the clip. Omit secondary/background people, their actions, passive scene events, camera motion, and objects that are merely visible.
- Preserve a simultaneous/parallel clause when it describes another visible, task-relevant action by the SAME primary operator's other hand. Use clear wording such as "while the left hand stabilizes the table" when it helps ground the manipulation.
- Do NOT join an unrelated/background action with "while", "as", or "during". In particular, omit actions of other people even when simultaneous and visually obvious.
- Do NOT add any extra keys beyond the schema above.
- Each step should be anchorable to visual evidence.


Now output the final strict JSON object only.
""".strip()

def build_stage2_user_prompt(high_level_goal: str, draft_plan_outline: str, num_frames: int) -> str:
    return f"""
You are an expert video step temporal localization assistant.
You are given:
1) {num_frames} dense frame-index boundary points from the FULL original video (chronological order), sampled at the configured video understanding FPS rather than a fixed-count grid.
2) A draft step list extracted from a plan (read-only; do NOT edit it).

High-level goal (context): {high_level_goal}

Draft steps (read-only):
{draft_plan_outline}

Note on indices:
- Some adjacent dense boundary points may look visually similar; avoid choosing a segment whose boundaries fall on visually identical frames with no time progress.

Task:
For EACH step, predict the corresponding time interval in the original video by selecting:
- `start_frame_index`: the 1-based index of the boundary timestamp where this step starts (inclusive).
- `end_frame_index`: the 1-based index of the boundary timestamp where this step ends (exclusive; the first frame AFTER the step ends).

{PRIMARY_CENTER_OPERATOR_POLICY}

Interpretation:
- Let `t(i)` be the timestamp of sampled frame `i`.
- The manifest also provides `source_frame_0based` for each index; this is the corresponding 0-based frame number in the original full video.
- The step clip is cut as the half-open interval `[t(start_frame_index), t(end_frame_index))`.
- Because boundaries are on a shared grid, `end_i` may equal `start_(i+1)` (contiguous, no overlap).

Procedure (MANDATORY TWO-PASS BOUNDARY DETERMINATION):

PASS 1 — Identify step completion moments:
For EACH step, scan the frames to find the LAST frame where this step's action is STILL IN PROGRESS or has JUST COMPLETED. This is the "completion frame" — the frame where:
  - The hand/tool has fully released the object AND the object is in its final resting position for this step, OR
  - The body has returned to a neutral/transition pose before starting the next action, OR
  - The described world-state change has become fully visible and stable.

PASS 2 — Identify next-step initiation moments:
For EACH step (except the last), find the FIRST frame where the NEXT step's action has CLEARLY BEGUN. This is the "initiation frame" — the frame where:
  - The hand/tool has begun reaching toward the NEXT step's target object, OR
  - A new grip/contact is being established on a DIFFERENT object, OR
  - The agent's body is clearly oriented toward the next task's workspace.

BOUNDARY PLACEMENT RULE:
- The boundary (end_frame_index for step i / start_frame_index for step i+1) should be placed BETWEEN the completion frame of step i and the initiation frame of step i+1.
- If the completion frame and initiation frame are adjacent (no gap), place the boundary so the completion frame belongs to step i (i.e., end_frame_index = completion_frame + 1).
- If there are idle/transition frames between completion and initiation, assign them to step i (the completed step) to avoid contaminating step i+1's clip with unrelated motion.

ASYMMETRIC ERROR POLICY (NON-NEGOTIABLE):
- It is MUCH WORSE to include next-step frames in the current step's clip than to lose a frame at the tail of the current step.
- When uncertain, prefer ending the current step EARLIER (even if you lose the last 1-2 frames of the step's action) rather than LATER (which risks including the beginning of the next step).
- Specifically: if you cannot determine whether frame F shows the current step completing or the next step beginning, assign frame F to the NEXT step (set end_frame_index of current step to F, not F+1).

WHAT CONSTITUTES A STEP BOUNDARY (visual cues to look for):
1. RELEASE-REACH transition: The hand releases the current object (fingers open, no contact), and then begins reaching toward a new object. The boundary is AFTER the release is complete but BEFORE any reaching toward the next object begins.
2. PLACEMENT-WITHDRAWAL transition: An object is placed in its final position (no longer moving), and the hand begins withdrawing. The boundary is AFTER the object is stationary.
3. TOOL CHANGE: A tool is put down and a different tool is picked up. The boundary is AFTER the first tool is released.
4. WORKSPACE SHIFT: The agent's focus/body orientation shifts from one area to another. The boundary is at the moment of shift.
5. POSE RESET: The agent returns to a neutral stance between actions. The boundary is at the neutral pose.

IMPORTANT:
- Output MUST contain only integer fields (no string fields and no extra text). All required keys MUST be present.
- Indices refer ONLY to the provided dense frame-index manifest (1..{num_frames}); choose the most accurate visible boundary available on that dense grid.
- You MAY output `{num_frames + 1}` ONLY for `end_frame_index` to indicate the exclusive boundary AFTER the last provided frame (typically for the last step to cover the video end).
- Output must contain exactly one entry per draft `step_id` and MUST NOT include any extra step_ids.
- Enforce monotonic, contiguous, non-overlapping segments (no gaps): for consecutive steps, `end_i == start_(i+1)`.
- Do NOT leave uncovered time between steps; if uncertain, choose boundaries that preserve full coverage rather than risking missing late-stage events.
- HARD full-video coverage (NON-NEGOTIABLE): `start_1` MUST be `1` and `end_last` MUST be `{num_frames + 1}` (use `{num_frames + 1}` to indicate the exclusive boundary AFTER the last provided frame). Do NOT end early.
- BOUNDARY ACCURACY (NON-NEGOTIABLE): Choose `start_frame_index` / `end_frame_index` so each step clip is accurate, complete, and rigorous:
  - Each step clip [start_frame_index, end_frame_index) MUST contain the COMPLETE execution of that step's action to its conclusion (the step's END STATE must be visible in the clip).
  - Each step clip MUST NOT contain frames showing the NEXT step's action in progress (no next-step reaching, gripping, or manipulating a different object).
  - Prefer the smallest interval that fully contains the step. Do NOT pad extra frames "just in case" — that causes next-step bleeding.
  - When the step ends with a release/placement, the boundary should capture the release completion but exclude any subsequent reaching toward a new object.
- Output MUST be exactly one JSON object with a single top-level key `steps` (no other top-level keys).

Output format (strict JSON only):

Field definitions (read carefully; output JSON must contain ONLY the keys in the template):
- `steps` (list): Exactly one entry per draft `step_id` (no extra/missing ids), in ascending `step_id` order.
- `steps[*].step_id` (int): Draft step identifier (must match exactly; do not renumber/reorder).
- `steps[*].start_frame_index` (int): Inclusive start boundary (1-based, within [1, {num_frames}]) on the dense manifest. Choose the boundary where the step begins; when uncertain, bias slightly earlier to preserve context for Stage 3.
- `steps[*].end_frame_index` (int): Exclusive end boundary (1-based, within [2, {num_frames + 1}]) on the dense manifest. Choose the first boundary AFTER the step's action has fully completed AND BEFORE any next-step action begins. Must satisfy `start_frame_index < end_frame_index`. Use `{num_frames + 1}` for the last step.
- `steps[*].value` (string): Must be exactly "yes" or "no" (lowercase). Output "yes" ONLY when the primary center operator's hands are visibly present for a sustained duration (more than a brief moment) AND performing meaningful, task-relevant hand actions (grasping, manipulating, operating objects) within this step's time interval. Output "no" if hands are absent, only briefly visible, merely hovering without contact, or performing no task-relevant manipulation.

Output JSON template (replace the numbers with your chosen indices; keep keys exactly):
{{
  "steps": [
    {{
      "step_id": 1,
      "start_frame_index": 1,
      "end_frame_index": 2,
      "value": "yes"
    }},
    {{
      "step_id": 2,
      "start_frame_index": 2,
      "end_frame_index": 5,
      "value": "no"
    }}
  ]
}}
""".strip()


def build_stage3_step_goal_refinement_user_prompt(draft_step_json: str, num_frames: int, step_id: int) -> str:
    return f"""
You are an expert video step-goal refinement assistant.
You are given ONE attached, already-cut single-step video clip plus a {num_frames}-point chronological timeline and the Stage 1 draft JSON for ONLY this same step.

{DOUBAO_VIDEO_INPUT_CONTRACT}

TASK (perform internally as two ordered passes; output ONLY the final result, never the intermediate draft):
Pass 1 - Refine the step's `step_goal` around the primary operator's visible task-relevant hand action(s) in THIS step clip. Preserve a simultaneous clause when the primary operator's other hand performs a meaningful supporting/manipulation action. Finalize this refined `step_goal` before starting Pass 2.
Pass 2 - Using that hand-action-centered `step_goal`, draft and self-review the task-relevant hand-object relations that directly ground EVERY retained primary-hand clause. Drop relations belonging only to secondary/background people or unrelated events. Output only the self-reviewed result.

{PRIMARY_CENTER_OPERATOR_POLICY}

Strict input boundary:
- Use ONLY the current step clip frames and the current Stage 1 draft step JSON below.
- Do NOT infer from other steps, the full video, future clips, previous clips, filenames, or any hidden context.
- Treat the frames as the primary evidence and the Stage 1 step JSON as a draft to correct.

Step-goal requirements:
- You MUST NOT change `step_id`.
- If the draft `step_goal` is already accurate for this clip, keep it with the smallest necessary edit.
- If the draft `step_goal` is too broad, too narrow, contains multiple independent actions, or includes details not visible in this clip, rewrite it to be clip-consistent.
- `step_goal` MUST be one concise English sentence whose main subject/agent is the primary operator's hand(s), whose main verb is the visible physical manipulation, and whose object is the directly manipulated object. It may contain a simultaneous clause for the SAME primary operator's other hand when that hand performs a meaningful task-relevant action.
- Focus on what the primary hands physically accomplish (for example grasp, lift, pour, wipe, press, place, rotate, insert, remove), not on a broad summary of what the entire video clip or all visible people accomplish.
- Include a visible result state only when it clarifies the hand action; do not replace the action with an abstract intention or business-level outcome.
- Keep parallel actions by the SAME primary operator's hands when they are visibly task-relevant. For example, keep "The right hand lifts the cup while the left hand stabilizes the table"; both clauses must be grounded in `hand_object_relations`.
- Omit actions performed only by secondary/background people, even when simultaneous and visually obvious. For example, reduce "The right hand lifts the cup while the left hand stabilizes the table and a nearby person turns their head" to "The right hand lifts the cup while the left hand stabilizes the table." Do not append the other person's action with "while", "as", or "during".
- Omit passive scene descriptions, camera movement, people merely standing nearby, and objects that the primary hands do not manipulate.
- Use objective, professional English.
- Do not hallucinate hidden states or off-screen objects.
- Use consistent object naming from the draft when it is visually supported.

Hand-object-relation requirements:
- `hand_object_relations` MUST be a JSON array. Each item MUST contain exactly `object`, `hand`, and `relation`.
- Use an empty array ONLY when no task-relevant hand-object relation by the primary center operator is visibly supported anywhere in this step clip. Never use a secondary edge/background hand or its object merely to make this array non-empty.
- Each item represents one distinct, task-relevant hand-object relation that is visibly supported in this step clip. If the manipulated object, participating hand, or meaningful relation changes, add another item in chronological order.
- `object` MUST be a specific, appearance-grounded English noun phrase for the object directly approached, contacted, grasped, held, supported, operated, or released by the hand. Do not use pronouns, infer hidden identity, or list objects that are merely visible.
- When multiple same-category objects are visible, give each manipulated instance a stable distinguishing descriptor (for example color, material, relative position, or ordinal) and reuse that exact name only for the same physical instance. Do not collapse two distinct objects into one generic label.
- `hand` MUST be exactly one of: `left hand`, `right hand`, `both hands`, or `unknown hand`.
- Use anatomical `left hand` or `right hand` only when laterality is visually reliable. Image-left/image-right position alone is not anatomical evidence. Use `both hands` only for a joint relation involving both hands; otherwise use `unknown hand` when laterality cannot be established without guessing.
- `relation` MUST be a concise lower-case English gerund or contact/spatial phrase describing the visible hand-to-object relation, such as `reaching toward`, `touching`, `grasping`, `holding`, `supporting`, `pressing`, or `releasing`. Do not write a full action caption in this field.
- Include all meaningful task-relevant relations needed to characterize the step, but do not enumerate incidental contact, imperceptible micro-adjustments, or unsupported occluded states.
- When a tool mediates the operation, annotate the visible direct relation between each hand and the tool or target object (for example, one hand `holding` the scissors and the other hand `stabilizing` the pouch).
- Each relation MUST be consistent with the Pass-1 refined `step_goal`: the `object` and `relation` must plausibly belong to the same intermediate world-state outcome described by the refined `step_goal`. Discard draft relations that belong to a different step or are not implied by the refined `step_goal`, even if visible in the clip.
- Relations must cover every retained primary-hand action stated in `step_goal`, including a simultaneous supporting action by the other hand; unrelated simultaneous/background actions must not appear in either field.

Shared output requirements:
- All required fields MUST be present and non-empty (no empty strings, empty objects, or null). The only permitted empty array is `hand_object_relations` under the no-visible-primary-relation condition above.
- In any string field, do NOT reference frame/image indices, timestamps, durations, or timecodes.
- Avoid placeholders like `N/A` or `...`. `unknown hand` is permitted only when anatomical handedness is not visually reliable; it is not permission to guess the object or relation.

Stage 1 draft step JSON (read-only; current step only):
```json
{draft_step_json}
```

Output format (strict JSON only; no markdown, no comments, no extra text):
{{
  "step_id": {int(step_id)},
  "step_goal": "One concise English sentence centered on the primary operator's hand-performed manipulation and directly handled object.",
  "hand_object_relations": [
    {{
      "object": "stainless steel pitcher",
      "hand": "right hand",
      "relation": "grasping"
    }}
  ]
}}

Additional constraints:
- `step_id` MUST match the draft step exactly.
- Do NOT add any extra keys beyond the schema above.

Now output the final strict JSON object only.
""".strip()


def build_stage4_user_prompt(
    high_level_goal: str,
    step_goal: str,
    step_annotation_json: str,
    num_frames: int,
) -> str:
    """Build the Stage 4 user prompt for atomic action decomposition of a single step clip.

    The VLM receives dense frame-index frames from the step clip (same pool as stage 3)
    and decomposes the step into conservative atomic actions with temporal boundaries.
    """
    return f"""
You are an expert Physical Interaction Analyst specializing in conservative, visually grounded atomic action decomposition.
You are given {num_frames} dense frame-index frames from a SINGLE STEP CLIP (chronological order), sampled at the configured video understanding FPS.

High-level goal (context): {high_level_goal}
Step goal (context): {step_goal}

Reference step annotation (read-only; for context — do NOT echo in output):
```json
{step_annotation_json}
```

{PRIMARY_CENTER_OPERATOR_POLICY}

Task:
Decompose this step into conservative, minimum-sufficient ATOMIC ACTIONS. Each atomic action should be one visually coherent physical operation
that creates a task-relevant state change or sustained manipulation phase (e.g., reach/grasp/lift when continuous, move, tilt, pour, place/release when continuous, rotate, press, pull, push, insert, align, adjust, slide, flip, fold, unfold, squeeze, peel, tear, screw, unscrew). Do NOT split one continuous operation just because contact, release, or hand pose changes briefly within it.

RECOMMENDED PROCEDURE (follow silently):
1) Scan ALL frames first to understand the full motion trajectory and state changes within this step.
2) Identify only task-relevant stable state-change boundaries: a new object becomes engaged, the primary patient/ownership/tool-use changes, or a distinct sub-goal is achieved. Contact established/broken or motion direction reversal is a boundary ONLY if it creates a new stable task state; otherwise keep it inside the same action.
4) For each segment between boundaries, assign exactly one atomic action with the correct verb and patient.
5) Perform an explicit self-check: for every boundary, verify that the frame before and the frame after show a clear task-relevant transition. If the transition is ambiguous or only a micro-phase difference, merge the adjacent actions instead of forcing a boundary.

For each atomic action, predict:
- `atomic_action_id`: sequential 1-based integer
- `start_frame_index`: 1-based inclusive start boundary on the dense {num_frames}-frame pool
- `end_frame_index`: 1-based exclusive end boundary (half-open `[start, end)`)
- `actor`: the specific body part or tool part that is the direct force applicator (use snake_case; prefer specifics like `right_hand`, `left_hand`, `both_hands`, `right_thumb_and_index`, `knife_blade`, `spatula_head`, `pliers_jaw`; avoid vague `hand` or `person`)
- `action`: a concrete verb phrase describing the atomic physical action including the physical mechanism (e.g., "grasp and lift the cup by its handle", "tilt the pitcher to pour liquid into the cup", "place the container onto the mat and release it after it is stable"); avoid vague verbs like "do", "use", "interact with", "handle"
- `patient`: the primary object being acted upon (snake_case, consistent with the step goal/reference annotation; must be exactly ONE entity — if multiple objects, name the primary and mention secondary objects in caption)
- `caption`: one detailed English sentence describing the atomic action, which MUST include: (a) the spatial relationship between actor and patient at the START of this action, (b) the physical motion or force application, (c) the visible state change by the END of this action. Example: "The right hand, positioned above the cup handle, descends to wrap fingers around the ceramic handle, establishing a pinch grip with thumb on top and three fingers curled underneath."

FRAME BOUNDARY CONSTRAINTS (NON-NEGOTIABLE):
- `start_frame_index` of the FIRST atomic action MUST be `1`.
- `end_frame_index` of the LAST atomic action MUST be `{num_frames + 1}` (exclusive boundary after the last frame; ensures full coverage).
- Contiguous: for consecutive atomic actions, `end_i == start_(i+1)` (no gaps, no overlaps).
- Each atomic action MUST satisfy `start_frame_index < end_frame_index`.
- All indices MUST be integers in [1, {num_frames + 1}]; `{num_frames + 1}` is allowed ONLY for the last `end_frame_index`.

BOUNDARY ACCURACY (NON-NEGOTIABLE):
- Place boundaries only at a VISIBLE TASK-RELEVANT STATE CHANGE: new object engagement, ownership/support change, tool-use change, or sub-goal completion. Contact established/broken, motion direction change, release, withdrawal, or minor repositioning is NOT enough by itself.
- For the half-open interval `[start_frame_index, end_frame_index)`, `end_frame_index` should be the FIRST frame index where the NEXT atomic action's defining visible state or motion has clearly begun. The boundary frame belongs to the next action, not the previous one.
- For each boundary, you MUST be able to point to a specific visual difference between the frame before and after that changes the task state, not merely hand pose or continuous motion phase.
- Boundaries must be anchored to the provided camera view; if the transition is occluded, off-screen, or ambiguous in the visible frames, merge rather than split.
- Do NOT place boundaries at arbitrary equal intervals; follow the actual motion dynamics.
- When uncertain between two adjacent frames, choose the first frame where the new action's defining task state is clearly visible; if no such frame is clear, merge rather than split.

DECOMPOSITION GUIDELINES:
- Default to fewer actions. Prefer ONE atomic action when a human would naturally describe the visible segment as one continuous manipulation with the same actor, patient, and step goal.
  - GOOD: one continuous "grasp and lift object" motion can include approach/contact/grip/lift when no stable intermediate task state needs its own clip.
  - GOOD: one continuous "place object on surface" motion can include lower/place/release/hand withdrawal when no separate task-relevant state change occurs.
  - GOOD: split only when both adjacent segments would be meaningful standalone clips with different task-relevant states and a precise visible boundary.
  - BAD: splitting reach/grasp, grasp/lift, lower/place/release, release hand, withdraw hand, brief pauses, or tiny stabilizing adjustments into separate actions when they are just parts of one continuous motion.
  - BAD: "micro-adjust finger angle" (too fine — not visually distinguishable between frames)
- Use only the number of atomic actions that are visually justified; short or simple steps should default to a single atomic action unless a separate task-relevant state change is clearly visible.
- Do NOT create a standalone atomic action shorter than 5 frame-index intervals (about 1 second at 5fps) when the step has multiple atomic actions. Merge such brief fragments into the adjacent action unless the entire step is represented by a single atomic action.
- If a candidate fragment would last less than about 1.0 second in the clip-local timestamps, it is too short to stand alone; merge it with the adjacent action.
- Before finalizing, merge adjacent actions that share the same actor/patient/object and form one continuous manipulation toward the same step goal.
- If two adjacent captions differ only by phase wording (e.g., approach vs contact, release vs withdraw) but not by task-relevant state, merge them.
- Each atomic action should correspond to a visually identifiable phase with a clear start/end in the frames.
- When the same manipulation continues across many frames with NO state change, keep it as ONE atomic action (e.g., sustained carrying/holding).
- Release, hand withdrawal, brief pauses, minor repositioning, small stabilizing adjustments, and very brief reach/grasp/setup fragments should NOT be standalone actions unless they cause a new task-relevant state change.
- Do NOT use release, withdraw, hold steady, pause, minor adjust, stabilize, move hand away, remove hand, or let go as the MAIN standalone action. Include those moments inside the adjacent task action.

CLIP-BOUNDARY EVIDENCE:
- The Stage 3 goal is context, not proof that the goal was completed inside this clip. Upstream screening can truncate an operation at either clip boundary.
- Describe the END state only from the last visible frames of this atomic action. If an object is still held, lifted, or moving when the interval ends, explicitly describe that ongoing state; do not invent later placement, release, stabilization, or goal completion.
- "placed", "resting stably", "released", and "cleaned" require directly visible evidence inside this interval. Lifting or carrying an object toward a destination does not establish that it has reached or settled at the destination. A wiping motion alone does not establish visible dirt removal.
- An atomic action may cover the visible part of an unfinished operation. Do not extend its caption beyond the step clip to make the operation sound complete.

PHYSICAL CONTINUITY:
- The `patient` of atomic action i's end state must be consistent with the `patient`'s start state in atomic action i+1.
- If the actor switches hands or tools and that switch creates a new task-relevant manipulation phase, make it a separate atomic action; otherwise keep the continuous manipulation together.
- The sequence of atomic actions must tell a coherent physical story: a human expert watching the clips should be able to reconstruct the full manipulation from the captions alone.

GROUNDING REQUIREMENTS:
- All text fields MUST be grounded in visual evidence from the frames. Do NOT hallucinate objects, contacts, or states not visible.
- Each action/caption must describe only what is visible within its own `[start_frame_index, end_frame_index)` interval; do NOT describe events before the start, after the end, or inferred outside the camera view.
- Do NOT reference frame indices, timestamps, durations, or timecodes in `action`, `caption`, or `patient` fields. The only allowed frame reference is the integer `start_frame_index` / `end_frame_index` fields.
- Use consistent object naming across all atomic actions (reuse object names from the step goal/reference annotation when visually supported).
- Avoid placeholders like "unknown", "N/A", "the object", "something".

Output format (strict JSON only; no markdown, no comments, no extra text):
{{
  "step_id": <must match the step annotation step_id>,
  "atomic_actions": [
    {{
      "atomic_action_id": 1,
      "start_frame_index": 1,
      "end_frame_index": 9,
      "actor": "right_hand",
      "action": "grasp and lift the cup by its handle",
      "patient": "cup",
      "caption": "The right hand moves from the counter toward the cup handle, curls around the handle, and applies upward force until the cup base separates from the counter."
    }},
    {{
      "atomic_action_id": 2,
      "start_frame_index": 9,
      "end_frame_index": {num_frames + 1},
      "actor": "right_hand",
      "action": "carry the lifted cup away from the counter area",
      "patient": "cup",
      "caption": "The right hand keeps the cup handle gripped and moves the lifted cup away from its original counter position while maintaining control of the cup."
    }}
  ]
}}

Additional constraints:
- `step_id` MUST match the step annotation's step_id exactly.
- `atomic_action_id` MUST start at 1 and increase by 1.
- All required keys MUST be present and non-empty (no empty strings, empty arrays, or null).
- Do NOT add any extra keys beyond the schema above.
- Output MUST be exactly one JSON object.

Now output the final strict JSON object only.
""".strip()


def build_stage2_boundary_verification_prompt(
    high_level_goal: str,
    draft_plan_outline: str,
    num_frames: int,
    current_boundaries_json: str,
) -> str:
    """Build the Stage 2 boundary verification prompt for the second-pass VLM call."""
    return f"""
You are an expert video step temporal boundary verifier.
You are given:
1) {num_frames} dense frame-index boundary points from the FULL original video (chronological order), with frame labels.
2) A draft step plan (read-only).
3) A PROPOSED set of step boundaries that you must VERIFY and CORRECT if needed.

High-level goal: {high_level_goal}

Draft steps:
{draft_plan_outline}

Proposed boundaries (to verify/correct):
{current_boundaries_json}

Task:
For EACH boundary between consecutive steps, examine the frames AT and AROUND the boundary and determine whether the boundary is correctly placed.

{PRIMARY_CENTER_OPERATOR_POLICY}

For each step boundary (where step i ends and step i+1 begins), check:
- The last 2-3 frames of step i's clip: Do they show step i's action completing, or have they already started step i+1's action (reaching for a new object, new grip, new motion direction)?
- The first 2-3 frames of step i+1's clip: Do they show step i+1's action beginning, or is step i's action still ongoing (object still moving, hand still in contact)?

BLEEDING DETECTION (the primary error to catch):
- If the last frames of step i's clip show the agent REACHING TOWARD or ENGAGING WITH the object of step i+1, the boundary is TOO LATE. Move it EARLIER.
- If the first frames of step i+1's clip show the agent still COMPLETING the action of step i (object still in motion, hand still gripping previous object), the boundary is TOO EARLY. Move it LATER. But ONLY if this does not cause next-step bleeding.

BOUNDARY CORRECTNESS CRITERIA:
- CORRECT boundary: The last frame of step i shows step i's action completed (object in final position, hand released or withdrawing, or neutral pose). The first frame of step i+1 shows the beginning of a new action or transition toward it.
- INCORRECT (boundary too late): The last frame of step i shows the agent already reaching toward step i+1's target object. FIX: move boundary earlier.
- INCORRECT (boundary too early): Step i's action is visibly incomplete in its clip (object not yet in final position). FIX: move boundary later, but ONLY if this does not cause next-step bleeding.

ASYMMETRIC CORRECTION RULE (NON-NEGOTIABLE):
- When in doubt, move boundaries EARLIER (trim the end of the current step) rather than LATER.
- It is acceptable to lose 1-2 tail frames of a step, but UNACCEPTABLE to include any next-step action frames.
- If a boundary is ambiguous and you cannot clearly determine the correct position, KEEP IT UNCHANGED.

CONSTRAINTS (same as original localization — do not violate):
- start_frame_index of the first step MUST be 1.
- end_frame_index of the last step MUST be {num_frames + 1}.
- Contiguous: end_i == start_{{i+1}} for consecutive steps. No gaps, no overlaps.
- start_frame_index < end_frame_index for each step.
- All indices are integers in [1, {num_frames + 1}]; {num_frames + 1} only allowed for last end_frame_index.
- Output must contain exactly one entry per step_id.

Output format (strict JSON only; no markdown, no extra text):
{{
  "steps": [
    {{
      "step_id": 1,
      "start_frame_index": 1,
      "end_frame_index": <corrected_or_unchanged>
    }},
    {{
      "step_id": 2,
      "start_frame_index": <must_equal_previous_end>,
      "end_frame_index": <corrected_or_unchanged>
    }}
  ]
}}
""".strip()


SYSTEM_PROMPT_OBJECT_INVENTORY = """
You are an expert object-identity resolution assistant for a video annotation pipeline.
Your output MUST be a single, syntactically flawless JSON object. JSON validity is a critical, non-negotiable requirement.
Return JSON only: no markdown, no comments, no extra text.
Use only the provided recording-level step and action text. Do not invent objects that are not mentioned.
""".strip()


def build_stage4_object_inventory_user_prompt(
    high_level_goal: str,
    steps_context: str,
    mention_list: List[str],
) -> str:
    """Build the recording-level object-identity-resolution user prompt for Stage 4.

    The model is given every step's goal, hand-object relations, and atomic actions
    (already generated text, no video), plus the de-duplicated list of distinct object
    mention strings used anywhere in the recording. It must cluster those mentions into
    a minimal object inventory and map every mention to exactly one inventory entry.
    """
    mentions_json = json.dumps(mention_list, ensure_ascii=False, indent=2)
    return f"""
You are given the full text plan for one recording (all steps, their hand-object relations, and their atomic actions, already extracted from the video), plus the complete list of distinct object-mention strings used anywhere in this recording.

High-level goal: {high_level_goal}

Recording plan (read-only; for context):
{steps_context}

Distinct object mentions used anywhere in this recording (read-only; every one MUST be mapped):
{mentions_json}

Task:
Produce a MINIMAL object inventory for this recording, then map every mention string above to exactly one inventory entry.

Each inventory entry has:
- `entity_ref`: a short, stable, snake_case identifier for this physical object/entity (e.g. "obj_wooden_blocks", "container_left", "container_right"). Choose IDs that will not need to change if the object's state or description changes later.
- `canonical_name`: a plain, standard name for the object (e.g. "wooden block", "plastic storage container").
- `identity_scope`: one of:
  - `instance`: this refers to ONE specific, individually trackable physical object (e.g. "this phone", "this red cup"), consistently the same instance across steps and viewpoints.
  - `group`: this refers to a large collection of same-type items that are not distinguished individually and do not need to be (e.g. "a pile of blocks", "a box of screws"). Treat the whole collection as one entity.
  - `unresolved`: identity truly cannot be determined from the given text. Never guess or fabricate an instance/group classification just to fill this in — use `unresolved` when uncertain.

RULES (NON-NEGOTIABLE):
- If two or more mentions clearly refer to the same type of object but are distinguishable individual instances (e.g. a left container and a right container that are both visible and separately referenced), you MUST create a separate `entity_ref` for each instance, both with `identity_scope: "instance"`. Do NOT collapse distinguishable instances into one entry.
- Do NOT create multiple entries for the same physical object just because its described state changed (e.g. "stacked blocks" vs "scattered blocks" is the SAME entity if it is the same physical thing — merge them into one `entity_ref`, and pick whichever `canonical_name` is clearest).
- Do NOT invent an object that is not present in the mention list.
- Every distinct mention string in the input list MUST be mapped to exactly one `entity_ref`, and every `entity_ref` used in the mapping MUST appear in `object_inventory`.
- `object_inventory` MUST NOT contain duplicate `entity_ref` values.
- Keep the inventory MINIMAL: only create as many entries as are actually needed to correctly capture distinguishable identities.

Output format (strict JSON only; no markdown, no extra text):
{{
  "object_inventory": [
    {{
      "entity_ref": "obj_wooden_blocks",
      "canonical_name": "wooden block",
      "identity_scope": "group"
    }},
    {{
      "entity_ref": "container_left",
      "canonical_name": "plastic storage container",
      "identity_scope": "instance"
    }},
    {{
      "entity_ref": "container_right",
      "canonical_name": "plastic storage container",
      "identity_scope": "instance"
    }}
  ],
  "mentions": {{
    "<mention_string_1>": "<entity_ref>",
    "<mention_string_2>": "<entity_ref>"
  }}
}}

Additional constraints:
- `mentions` MUST contain exactly one key for every string in the "Distinct object mentions" list above, verbatim (same string), mapped to its `entity_ref`.
- Do NOT add any extra keys beyond the schema above.
- Output MUST be exactly one JSON object.

Now output the final strict JSON object only.
""".strip()
