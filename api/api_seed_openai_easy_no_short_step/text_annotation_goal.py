"""Compose a video goal by copying each chronological subtask's first sentence."""
from __future__ import annotations
import re
from typing import Iterable

NO_SUBTASK_GOAL = "No valid subtask was produced for this video."


def first_subtask_sentence(subtask: str) -> str:
    if not isinstance(subtask, str) or not subtask.strip():
        raise ValueError("Every subtask must have a nonempty action description.")
    # Exported tasks use a semicolon to separate the goal from hand evidence.
    action = re.split(r"[;；]", subtask, maxsplit=1)[0].strip()
    if not action:
        raise ValueError("Subtask has no action before its hand evidence.")
    for match in re.finditer(r"[。！？]|[.!?](?=\s|$)", action):
        # A decimal point has no following whitespace; common abbreviations
        # and dotted initials are also part of the sentence, not its endpoint.
        token = action[:match.start()+1].rsplit(None, 1)[-1]
        if match.group() == "." and (
            token.lower() in {"mr.", "mrs.", "ms.", "dr.", "st.", "no.", "e.g.", "i.e."}
            or re.fullmatch(r"(?:[A-Za-z]\.)+", token)
        ):
            continue
        action = action[:match.end()]
        break
    sentence = " ".join(action.split())
    if not sentence.endswith((".", "!", "?", "。", "！", "？")):
        sentence += "。" if re.search(r"[\u4e00-\u9fff]", sentence) else "."
    return sentence


def compose_video_goal(subtasks: Iterable[str]) -> str:
    # Preserve order and repeated actions; every segment contributes a sentence.
    return " ".join(first_subtask_sentence(task) for task in subtasks)
