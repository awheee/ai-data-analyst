from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class MemoryContext:
    """
    In-session conversational memory.

    We keep this deliberately small (last N turns) so the prompt stays focused.
    """

    recent_questions: List[str]
    recent_assistant_summaries: List[str]


def get_memory_context(session_state: Dict[str, Any], max_questions: int = 3) -> MemoryContext:
    messages: List[Dict[str, Any]] = session_state.get("messages", [])

    # Build turn pairs (user -> assistant) to keep memory relevant.
    turns: List[Dict[str, str]] = []
    pending_question: str | None = None
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role == "user" and content:
            pending_question = content
        elif role == "assistant" and content and pending_question is not None:
            turns.append({"question": pending_question, "assistant_summary": content})
            pending_question = None

    # If the user asked again without an assistant response, include the last question only.
    if pending_question:
        turns.append({"question": pending_question, "assistant_summary": ""})

    turns = turns[-max_questions:]
    recent_questions = [t["question"] for t in turns if t["question"]]
    recent_assistant_summaries = [t["assistant_summary"] for t in turns if t.get("assistant_summary")]

    return MemoryContext(
        recent_questions=recent_questions,
        recent_assistant_summaries=recent_assistant_summaries,
    )


def append_user_message(session_state: Dict[str, Any], question: str) -> None:
    session_state.setdefault("messages", [])
    session_state["messages"].append({"role": "user", "content": question})


def append_assistant_message(
    session_state: Dict[str, Any],
    content: str,
    artifacts: Dict[str, Any] | None = None,
) -> None:
    session_state.setdefault("messages", [])
    msg: Dict[str, Any] = {"role": "assistant", "content": content}
    if artifacts:
        msg["artifacts"] = artifacts
    session_state["messages"].append(msg)

