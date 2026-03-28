from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Dict, List, Optional

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"


@dataclass
class LLMResponse:
    template_spec: Dict[str, Any]
    # Optional direct answer for text-qa mode.
    answer_text: Optional[str] = None


_TEMPLATES = [
    "data_summary",
    "select_columns",
    "describe",
    "filter",
    "group_aggregate",
    "top_n",
    "time_series",
    "correlation_scatter",
    "none",
]

_CHART_TYPES = ["bar", "line", "scatter", "pie", "none"]


def _build_prompt(
    *,
    question: str,
    dataset_summary: Dict[str, Any],
    dataset_schema: List[Dict[str, str]],
    recent_questions: List[str],
    recent_assistant_summaries: List[str],
) -> List[Dict[str, str]]:
    schema_lines = []
    for s in dataset_schema:
        schema_lines.append(f"- {s['name']} : {s['type']}")
    schema_text = "\n".join(schema_lines) if schema_lines else "(no structured schema)"

    recent_text = ""
    if recent_questions:
        recent_text = "Recent user questions:\n" + "\n".join([f"- {q}" for q in recent_questions])
    if recent_assistant_summaries:
        recent_text += (
            "\n\nRecent assistant replies (for follow-ups—use exact column names from schema):\n"
            + "\n".join([f"- {(s or '')[:600]}" for s in recent_assistant_summaries])
        )

    system = (
        "You are an AI data analyst. You will receive a dataset summary and a schema (column names + rough types). "
        "Your task is NOT to write code. Instead, you must produce a SAFE JSON template spec that a deterministic "
        "executor will run using pandas.\n\n"
        "Rules:\n"
        "- Use ONLY column names that exist in the provided schema.\n"
        "- Follow the user's intent. If the question is a follow-up (e.g., 'only for 2023', 'now only ...'), "
        "add/modify filters accordingly.\n"
        "- If the user asks for a **dataset overview / summary / what columns / what's in this file / describe the data**, "
        "use template **data_summary** (accurate stats are computed in code—do not invent numbers).\n"
        "- If the user asks to **show / extract / print / list** one or more **specific columns** (by name), "
        "use template **select_columns** with parameters.columns as a list of exact schema column names.\n"
        "- If the question needs something you cannot express with the allowed templates "
        "(e.g. machine learning, predictions, joins across separate tables, SQL, web search, running arbitrary code, "
        "reading files outside this dataset), use template **none** with parameters.reason explaining briefly why.\n"
        "- When using **none**, start parameters.reason with **I can't answer this:** and be explicit.\n"
        "- Choose exactly one template from: " + ", ".join(_TEMPLATES) + ".\n"
        "- Set chart_type to one of: " + ", ".join(_CHART_TYPES) + ". If uncertain, use 'none'.\n"
        "- Return ONLY valid JSON (no markdown, no extra keys).\n"
    )

    human = (
        f"{recent_text}\n\n"
        f"Dataset summary:\n{json.dumps(dataset_summary, ensure_ascii=False)}\n\n"
        f"Dataset schema:\n{schema_text}\n\n"
        f"User question:\n{question}\n\n"
        "Return JSON in this format:\n"
        "{\n"
        '  "mode": "dataframe_analysis",\n'
        '  "template": "data_summary" | "select_columns" | "describe" | "filter" | "group_aggregate" | "top_n" | "time_series" | "correlation_scatter" | "none",\n'
        '  "parameters": { ... },\n'
        '  "chart_type": "bar" | "line" | "scatter" | "pie" | "none",\n'
        '  "chart_hint": { ... }\n'
        "}\n\n"
        "Template parameters (examples):\n"
        '- data_summary: {"include_sample": true}\n'
        '- select_columns: {"columns":["col_a","col_b"], "limit": 5000}\n'
        '- describe: {"columns":"all"}\n'
        '- filter: {"conditions":[{"column":"col","op":"eq|neq|contains|gt|gte|lt|lte","value":"..."}], "limit": 2000}\n'
        '- group_aggregate: {"group_by":["col1"], "agg":[{"column":"col2","fn":"mean|sum|count|min|max"}]}\n'
        '- top_n: {"sort_by":"col","n":10}\n'
        '- time_series: {"date_column":"date_col","value_column":"val_col","bucket":"day|month|year","agg_fn":"mean|sum|count"}\n'
        '- correlation_scatter: {"x_column":"x","y_column":"y"}\n'
        '- none: {"reason":"I can\'t answer this: (one short sentence)"}\n\n'
        "chart_hint fields (optional, but keep simple):\n"
        '- bar: {"x":"group_col","y":"agg_value_col"}\n'
        '- line: {"x":"date_bucket","y":"agg_value_col"}\n'
        '- scatter: {"x":"x_column","y":"y_column"}\n'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": human}]


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    # Common failure mode: model wraps json in fences.
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)


def _chat_completion(client: Any, *, model: str, messages: List[Dict[str, str]], use_json_object: bool) -> str:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }
    if use_json_object:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


def generate_template_spec(
    *,
    question: str,
    dataset_summary: Dict[str, Any],
    dataset_schema: List[Dict[str, str]],
    recent_questions: List[str],
    recent_assistant_summaries: List[str],
    api_key: str,
    model: Optional[str] = None,
) -> LLMResponse:
    """
    Generate a SAFE JSON template spec for deterministic execution (Groq via OpenAI-compatible API).
    """
    from openai import OpenAI

    resolved_model = model or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    messages = _build_prompt(
        question=question,
        dataset_summary=dataset_summary,
        dataset_schema=dataset_schema,
        recent_questions=recent_questions,
        recent_assistant_summaries=recent_assistant_summaries,
    )

    # Two parse rounds; each round tries JSON mode first, then plain completion if Groq rejects it.
    for _parse_round in range(2):
        content: Optional[str] = None
        last_api_err: Optional[BaseException] = None
        for use_json in (True, False):
            try:
                content = _chat_completion(
                    client,
                    model=resolved_model,
                    messages=messages,
                    use_json_object=use_json,
                )
                break
            except Exception as e:
                last_api_err = e
                continue
        if content is None and last_api_err is not None:
            raise last_api_err

        try:
            spec = _extract_json(content or "")
            if not isinstance(spec, dict) or "template" not in spec:
                raise ValueError("Missing template key")
            if spec.get("template") not in _TEMPLATES:
                raise ValueError("Template not in allow-list")
            if spec.get("chart_type") not in _CHART_TYPES:
                raise ValueError("chart_type not in allow-list")
            if "parameters" not in spec or not isinstance(spec.get("parameters"), dict):
                raise ValueError("parameters must be an object")
            if spec.get("mode") not in {"dataframe_analysis"}:
                spec["mode"] = "dataframe_analysis"

            return LLMResponse(template_spec=spec)
        except Exception:
            messages = [
                {"role": "system", "content": messages[0]["content"]},
                {
                    "role": "user",
                    "content": messages[1]["content"]
                    + "\n\nIMPORTANT: Return ONLY valid JSON. No code fences, no markdown, no explanations.",
                },
            ]

    raise ValueError("Failed to parse LLM JSON template spec.")
