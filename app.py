from __future__ import annotations

import os
from typing import Any, Dict

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
import plotly.express as px

from data_handler import LoadedDataset, load_dataset
from executor import ExecutionResult, execute_template
from llm_engine import generate_template_spec
from memory import append_assistant_message, append_user_message, get_memory_context


st.set_page_config(page_title="AI Data Analyst", layout="wide")
st.title("AI Data Analyst")

MAX_ARTIFACT_ROWS = 5000


def _get_groq_api_key() -> str | None:
    """Groq key from Streamlit secrets (Cloud/local) or GROQ_API_KEY env var."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return str(st.secrets["GROQ_API_KEY"]).strip() or None
    except StreamlitSecretNotFoundError:
        pass
    return (os.getenv("GROQ_API_KEY") or "").strip() or None


def _format_llm_error(exc: BaseException) -> str:
    """User-friendly messages for Groq / OpenAI client errors."""
    try:
        from openai import APIStatusError, RateLimitError

        if isinstance(exc, RateLimitError):
            return (
                "Groq rate limit or quota exceeded. Wait a minute and try again, "
                "or check your plan at https://console.groq.com"
            )
        if isinstance(exc, APIStatusError):
            code = getattr(exc, "status_code", None)
            body = getattr(exc, "body", None) or str(exc)
            if code == 401:
                return "Invalid Groq API key. Check `GROQ_API_KEY` in secrets or your environment."
            if code == 429:
                return (
                    "Too many requests or quota exceeded on Groq. Try again later or check "
                    "https://console.groq.com"
                )
            return f"Groq API error ({code}): {body}"
    except Exception:
        pass
    return str(exc)


def _render_chart(chart_type: str, chart_data: Dict[str, Any]) -> None:
    if chart_type == "none" or not chart_data:
        return

    if chart_type == "bar":
        x = chart_data.get("x")
        y = chart_data.get("y")
        data = chart_data.get("data")
        if x and y and isinstance(data, list):
            df = pd.DataFrame(data)
            fig = px.bar(df, x=x, y=y)
            st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "line":
        x = chart_data.get("x")
        y = chart_data.get("y")
        data = chart_data.get("data")
        if x and y and isinstance(data, list):
            df = pd.DataFrame(data)
            fig = px.line(df, x=x, y=y)
            st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "scatter":
        x = chart_data.get("x")
        y = chart_data.get("y")
        data = chart_data.get("data")
        if x and y and isinstance(data, list):
            df = pd.DataFrame(data)
            fig = px.scatter(df, x=x, y=y)
            st.plotly_chart(fig, use_container_width=True)


def _artifacts_from_execution(er: ExecutionResult) -> Dict[str, Any] | None:
    """Serialize tables/chart for session replay (caps row count)."""
    has_table = er.result_df is not None and not er.result_df.empty
    has_secondary = er.secondary_df is not None and not er.secondary_df.empty
    has_chart = bool(er.chart_data) and er.chart_type != "none"
    if not has_table and not has_secondary and not has_chart:
        return None

    art: Dict[str, Any] = {
        "chart_type": er.chart_type,
        "chart_data": dict(er.chart_data) if er.chart_data else {},
    }
    if has_table:
        n = len(er.result_df)  # type: ignore
        take = er.result_df.head(MAX_ARTIFACT_ROWS)  # type: ignore
        art["result"] = take.to_dict(orient="records")
        art["result_truncated"] = n > MAX_ARTIFACT_ROWS
        art["result_total_rows"] = n
    else:
        art["result"] = None
    if has_secondary:
        n2 = len(er.secondary_df)  # type: ignore
        take2 = er.secondary_df.head(MAX_ARTIFACT_ROWS)  # type: ignore
        art["secondary"] = take2.to_dict(orient="records")
        art["secondary_truncated"] = n2 > MAX_ARTIFACT_ROWS
    else:
        art["secondary"] = None
    return art


def _render_assistant_artifacts(artifacts: Dict[str, Any]) -> None:
    if artifacts.get("result"):
        st.dataframe(pd.DataFrame(artifacts["result"]), use_container_width=True)
        if artifacts.get("result_truncated"):
            total = artifacts.get("result_total_rows", "?")
            st.caption(f"Showing first {len(artifacts['result'])} of {total} rows (session limit {MAX_ARTIFACT_ROWS}).")
    if artifacts.get("secondary"):
        st.subheader("Sample rows")
        st.dataframe(pd.DataFrame(artifacts["secondary"]), use_container_width=True)
        if artifacts.get("secondary_truncated"):
            st.caption(f"Sample truncated; max {MAX_ARTIFACT_ROWS} rows stored in chat history.")
    _render_chart(artifacts.get("chart_type", "none"), artifacts.get("chart_data") or {})


# Session state initialization
st.session_state.setdefault("messages", [])
st.session_state.setdefault("dataset", None)  # LoadedDataset


with st.sidebar:
    st.subheader("Upload dataset")
    uploaded_file = st.file_uploader(
        "Upload CSV/XLSX/PDF/DOCX",
        type=["csv", "xlsx", "pdf", "docx"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        try:
            dataset: LoadedDataset = load_dataset(uploaded_file)
            st.session_state["dataset"] = dataset
            st.session_state["messages"] = []

            st.success("Dataset loaded.")
            st.write("Mode:", dataset.mode)
            with st.expander("Quick facts (from file)", expanded=False):
                st.json(dataset.summary)
            st.caption(
                "Ask in chat for a full **dataset summary** or to **show specific columns**. "
                "Uploading a **new file clears** the conversation."
            )
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")


dataset = st.session_state.get("dataset")
if dataset is None:
    st.info("Upload a dataset to start chatting.")
    st.stop()


for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg.get("content", ""))
        else:
            st.markdown(msg.get("content", ""))
            art = msg.get("artifacts")
            if isinstance(art, dict):
                _render_assistant_artifacts(art)


user_question = st.chat_input("Ask a question about your data...")
if user_question:
    append_user_message(st.session_state, user_question)
    with st.chat_message("user"):
        st.write(user_question)

    mem = get_memory_context(st.session_state, max_questions=5)
    recent_questions = mem.recent_questions
    recent_assistant_summaries = mem.recent_assistant_summaries

    api_key = _get_groq_api_key()
    if not api_key:
        with st.chat_message("assistant"):
            st.error(
                "Missing **GROQ_API_KEY**. Add it to `.streamlit/secrets.toml` locally or "
                "Streamlit Cloud secrets, or run: `export GROQ_API_KEY='your_key'`"
            )
            append_assistant_message(st.session_state, "Missing GROQ_API_KEY.")
        st.stop()

    with st.chat_message("assistant"):
        try:
            prompt_ctx = {
                "dataset_summary": dataset.summary,
                "dataset_schema": dataset.schema,
                "recent_questions": recent_questions,
                "recent_assistant_summaries": recent_assistant_summaries,
            }

            llm_resp = generate_template_spec(
                question=user_question,
                dataset_summary=prompt_ctx["dataset_summary"],
                dataset_schema=prompt_ctx["dataset_schema"],
                recent_questions=prompt_ctx["recent_questions"],
                recent_assistant_summaries=prompt_ctx["recent_assistant_summaries"],
                api_key=api_key,
            )

            execution_result = execute_template(
                llm_resp.template_spec,
                df=dataset.df,
                dataset_text=dataset.extracted_text,
            )

            final_text = execution_result.answer_text or "Done."
            if execution_result.answer_text:
                st.markdown(execution_result.answer_text)

            artifacts = _artifacts_from_execution(execution_result)
            if artifacts:
                _render_assistant_artifacts(artifacts)

            append_assistant_message(st.session_state, final_text, artifacts=artifacts)

        except ValueError as e:
            st.error(f"Request error: {e}")
            append_assistant_message(st.session_state, f"Request error: {e}")
        except Exception as e:
            msg = _format_llm_error(e)
            st.error(msg)
            append_assistant_message(st.session_state, msg)
