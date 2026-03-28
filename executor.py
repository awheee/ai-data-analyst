from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, List

import pandas as pd

from data_handler import build_accurate_column_profile, build_dataset_narrative


ChartType = Literal["bar", "line", "scatter", "pie", "none"]

_OUT_OF_DOMAIN_FALLBACK = (
    "I can't answer this with the tools available in this app. "
    "Supported tasks: **dataset summary** (profile + narrative + sample rows), **show/extract columns**, "
    "**filtering**, **group-by aggregates**, **top-N rows**, **simple time buckets**, **correlation/scatter**, "
    "and **column statistics** (`describe`). "
    "I cannot do joins across tables, ML/predictions, custom code, external APIs, or SQL."
)


@dataclass
class ExecutionResult:
    result_df: Optional[pd.DataFrame]
    chart_type: ChartType
    chart_data: Dict[str, Any]
    caption: str
    answer_text: str
    secondary_df: Optional[pd.DataFrame] = None


def execute_template(template_spec: Dict[str, Any], df: Optional[pd.DataFrame], dataset_text: Optional[str] = None) -> ExecutionResult:
    """
    Execute a safe, deterministic template.

    The LLM provides a JSON "template spec". We validate it against an allow-list and
    then run deterministic pandas logic (no exec/eval).
    """
    try:
        if not isinstance(template_spec, dict):
            raise ValueError("LLM returned invalid template spec (expected object).")

        mode = template_spec.get("mode", "dataframe_analysis")
        template = template_spec.get("template", "none")
        chart_type: ChartType = template_spec.get("chart_type", "none")
        chart_hint = template_spec.get("chart_hint") or {}

        if df is None:
            # Best-effort documents: without a DataFrame, we currently can't compute tables/charts.
            if mode == "dataframe_analysis":
                return ExecutionResult(
                    result_df=None,
                    chart_type="none",
                    chart_data={},
                    caption="No structured table available.",
                    answer_text="No table could be extracted from this document (or no tabular data was found). Please upload a CSV/XLSX or a PDF/DOCX with an actual table.",
                )

            return ExecutionResult(
                result_df=None,
                chart_type="none",
                chart_data={},
                caption="Text-only answers not implemented in this MVP.",
                answer_text="This document contains unstructured text. Table/chart extraction is not available for this file type in the MVP.",
            )

        if template not in {
            "data_summary",
            "select_columns",
            "describe",
            "filter",
            "group_aggregate",
            "top_n",
            "time_series",
            "correlation_scatter",
            "none",
        }:
            return ExecutionResult(
                result_df=None,
                chart_type="none",
                chart_data={},
                caption="Unsupported template.",
                answer_text=f"Unsupported operation requested: {template!r}",
            )

        if template == "none":
            params = template_spec.get("parameters") or {}
            reason = (params.get("reason") or "").strip()
            if not reason:
                reason = _OUT_OF_DOMAIN_FALLBACK
            elif not reason.lower().startswith("i can't answer"):
                reason = "I can't answer this: " + reason
            return ExecutionResult(
                result_df=None,
                chart_type="none",
                chart_data={},
                caption=reason,
                answer_text=reason,
            )

        allowed_chart_types: set[str] = {"bar", "line", "scatter", "pie", "none"}
        if chart_type not in allowed_chart_types:
            chart_type = "none"

        # Helpers
        def _col_exists(col: str) -> bool:
            return str(col) in set(df.columns)

        def _bucket(series: pd.Series) -> str:
            if pd.api.types.is_numeric_dtype(series):
                return "number"
            if pd.api.types.is_datetime64_any_dtype(series):
                return "datetime"
            return "text"

        # Template implementations
        if template == "data_summary":
            params = template_spec.get("parameters") or {}
            include_sample = params.get("include_sample", True)
            if not isinstance(include_sample, bool):
                include_sample = bool(include_sample)

            profile = build_accurate_column_profile(df)
            n, p = df.shape
            narrative = build_dataset_narrative(profile, n, p)
            intro = (
                "### Dataset overview\n\n"
                f"{narrative}\n\n"
                f"The table below is an **accurate column profile** (counts, null %, unique values—computed in code, not guessed)."
            )
            sample = df.head(5).reset_index(drop=True) if include_sample else None
            if sample is not None:
                intro += "\n\nSample rows (first 5) are shown under the profile."

            return ExecutionResult(
                result_df=profile,
                chart_type="none",
                chart_data={},
                caption="Accurate dataset summary",
                answer_text=intro,
                secondary_df=sample,
            )

        if template == "select_columns":
            params = template_spec.get("parameters") or {}
            cols = params.get("columns") or []
            limit = int(params.get("limit", 5000))
            limit = max(1, min(limit, 50_000))
            if not isinstance(cols, list) or not cols:
                raise ValueError("select_columns requires parameters.columns as a non-empty list of column names.")
            valid = [str(c) for c in cols if isinstance(c, str) and _col_exists(c)]
            if not valid:
                raise ValueError("None of the requested column names exist in the dataset. Check spelling against the schema.")
            out = df[valid].head(limit).copy()
            out.insert(0, "_row", range(len(out)))
            n_avail = len(df)
            truncated = n_avail > limit
            msg = f"Showing **{len(out)}** row(s) for: {', '.join('`'+c+'`' for c in valid)}."
            if truncated:
                msg += f" (Dataset has {n_avail:,} rows; increase `limit` in the tool up to 50,000 to see more.)"
            return ExecutionResult(
                result_df=out,
                chart_type="none",
                chart_data={},
                caption="Selected columns",
                answer_text=msg,
            )

        if template == "describe":
            params = template_spec.get("parameters") or {}
            columns = params.get("columns", "all")
            if columns == "all" or columns is None:
                result_df = df.describe(include="all").transpose().reset_index().rename(columns={"index": "column"})
            else:
                if isinstance(columns, list):
                    cols = [c for c in columns if _col_exists(c)]
                    if not cols:
                        raise ValueError("No valid columns found for describe.")
                    result_df = df[cols].describe(include="all").transpose().reset_index().rename(columns={"index": "column"})
                else:
                    raise ValueError("Invalid 'columns' parameter for describe.")

            return ExecutionResult(
                result_df=result_df,
                chart_type="none",
                chart_data={},
                caption="Dataset overview (summary statistics).",
                answer_text="Dataset overview (summary statistics) is shown below.",
            )

        if template == "filter":
            params = template_spec.get("parameters") or {}
            conditions = params.get("conditions") or []
            limit = params.get("limit", 2000)
            if not isinstance(conditions, list) or not conditions:
                raise ValueError("Filter template requires a non-empty 'conditions' list.")

            out = df.copy()
            for cond in conditions:
                if not isinstance(cond, dict):
                    continue
                col = cond.get("column")
                op = (cond.get("op") or "").lower()
                value = cond.get("value")
                if not isinstance(col, str) or not _col_exists(col):
                    raise ValueError(f"Unknown column in filter: {col!r}")

                s = out[col]
                if op == "eq":
                    out = out[out[col] == value]
                elif op == "neq":
                    out = out[out[col] != value]
                elif op == "contains":
                    out = out[out[col].astype(str).str.contains(str(value), case=False, na=False)]
                elif op in {"gt", "gte", "lt", "lte"}:
                    if _bucket(s) != "number":
                        # Attempt numeric coercion.
                        out[col] = pd.to_numeric(out[col], errors="coerce")
                        s = out[col]
                    try:
                        num = float(value)
                    except Exception:
                        num = pd.to_numeric(value, errors="coerce")
                        if pd.isna(num):
                            raise ValueError(f"Filter value for {col!r} is not a valid number: {value!r}")
                    if op == "gt":
                        out = out[out[col] > num]
                    elif op == "gte":
                        out = out[out[col] >= num]
                    elif op == "lt":
                        out = out[out[col] < num]
                    elif op == "lte":
                        out = out[out[col] <= num]
                else:
                    raise ValueError(f"Unsupported filter operator: {op!r}")

            if out.empty:
                return ExecutionResult(
                    result_df=out.reset_index(drop=True),
                    chart_type="none",
                    chart_data={},
                    caption="No rows match your filter conditions.",
                    answer_text="No rows match your filter conditions. Try a different filter.",
                )

            if isinstance(limit, int) and limit > 0:
                result_df = out.head(limit).reset_index(drop=True)
            else:
                result_df = out.reset_index(drop=True)

            n_rows = result_df.shape[0]
            return ExecutionResult(
                result_df=result_df,
                chart_type="none",
                chart_data={},
                caption=f"Filtered results: {n_rows} rows.",
                answer_text=f"Filtered results: {n_rows} rows (showing up to {limit} rows).",
            )

        if template == "group_aggregate":
            params = template_spec.get("parameters") or {}
            group_by = params.get("group_by") or []
            agg = params.get("agg") or []

            if not isinstance(group_by, list) or len(group_by) < 1:
                raise ValueError("group_aggregate requires 'group_by' as a non-empty list.")
            if not isinstance(agg, list) or len(agg) < 1:
                raise ValueError("group_aggregate requires 'agg' as a non-empty list.")

            missing_cols = [c for c in group_by if not isinstance(c, str) or not _col_exists(c)]
            if missing_cols:
                raise ValueError(f"Unknown group_by columns: {missing_cols!r}")

            group_df = df.copy()
            # Build aggregations
            agg_pieces: List[tuple[str, pd.Series]] = []
            out_cols: List[str] = []
            result_df = None

            # Coerce numeric columns early when needed (safer for groupby results).
            for item in agg:
                if not isinstance(item, dict):
                    continue
                col = item.get("column")
                fn = (item.get("fn") or "").lower()
                if isinstance(col, str) and _col_exists(col) and fn in {"sum", "mean", "min", "max"}:
                    group_df[col] = pd.to_numeric(group_df[col], errors="coerce")

            grouped = group_df.groupby(group_by, dropna=False)

            agg_results = []
            for item in agg:
                if not isinstance(item, dict):
                    continue
                col = item.get("column")
                fn = (item.get("fn") or "").lower()
                if not isinstance(col, str) or not _col_exists(col):
                    raise ValueError(f"Unknown agg column: {col!r}")

                out_name = f"{fn}_{col}"
                if fn == "count":
                    agg_res = grouped[col].count().rename(out_name)
                else:
                    # Coerce to numeric if possible.
                    if fn in {"sum", "mean", "min", "max"}:
                        group_df[col] = pd.to_numeric(group_df[col], errors="coerce")
                    if fn == "sum":
                        agg_res = grouped[col].sum().rename(out_name)
                    elif fn == "mean":
                        agg_res = grouped[col].mean().rename(out_name)
                    elif fn == "min":
                        agg_res = grouped[col].min().rename(out_name)
                    elif fn == "max":
                        agg_res = grouped[col].max().rename(out_name)
                    else:
                        raise ValueError(f"Unsupported agg_fn: {fn!r}")
                agg_results.append(agg_res)
                out_cols.append(out_name)

            if not agg_results:
                raise ValueError("No valid aggregation specs found.")

            result_df = pd.concat(agg_results, axis=1).reset_index()

            if result_df.empty:
                return ExecutionResult(
                    result_df=result_df,
                    chart_type="none",
                    chart_data={},
                    caption="No aggregated results (check your filter/columns).",
                    answer_text="No aggregated results (check the chosen columns and data).",
                )

            # Decide chart type if none specified.
            inferred_chart = chart_type
            if inferred_chart == "none" and group_by:
                if len(group_by) == 1 and len(out_cols) == 1:
                    x_col = group_by[0]
                    y_col = out_cols[0]
                    inferred_chart = "line" if _bucket(result_df[x_col]) == "datetime" else "bar"
                    chart_type = inferred_chart  # align
                    chart_hint = {"x": x_col, "y": y_col}

            # If chart_type is requested but hint is missing, infer x/y.
            x = chart_hint.get("x")
            y = chart_hint.get("y")
            if chart_type in {"bar", "line", "scatter"} and (not x or not y):
                if len(group_by) >= 1 and len(out_cols) >= 1:
                    x = group_by[0]
                    y = out_cols[0]

            chart_payload = {}
            if chart_type in {"bar", "line"} and isinstance(x, str) and isinstance(y, str) and x in result_df.columns and y in result_df.columns:
                chart_payload = {
                    "x": x,
                    "y": y,
                    "data": result_df[[x, y]].dropna().to_dict(orient="records"),
                }
            elif chart_type == "scatter" and isinstance(x, str) and isinstance(y, str) and x in result_df.columns and y in result_df.columns:
                chart_payload = {
                    "x": x,
                    "y": y,
                    "data": result_df[[x, y]].dropna().to_dict(orient="records"),
                }

            caption = f"Aggregated results ({len(result_df)} rows)."
            return ExecutionResult(
                result_df=result_df,
                chart_type=chart_type,
                chart_data=chart_payload,
                caption=caption,
                answer_text=caption,
            )

        if template == "top_n":
            params = template_spec.get("parameters") or {}
            sort_by = params.get("sort_by")
            n = int(params.get("n", 10))
            ascending = bool(params.get("ascending", False)) if "ascending" in params else False
            if not isinstance(sort_by, str) or not _col_exists(sort_by):
                raise ValueError(f"Unknown sort_by column: {sort_by!r}")
            if n <= 0:
                raise ValueError("n must be > 0 for top_n.")

            s = pd.to_numeric(df[sort_by], errors="coerce")
            tmp = df.copy()
            tmp[sort_by] = s
            result_df = tmp.sort_values(sort_by, ascending=ascending).head(n).reset_index(drop=True)

            caption = f"Top {n} rows by `{sort_by}`."
            return ExecutionResult(
                result_df=result_df,
                chart_type="none",
                chart_data={},
                caption=caption,
                answer_text=caption,
            )

        if template == "time_series":
            params = template_spec.get("parameters") or {}
            date_col = params.get("date_column")
            value_col = params.get("value_column")
            bucket = (params.get("bucket") or "month").lower()
            agg_fn = (params.get("agg_fn") or "mean").lower()
            if not isinstance(date_col, str) or not _col_exists(date_col):
                raise ValueError(f"Unknown date_column: {date_col!r}")
            if not isinstance(value_col, str) or not _col_exists(value_col):
                raise ValueError(f"Unknown value_column: {value_col!r}")

            tmp = df.copy()
            tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
            tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
            tmp = tmp.dropna(subset=[date_col])
            if tmp.empty:
                return ExecutionResult(
                    result_df=None,
                    chart_type="none",
                    chart_data={},
                    caption="No time-series data found (date parsing produced no valid timestamps).",
                    answer_text="No time-series data found. The date column couldn't be parsed as dates, or there are no valid rows.",
                )

            if bucket == "day":
                tmp["date_bucket"] = tmp[date_col].dt.date.astype(str)
            elif bucket == "month":
                tmp["date_bucket"] = tmp[date_col].dt.to_period("M").astype(str)
            elif bucket == "year":
                tmp["date_bucket"] = tmp[date_col].dt.to_period("Y").astype(str)
            else:
                raise ValueError(f"Unsupported bucket: {bucket!r}")

            grouped = tmp.groupby("date_bucket", dropna=False)
            if agg_fn == "count":
                agg_series = grouped[value_col].count()
            elif agg_fn == "sum":
                agg_series = grouped[value_col].sum()
            elif agg_fn == "mean":
                agg_series = grouped[value_col].mean()
            else:
                raise ValueError(f"Unsupported agg_fn for time_series: {agg_fn!r}")

            y_col = f"{agg_fn}_{value_col}"
            result_df = agg_series.rename(y_col).reset_index()
            if result_df.empty:
                return ExecutionResult(
                    result_df=result_df,
                    chart_type="none",
                    chart_data={},
                    caption="No time-series results.",
                    answer_text="No time-series results found for the chosen date/value columns.",
                )
            chart_payload = {
                "x": "date_bucket",
                "y": y_col,
                "data": result_df[["date_bucket", y_col]].dropna().to_dict(orient="records"),
            }
            caption = f"Time series ({bucket}) computed."
            return ExecutionResult(
                result_df=result_df,
                chart_type="line",
                chart_data=chart_payload,
                caption=caption,
                answer_text=caption,
            )

        if template == "correlation_scatter":
            params = template_spec.get("parameters") or {}
            x_col = params.get("x_column")
            y_col = params.get("y_column")
            if not isinstance(x_col, str) or not _col_exists(x_col):
                raise ValueError(f"Unknown x_column: {x_col!r}")
            if not isinstance(y_col, str) or not _col_exists(y_col):
                raise ValueError(f"Unknown y_column: {y_col!r}")

            tmp = df[[x_col, y_col]].copy()
            tmp[x_col] = pd.to_numeric(tmp[x_col], errors="coerce")
            tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce")
            tmp = tmp.dropna(subset=[x_col, y_col])
            if tmp.empty:
                return ExecutionResult(
                    result_df=None,
                    chart_type="none",
                    chart_data={},
                    caption="Not enough numeric data for correlation/plot.",
                    answer_text="Not enough numeric data to plot correlation. Check that both columns contain numbers.",
                )
            if tmp.shape[0] > 5000:
                tmp = tmp.sample(5000, random_state=0)

            corr = tmp[x_col].corr(tmp[y_col])
            scatter_df = tmp.reset_index(drop=True)
            caption = f"Scatter plot of `{x_col}` vs `{y_col}` (corr={corr:.3f})." if pd.notna(corr) else f"Scatter plot of `{x_col}` vs `{y_col}`."
            chart_payload = {
                "x": x_col,
                "y": y_col,
                "data": scatter_df[[x_col, y_col]].to_dict(orient="records"),
            }
            return ExecutionResult(
                result_df=scatter_df,
                chart_type="scatter",
                chart_data=chart_payload,
                caption=caption,
                answer_text=caption,
            )

        # Fallback (shouldn't happen)
        return ExecutionResult(
            result_df=None,
            chart_type="none",
            chart_data={},
            caption="Couldn't execute the request.",
            answer_text="Couldn't execute the request.",
        )
    except Exception as e:
        return ExecutionResult(
            result_df=None,
            chart_type="none",
            chart_data={},
            caption="Execution error.",
            answer_text=f"Could not complete the analysis: {e}",
        )

