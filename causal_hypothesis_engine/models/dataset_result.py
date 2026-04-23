"""DatasetResult — result of a DatasetBuilder run."""

from __future__ import annotations

from pydantic import BaseModel


class DatasetResult(BaseModel):
    """Captures the outcome of a DatasetBuilder.run() call.

    Stored to disk as JSON alongside the Parquet file.  Not persisted to
    SQLite in v1 — DB attachment is deferred until BacktestAgent integration.
    """

    version_id: str
    manifest_path: str
    columns: list[str]       # one per manifest node label, in manifest order
    start_date: str          # ISO 8601 — actual realised start
    end_date: str            # ISO 8601 — actual realised end
    frequency: str           # "weekly" | "daily" | "monthly"
    adf_results: dict[str, dict]
    # adf_results[label] = {
    #     "statistic": float,
    #     "pvalue": float,
    #     "passed": bool,            # pvalue <= 0.05
    #     "transform_applied": str   # e.g. "diff", "diff+diff", "log_return+diff",
    #                                #       "diff" (for none+diff), "none",
    #                                #       "insufficient_data"
    # }
    warnings: list[str]
    output_path: str         # absolute path to the written Parquet file
