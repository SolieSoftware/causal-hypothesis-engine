"""DatasetResult — result of a DatasetBuilder run."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from .._time import utcnow


class DatasetResult(BaseModel):
    """Captures the outcome of a DatasetBuilder.run() call.

    Persisted to SQLite (``dataset_results``) and keyed by ``version_id``, so
    the question "which data belongs to this hypothesis?" is answerable. It
    used to exist only as an in-memory object beside an orphan Parquet file.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=utcnow)
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
