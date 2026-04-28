"""DatasetBuilder — deterministic pipeline from manifest to feature-matrix Parquet.

Pipeline per run:
  1. Parse YAML manifest (via FinancialDataAdapter._load_manifest_data).
  2. Fetch + resample + apply declared transforms for each node.
  3. Check label alignment with the DAGVersion.
  4. Guard short series (< 20 observations) before ADF.
  5. Run ADF; if non-stationary, apply one extra diff and re-test.
  6. Compile DatasetResult with adf_results and accumulated warnings.
  7. Write Parquet atomically: write to .tmp → fsync → os.rename.

This component makes no LLM calls.  Named DatasetBuilder (not an Agent)
to clearly distinguish it from DAGAgent, ModificationAgent, BacktestAgent.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from .adapters.financial import FinancialDataAdapter
from .models.dag_version import DAGVersion
from .models.dataset_result import DatasetResult

logger = logging.getLogger(__name__)

_MIN_OBS_FOR_ADF = 20
_DEFAULT_DATASETS_DIR = Path.home() / ".causal_engine" / "datasets"


class DatasetBuilder:
    """Fetch, transform, validate, and persist a financial feature matrix."""

    def run(
        self,
        version: DAGVersion,
        manifest_path: str | Path,
        out_path: str | Path | None = None,
        strict: bool = False,
    ) -> DatasetResult:
        """Execute the full pipeline and return a DatasetResult.

        Args:
            version:       Resolved DAGVersion object (not an ID string).
            manifest_path: Path to the YAML manifest file.
            out_path:      Output Parquet path.  If None, a default path under
                           ~/.causal_engine/datasets/ is used (includes
                           version-id, manifest stem, and UTC timestamp so
                           repeated runs do not overwrite each other).
            strict:        When True, label mismatches between the manifest
                           and the DAGVersion are treated as errors rather
                           than warnings.
        """
        manifest_path = Path(manifest_path)

        # ------------------------------------------------------------------
        # Step 1: re-parse manifest for metadata we need at this level
        # ------------------------------------------------------------------
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "[ERROR] Problem: PyYAML not installed. "
                "Cause: pyyaml is required to parse manifests. "
                "Fix: pip install 'causal-hypothesis-engine[financial]'"
            ) from exc

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"[ERROR] Problem: Manifest file not found. "
                f"Cause: {manifest_path} does not exist. "
                "Fix: Check the manifest path."
            )

        with open(manifest_path) as fh:
            manifest: dict = yaml.safe_load(fh)

        frequency: str = manifest.get("frequency", "weekly")
        node_cfgs: list[dict] = manifest.get("nodes", [])
        declared_transforms: dict[str, str] = {
            n["label"]: n.get("transform", "none") for n in node_cfgs
        }

        # ------------------------------------------------------------------
        # Step 2: fetch + resample + apply declared transforms
        # ------------------------------------------------------------------
        adapter = FinancialDataAdapter()
        df, _transform_map, warnings = adapter._load_manifest_data(
            manifest_path, strict=strict
        )

        # ------------------------------------------------------------------
        # Step 3: label alignment checks
        # ------------------------------------------------------------------
        version_labels = {n.label for n in version.nodes}
        manifest_labels = set(df.columns)

        for label in manifest_labels - version_labels:
            msg = f"[WARN] Manifest node \"{label}\" is not in DAGVersion nodes."
            if strict:
                raise ValueError(
                    f"[ERROR] Problem: Manifest node \"{label}\" not in DAGVersion. "
                    "Cause: Label mismatch (strict mode). "
                    "Fix: Check spelling in manifest or DAG."
                )
            warnings.append(msg)
            logger.warning(msg)

        for label in version_labels - manifest_labels:
            msg = (
                f"[WARN] DAGVersion node \"{label}\" has no corresponding "
                "manifest column — it will be absent from the dataset."
            )
            warnings.append(msg)
            logger.warning(msg)

        # ------------------------------------------------------------------
        # Step 4 + 5: ADF tests with optional auto-diff
        # ------------------------------------------------------------------
        try:
            from statsmodels.tsa.stattools import adfuller  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "[ERROR] Problem: statsmodels not installed. "
                "Cause: statsmodels is required for ADF tests. "
                "Fix: pip install 'causal-hypothesis-engine[financial]'"
            ) from exc

        adf_results: dict[str, dict] = {}
        final_series: dict[str, pd.Series] = {}

        for label in df.columns:
            series = df[label].dropna()
            declared = declared_transforms.get(label, "none")

            # Short-series guard
            if len(series) < _MIN_OBS_FOR_ADF:
                msg = (
                    f"[WARN] Series \"{label}\" has only {len(series)} observations "
                    f"— ADF test skipped (need >= {_MIN_OBS_FOR_ADF})."
                )
                warnings.append(msg)
                logger.warning(msg)
                adf_results[label] = {
                    "statistic": None,
                    "pvalue": None,
                    "passed": False,
                    "transform_applied": "insufficient_data",
                }
                final_series[label] = series
                continue

            # First ADF pass
            try:
                adf_stat, pvalue, *_ = adfuller(series)
            except Exception as exc:
                msg = f"[WARN] ADF test raised for \"{label}\": {exc}"
                warnings.append(msg)
                logger.warning(msg)
                adf_results[label] = {
                    "statistic": None,
                    "pvalue": None,
                    "passed": False,
                    "transform_applied": declared,
                }
                final_series[label] = series
                continue

            if pvalue <= 0.05:
                # Already stationary
                adf_results[label] = {
                    "statistic": float(adf_stat),
                    "pvalue": float(pvalue),
                    "passed": True,
                    "transform_applied": declared,
                }
                final_series[label] = series
                continue

            # Non-stationary — apply one extra diff and re-test
            series_diff = series.diff().dropna()
            compound = "diff" if declared == "none" else f"{declared}+diff"

            if len(series_diff) < _MIN_OBS_FOR_ADF:
                msg = (
                    f"[WARN] Series \"{label}\" still non-stationary after extra diff "
                    f"and has only {len(series_diff)} observations — included as-is."
                )
                warnings.append(msg)
                logger.warning(msg)
                adf_results[label] = {
                    "statistic": float(adf_stat),
                    "pvalue": float(pvalue),
                    "passed": False,
                    "transform_applied": compound,
                }
                final_series[label] = series
                continue

            try:
                adf_stat2, pvalue2, *_ = adfuller(series_diff)
            except Exception as exc:
                msg = f"[WARN] ADF re-test raised for \"{label}\": {exc}"
                warnings.append(msg)
                logger.warning(msg)
                adf_results[label] = {
                    "statistic": float(adf_stat),
                    "pvalue": float(pvalue),
                    "passed": False,
                    "transform_applied": compound,
                }
                final_series[label] = series
                continue

            if pvalue2 <= 0.05:
                adf_results[label] = {
                    "statistic": float(adf_stat2),
                    "pvalue": float(pvalue2),
                    "passed": True,
                    "transform_applied": compound,
                }
                final_series[label] = series_diff
            else:
                msg = (
                    f"[WARN] Series \"{label}\" still non-stationary after extra diff "
                    "(p={pvalue2:.4f}) — included as-is."
                )
                warnings.append(msg)
                logger.warning(msg)
                adf_results[label] = {
                    "statistic": float(adf_stat2),
                    "pvalue": float(pvalue2),
                    "passed": False,
                    "transform_applied": compound,
                }
                final_series[label] = series_diff

        # ------------------------------------------------------------------
        # Step 6: build final DataFrame and compute actual date range
        # ------------------------------------------------------------------
        final_df = pd.DataFrame(final_series)
        if not final_df.empty:
            actual_start = str(final_df.index.min().date())
            actual_end = str(final_df.index.max().date())
        else:
            actual_start = manifest.get("start_date", "")
            actual_end = manifest.get("end_date", "")

        # ------------------------------------------------------------------
        # Step 7: resolve output path and write Parquet atomically
        # ------------------------------------------------------------------
        if out_path is None:
            _DEFAULT_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
            manifest_stem = manifest_path.stem
            out_path = (
                _DEFAULT_DATASETS_DIR
                / f"{version.version_id}-{manifest_stem}-{timestamp}.parquet"
            )
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = out_path.with_suffix(".parquet.tmp")
        try:
            final_df.to_parquet(tmp_path)
            with open(tmp_path, "rb") as fh:
                os.fsync(fh.fileno())
            os.rename(tmp_path, out_path)
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

        return DatasetResult(
            version_id=version.version_id,
            manifest_path=str(manifest_path.resolve()),
            columns=list(final_df.columns),
            start_date=actual_start,
            end_date=actual_end,
            frequency=frequency,
            adf_results=adf_results,
            warnings=warnings,
            output_path=str(out_path.resolve()),
        )
