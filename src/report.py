"""Builds a self-contained, downloadable HTML summary report."""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape


def build_html_report(
    dataset_name: str,
    n_rows: int,
    n_cols: int,
    text_col: str,
    label_col: str,
    preprocessing_config: dict,
    model_results: list[dict],
) -> str:
    """`model_results`: list of dicts with model_key, accuracy, precision, recall, f1,
    roc_auc (nullable), train_time_sec, predict_time_sec, n_train, n_test."""
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    best_model = max(model_results, key=lambda r: r["f1"], default=None) if model_results else None

    preprocessing_rows = "".join(
        f"<tr><td>{escape(str(k))}</td><td>{escape(str(v))}</td></tr>"
        for k, v in preprocessing_config.items()
    )

    model_rows = "".join(
        f"""<tr class="{"best" if best_model and r["model_key"] == best_model["model_key"] else ""}">
            <td>{escape(r["model_key"])}</td>
            <td>{r["accuracy"]:.3f}</td>
            <td>{r["precision"]:.3f}</td>
            <td>{r["recall"]:.3f}</td>
            <td>{r["f1"]:.3f}</td>
            <td>{f"{r['roc_auc']:.3f}" if r.get("roc_auc") is not None else "N/A"}</td>
            <td>{r["train_time_sec"]:.3f}s</td>
            <td>{r["predict_time_sec"]:.3f}s</td>
        </tr>"""
        for r in model_results
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Model report: {escape(dataset_name)}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 900px; margin: 2rem auto; color: #1a1a1a; }}
  h1 {{ font-size: 1.5rem; }}
  h2 {{ font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: .3rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: .75rem; }}
  th, td {{ text-align: left; padding: .4rem .6rem; border-bottom: 1px solid #eee; font-size: .9rem; }}
  th {{ background: #f5f5f5; }}
  tr.best {{ background: #eaf7ee; font-weight: 600; }}
  .meta {{ color: #666; font-size: .85rem; }}
</style>
</head>
<body>
<h1>Spam/Ham classification report</h1>
<p class="meta">Generated {generated_at} &middot; Dataset: {escape(dataset_name)}</p>

<h2>Dataset</h2>
<table>
  <tr><td>Rows</td><td>{n_rows:,}</td></tr>
  <tr><td>Columns</td><td>{n_cols}</td></tr>
  <tr><td>Text column</td><td>{escape(str(text_col))}</td></tr>
  <tr><td>Label column</td><td>{escape(str(label_col))}</td></tr>
</table>

<h2>Preprocessing configuration</h2>
<table>{preprocessing_rows}</table>

<h2>Model results</h2>
<table>
  <tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th><th>Train time</th><th>Predict time</th></tr>
  {model_rows}
</table>
{f'<p class="meta">Best model by F1-score: <strong>{escape(best_model["model_key"])}</strong></p>' if best_model else ""}

</body>
</html>"""
