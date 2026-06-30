"""HTML experiment dashboard generator.

Reads results from the ``results/`` directory and generates an HTML report
with success rate trends, latency distributions, and training curves.

Usage::

    python pipeline/dashboard.py --results-dir results/ --output results/dashboard.html
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
from typing import Any

logger = logging.getLogger(__name__)


def _load_json_results(results_dir: pathlib.Path) -> list[dict[str, Any]]:
    """Load all JSON result files from a directory."""
    results = []
    for f in sorted(results_dir.glob("**/*.json")):
        if f.name == "job_registry.json":
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
            data["_source_file"] = str(f)
            results.append(data)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping %s: %s", f, e)
    return results


def _load_job_registry(results_dir: pathlib.Path) -> list[dict[str, Any]]:
    """Load job registry if it exists."""
    registry_path = results_dir / "job_registry.json"
    if not registry_path.exists():
        return []
    try:
        with open(registry_path) as f:
            data = json.load(f)
        return data.get("jobs", [])
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load job registry: %s", e)
        return []


_STATUS_COLORS = {
    "RUNNING": "#28a745",
    "PENDING": "#ffc107",
    "COMPLETED": "#17a2b8",
    "FAILED": "#dc3545",
    "CANCELLED": "#6c757d",
    "TIMEOUT": "#fd7e14",
}


def _generate_html(
    results: list[dict[str, Any]],
    jobs: list[dict[str, Any]] | None = None,
    title: str = "pi0 Experiment Dashboard",
) -> str:
    """Generate an HTML dashboard from loaded results."""
    jobs = jobs or []

    # --- Eval results table ---
    rows: list[str] = []
    for r in results:
        source = r.get("_source_file", "unknown")
        sr = r.get("success_rate", "N/A")
        if isinstance(sr, float):
            sr = f"{sr * 100:.1f}%"
        total_ep = r.get("total_episodes", "N/A")
        suite = r.get("task_suite", "N/A")

        rows.append(
            f"<tr><td>{source}</td><td>{suite}</td>"
            f"<td>{sr}</td><td>{total_ep}</td></tr>"
        )

    eval_rows = "\n".join(rows) if rows else "<tr><td colspan='4'>No results found</td></tr>"

    # --- Job tracker table ---
    job_rows: list[str] = []
    for j in jobs:
        status = j.get("status", "UNKNOWN")
        color = _STATUS_COLORS.get(status, "#6c757d")
        badge = f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:4px;font-size:0.85em">{status}</span>'
        job_rows.append(
            f"<tr><td><code>{j.get('job_id','')}</code></td>"
            f"<td>{j.get('name','')}</td>"
            f"<td>{j.get('partition','')}</td>"
            f"<td>{j.get('commit','')}</td>"
            f"<td>{badge}</td>"
            f"<td>{j.get('note','')}</td></tr>"
        )

    job_table_rows = "\n".join(job_rows) if job_rows else "<tr><td colspan='6'>No jobs registered</td></tr>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
h1 {{ color: #333; }}
table {{ border-collapse: collapse; width: 100%; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 30px; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
th {{ background: #4a90d9; color: white; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
tr:hover {{ background: #e8f0fe; }}
.summary {{ margin: 20px 0; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
code {{ background: #f0f0f0; padding: 1px 4px; border-radius: 3px; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="summary">
  <p><strong>Total experiments:</strong> {len(results)}</p>
  <p><strong>Tracked jobs:</strong> {len(jobs)}</p>
  <p><strong>Generated:</strong> {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>

<h2>Slurm Job Tracker</h2>
<table>
<tr><th>Job ID</th><th>Name</th><th>Partition</th><th>Commit</th><th>Status</th><th>Note</th></tr>
{job_table_rows}
</table>

<h2>Evaluation Results</h2>
<table>
<tr><th>Source</th><th>Suite</th><th>Success Rate</th><th>Episodes</th></tr>
{eval_rows}
</table>
</body>
</html>"""


def generate_dashboard(
    results_dir: str = "results",
    output: str = "results/dashboard.html",
    title: str = "pi0 Experiment Dashboard",
) -> pathlib.Path:
    """Generate the HTML dashboard."""
    rd = pathlib.Path(results_dir)
    results = _load_json_results(rd)
    jobs = _load_job_registry(rd)
    logger.info("Loaded %d result files and %d jobs from %s", len(results), len(jobs), rd)

    html = _generate_html(results, jobs, title)

    out = pathlib.Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(html)
    logger.info("Dashboard saved to %s", out)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Experiment dashboard")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="results/dashboard.html")
    parser.add_argument("--title", type=str, default="pi0 Experiment Dashboard")
    args = parser.parse_args(argv)

    generate_dashboard(args.results_dir, args.output, args.title)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
