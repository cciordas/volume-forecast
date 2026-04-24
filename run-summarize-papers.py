#!/usr/bin/env python3
"""
Summarize all papers in the papers/ directory.

Scans papers/ for PDF files, determines which need summaries (no
corresponding .md file), and runs the qr-paper-summarizer agent on
each one. Skips PDFs that already have summaries unless --force is
used.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_claude(agent: str, prompt: str) -> str:
    """
    Run a claude agent session and return its stdout.

    Parameters
    ----------
    agent:  Agent name (e.g., "qr-paper-summarizer").
    prompt: Prompt to send to the agent.

    Returns
    -------
    The agent's stdout output.

    Raises
    ------
    SystemExit
        If the claude process fails.
    """
    result = subprocess.run(
        ["claude", "--agent", agent, "--dangerously-skip-permissions",
         "-p", prompt],
        capture_output=True, text=True,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print(f"Error: claude exited with code {result.returncode}",
              file=sys.stderr)
        sys.exit(1)
    return result.stdout


def main() -> None:
    """
    Parse arguments and run paper summarization.
    """
    parser = argparse.ArgumentParser(
        description="Summarize all papers in the papers/ directory.",
    )
    parser.add_argument(
        "--project-dir", type=Path, default=Path("."),
        help="Path to the project directory (default: current directory).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-summarize even if summary already exists.",
    )
    args = parser.parse_args()

    project = args.project_dir.resolve()

    # --- Preflight checks ---------------------------------------------------

    papers_dir = project / "papers"
    if not papers_dir.is_dir():
        print(f"Error: papers/ directory not found in {project}",
              file=sys.stderr)
        sys.exit(1)

    import os
    os.chdir(project)

    # --- Find PDFs needing summaries ----------------------------------------

    pdfs = sorted(papers_dir.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found in papers/")
        return

    to_summarize: list[Path] = []
    skipped = 0
    for pdf in pdfs:
        summary = pdf.with_suffix(".md")
        if summary.exists() and not args.force:
            skipped += 1
            continue
        to_summarize.append(pdf)

    if not to_summarize:
        print(f"All {len(pdfs)} papers already have summaries")
        return

    # --- Summarize ----------------------------------------------------------

    print("=" * 60)
    print("Paper summarization")
    print(f"Project:  {project}")
    print(f"Papers:   {len(to_summarize)} to summarize, {skipped} skipped")
    print("=" * 60)

    markdown_dir = papers_dir / "markdown"

    failures = 0
    for i, pdf in enumerate(to_summarize, 1):
        summary = pdf.with_suffix(".md")
        markdown = markdown_dir / (pdf.stem + ".md")
        source = markdown if markdown.exists() else pdf
        print(f"\n=== [{i}/{len(to_summarize)}] {pdf.name} "
              f"({'markdown' if source == markdown else 'pdf'}) ===")

        prompt = (
            f"Read the paper at {source.relative_to(project)} and produce "
            f"a structured summary. "
            f"Write the summary to {summary.relative_to(project)}."
        )

        try:
            run_claude("qr-paper-summarizer", prompt)
        except SystemExit:
            failures += 1
            print(f"Failed to summarize {pdf.name}, continuing...",
                  file=sys.stderr)
            continue

    # --- Report -------------------------------------------------------------

    created = sum(
        1 for pdf in to_summarize if pdf.with_suffix(".md").exists()
    )
    print()
    print("=" * 60)
    print(f"Summarization complete. {created}/{len(to_summarize)} "
          f"summaries created.")
    if failures:
        print(f"  {failures} failure(s) — re-run to retry failed papers.")
    print("=" * 60)


if __name__ == "__main__":
    main()
