#!/usr/bin/env python3
"""
Automate the Analyst adversarial refinement protocol.

The Analyst clusters research papers into independent modeling directions
using adversarial refinement: a proposer produces a clustering, a critic
reviews it, and the proposer revises. This script runs the full protocol
as separate claude sessions (one per step) to preserve adversarial
independence — the proposer and critic never share conversation context.

Two modes:

  Adaptive (default): The Referee agent reads each critique and decides
  whether to continue or stop. The process runs until the Referee calls
  CONVERGED, up to a maximum number of rounds.

  Fixed: Run exactly N rounds (no Referee). Useful when you know how
  many rounds you want, or for reproducibility.

In both modes, the proposer always runs one final time after the last
critique, so that every critique is incorporated into a revision.

The script is composable: it detects whether a previous run left state
in work/analyst/ and continues from there.
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
    agent:  Agent name (e.g., "qr-analyst").
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
    # Print stdout so the user sees agent output in real time
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print(f"Error: claude exited with code {result.returncode}",
              file=sys.stderr)
        sys.exit(1)
    return result.stdout


def latest_file(pattern: str, directory: Path) -> Path | None:
    """
    Find the highest-numbered file matching a glob pattern.

    Parameters
    ----------
    pattern:   Glob pattern (e.g., "research_directions_draft_*.md").
    directory: Directory to search in.

    Returns
    -------
    Path to the latest file, or None if no matches.
    """
    files = sorted(directory.glob(pattern))
    return files[-1] if files else None


def extract_number(path: Path, prefix: str) -> int | None:
    """
    Extract a number from a filename after a prefix.

    Parameters
    ----------
    path:   File path (e.g., "research_directions_draft_3.md").
    prefix: Prefix before the number (e.g., "draft_").

    Returns
    -------
    The extracted number, or None if not found.
    """
    stem = path.stem
    idx = stem.find(prefix)
    if idx == -1:
        return None
    num_str = stem[idx + len(prefix):]
    try:
        return int(num_str)
    except ValueError:
        return None


def main():
    """
    Parse arguments and run the adversarial refinement protocol.
    """
    parser = argparse.ArgumentParser(
        description="Run the Analyst adversarial refinement protocol.",
    )
    parser.add_argument(
        "--rounds", type=int, default=None,
        help="Fixed mode: run exactly N rounds (disables Referee).",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=10,
        help="Adaptive mode: max rounds before forcing stop (default: 10).",
    )
    parser.add_argument(
        "--project-dir", type=Path, default=Path("."),
        help="Path to the project directory (default: current directory).",
    )
    args = parser.parse_args()

    project = args.project_dir.resolve()

    # --- Preflight checks ---------------------------------------------------

    if not (project / "project_description.md").exists():
        print(f"Error: project_description.md not found in {project}",
              file=sys.stderr)
        sys.exit(1)

    if not (project / "artifacts" / "paper_manifest.md").exists():
        print("Error: artifacts/paper_manifest.md not found "
              "— run the Librarian first", file=sys.stderr)
        sys.exit(1)

    import os
    os.chdir(project)

    work = project / "work" / "analyst"
    work.mkdir(parents=True, exist_ok=True)

    # --- Print configuration ------------------------------------------------

    fixed_mode = args.rounds is not None
    print("=" * 60)
    print("Analyst adversarial refinement")
    print(f"Project: {project}")
    if fixed_mode:
        print(f"Mode:    fixed ({args.rounds} rounds, no Referee)")
    else:
        print(f"Mode:    adaptive (Referee decides, max {args.max_rounds} "
              "rounds)")
    print("=" * 60)

    # --- Detect state from previous runs ------------------------------------

    latest_draft = latest_file("research_directions_draft_*.md", work)

    if latest_draft is not None:
        draft_num = extract_number(latest_draft, "draft_")
        critique = work / f"analyst_critique_{draft_num}.md"
        if not critique.exists():
            print(f"\n=== Continuing from previous run. "
                  f"Critic reviews draft {draft_num} ===")
            run_claude("qr-analyst", "You are the critic.")

    # --- Fixed mode ---------------------------------------------------------

    if fixed_mode:
        for r in range(1, args.rounds + 1):
            print(f"\n=== Round {r}/{args.rounds}: Proposer ===")
            run_claude("qr-analyst", "You are the proposer.")

            print(f"\n=== Round {r}/{args.rounds}: Critic ===")
            run_claude("qr-analyst", "You are the critic.")

    # --- Adaptive mode ------------------------------------------------------

    else:
        # Initial proposer if no drafts exist yet
        latest_draft = latest_file("research_directions_draft_*.md", work)
        if latest_draft is None:
            print("\n=== Round 1: Proposer (initial) ===")
            run_claude("qr-analyst", "You are the proposer.")

        max_rounds = args.max_rounds
        for r in range(1, max_rounds + 1):
            print(f"\n=== Round {r}/{max_rounds}: Critic ===")
            run_claude("qr-analyst", "You are the critic.")

            # Find the latest critique for the Referee
            latest_critique = latest_file("analyst_critique_*.md", work)
            if latest_critique is None:
                print("ERROR: critic did not produce a critique file",
                      file=sys.stderr)
                print("Check the critic's output above for errors.",
                      file=sys.stderr)
                sys.exit(1)

            print(f"\n=== Round {r}/{max_rounds}: Referee assessing "
                  "critique ===")
            verdict = run_claude(
                "qr-referee",
                f"Read the critique at {latest_critique} and decide: "
                "CONTINUE or CONVERGED.",
            )

            if "CONVERGED" in verdict:
                print(f"\n=== Referee called CONVERGED after round {r} ===")
                break

            if r >= max_rounds:
                print(f"\n=== Max rounds ({max_rounds}) reached "
                      "— stopping ===")
                break

            print(f"\n=== Round {r + 1}: Proposer (revision) ===")
            run_claude("qr-analyst", "You are the proposer.")

    # --- Final proposer step ------------------------------------------------

    print("\n=== Final: Proposer ===")
    run_claude("qr-analyst", "You are the proposer.")

    # --- Copy final draft to canonical location -----------------------------

    latest_draft = latest_file("research_directions_draft_*.md", work)
    dest = project / "artifacts" / "research_directions.md"

    if latest_draft is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(latest_draft, dest)
        print()
        print("=" * 60)
        print(f"Analyst complete. Copied {latest_draft.name}")
        print(f"  to {dest.relative_to(project)}")
        print("=" * 60)
    else:
        print()
        print("=" * 60)
        print("Analyst complete. Warning: no draft found in work/analyst/")
        print("=" * 60)


if __name__ == "__main__":
    main()
