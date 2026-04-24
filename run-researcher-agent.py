#!/usr/bin/env python3
"""
Automate the Researcher adversarial refinement protocol for one direction.

The Researcher deeply analyzes research papers for a specific modeling
direction and produces an implementation specification. Like the Analyst,
it uses adversarial refinement: a proposer produces the spec, a critic
reviews it, and the proposer revises. This script runs the full protocol
as separate claude sessions to preserve adversarial independence.

Two modes:

  Adaptive (default): The Referee agent reads each critique and decides
  whether to continue or stop. The process runs until the Referee calls
  CONVERGED, up to a maximum number of rounds.

  Fixed: Run exactly N rounds (no Referee). Useful when you know how
  many rounds you want, or for reproducibility.

In both modes, the proposer always runs one final time after the last
critique, so that every critique is incorporated into a revision.

The agent reads artifacts/research_directions.md to find the direction
name, description, and assigned papers — you only need to provide the
direction number and run number.

Each run uses a run number so that multiple independent runs for the same
direction don't conflict. After the adversarial cycle completes, the
Auditor agent updates the findings tracker for cross-run comparison.

The script is composable: it detects whether a previous run left state
and continues from there.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run_claude(agent: str, prompt: str) -> str:
    """
    Run a claude agent session and return its stdout.

    Parameters
    ----------
    agent:  Agent name (e.g., "qr-researcher").
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


def latest_file(pattern: str, directory: Path) -> Path | None:
    """
    Find the highest-numbered file matching a glob pattern.

    Parameters
    ----------
    pattern:   Glob pattern (e.g., "impl_spec_draft_*.md").
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
    path:   File path (e.g., "impl_spec_draft_3.md").
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
        description="Run the Researcher adversarial refinement protocol "
                    "for one direction.",
    )
    parser.add_argument(
        "--direction", type=int, required=True,
        help="Direction number.",
    )
    parser.add_argument(
        "--run", type=int, required=True,
        help="Run number. Use different numbers for independent runs.",
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
        "--no-audit", action="store_true",
        help="Skip the Auditor step after the run completes.",
    )
    parser.add_argument(
        "--project-dir", type=Path, default=Path("."),
        help="Path to the project directory (default: current directory).",
    )
    args = parser.parse_args()

    project = args.project_dir.resolve()
    dir_num = args.direction
    run_num = args.run

    # --- Preflight checks ---------------------------------------------------

    if not (project / "project_description.md").exists():
        print(f"Error: project_description.md not found in {project}",
              file=sys.stderr)
        sys.exit(1)

    if not (project / "artifacts" / "research_directions.md").exists():
        print("Error: artifacts/research_directions.md not found "
              "— run the Analyst first", file=sys.stderr)
        sys.exit(1)

    import os
    os.chdir(project)

    ctx = f"direction {dir_num} run {run_num}"
    work = project / "work" / "researcher" / f"direction_{dir_num}" / f"run_{run_num}"
    work.mkdir(parents=True, exist_ok=True)

    fixed_mode = args.rounds is not None

    # --- Print configuration ------------------------------------------------

    print("=" * 60)
    print("Researcher adversarial refinement")
    print(f"Project:   {project}")
    print(f"Direction: {dir_num}")
    print(f"Run:       {run_num}")
    print(f"Work dir:  {work.relative_to(project)}")
    if fixed_mode:
        print(f"Mode:      fixed ({args.rounds} rounds, no Referee)")
    else:
        print(f"Mode:      adaptive (Referee decides, "
              f"max {args.max_rounds} rounds)")
    print(f"Auditor:   {'disabled' if args.no_audit else 'enabled'}")
    print("=" * 60)

    # --- Detect state from previous runs ------------------------------------

    latest_draft = latest_file("impl_spec_draft_*.md", work)

    if latest_draft is not None:
        draft_num = extract_number(latest_draft, "draft_")
        critique = work / f"researcher_critique_{draft_num}.md"
        if not critique.exists():
            print(f"\n=== Continuing from previous run. "
                  f"Critic reviews draft {draft_num} ===")
            run_claude("qr-researcher",
                       f"You are the critic for {ctx}.")

    # --- Fixed mode ---------------------------------------------------------

    if fixed_mode:
        for r in range(1, args.rounds + 1):
            print(f"\n=== Round {r}/{args.rounds}: Proposer ===")
            run_claude("qr-researcher",
                       f"You are the proposer for {ctx}.")

            print(f"\n=== Round {r}/{args.rounds}: Critic ===")
            run_claude("qr-researcher",
                       f"You are the critic for {ctx}.")

    # --- Adaptive mode ------------------------------------------------------

    else:
        # Initial proposer if no drafts exist yet
        latest_draft = latest_file("impl_spec_draft_*.md", work)
        if latest_draft is None:
            print("\n=== Round 1: Proposer (initial) ===")
            run_claude("qr-researcher",
                       f"You are the proposer for {ctx}.")

        max_rounds = args.max_rounds
        for r in range(1, max_rounds + 1):
            print(f"\n=== Round {r}/{max_rounds}: Critic ===")
            run_claude("qr-researcher",
                       f"You are the critic for {ctx}.")

            # Find the latest critique for the Referee
            latest_critique = latest_file("researcher_critique_*.md", work)
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
            run_claude("qr-researcher",
                       f"You are the proposer for {ctx}.")

    # --- Final proposer step ------------------------------------------------

    print(f"\n=== Final: Proposer ===")
    run_claude("qr-researcher", f"You are the proposer for {ctx}.")

    # --- Copy final draft to canonical location -----------------------------

    latest_draft = latest_file("impl_spec_draft_*.md", work)
    dest = project / "artifacts" / f"direction_{dir_num}" / "impl_spec.md"

    if latest_draft is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(latest_draft, dest)
        print()
        print("-" * 60)
        print(f"Adversarial cycle complete. Copied {latest_draft.name}")
        print(f"  to {dest.relative_to(project)}")
        print("-" * 60)
    else:
        print()
        print("-" * 60)
        print(f"Warning: no draft found in {work.relative_to(project)}")
        print("-" * 60)

    # --- Auditor step -------------------------------------------------------
    # Update the findings tracker for this direction. The Auditor compares
    # this run's final spec against the accumulated tracker and reports
    # whether the run added new findings.

    if not args.no_audit and latest_draft is not None:
        tracker_dir = project / "work" / "researcher" / f"direction_{dir_num}"
        tracker_path = tracker_dir / "tracker.md"

        print(f"\n=== Auditor: updating tracker for direction {dir_num} ===")
        verdict = run_claude(
            "qr-auditor",
            f"Direction {dir_num}, run {run_num} just completed. "
            f"Read the final spec at {latest_draft} and update the "
            f"tracker at {tracker_path}.",
        )
        print()
        print("=" * 60)
        print(f"Researcher run {run_num} complete for direction {dir_num}.")
        # Extract, print, and save the VERDICT line if present
        for line in verdict.splitlines():
            if line.strip().startswith("VERDICT:"):
                verdict_line = line.strip()
                print(f"Auditor: {verdict_line}")
                # Append to tracker
                with open(tracker_path, "a") as f:
                    f.write(f"\n**Run {run_num} audit:** {verdict_line}\n")
                # Append to progress log
                progress = project / "logs" / "progress.md"
                with open(progress, "a") as f:
                    f.write(
                        f"\n#### Auditor — Direction {dir_num}, "
                        f"Run {run_num}\n{verdict_line}\n"
                    )
                break
        print("=" * 60)
    else:
        print()
        print("=" * 60)
        print(f"Researcher run {run_num} complete for direction {dir_num}.")
        if args.no_audit:
            print("Auditor: skipped (--no-audit)")
        print("=" * 60)


if __name__ == "__main__":
    main()
