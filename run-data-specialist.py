#!/usr/bin/env python3
"""
Run the Data Specialist through all three steps for a direction.

Step 1: Requirements mapping (impl spec → data sources)
Step 2: Raw data acquisition (download from sources)
Step 3: Data preparation (transform raw → algorithm-ready)

Each step runs as a separate claude session. The human can stop
between steps by passing --step to run only one.
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
    agent:  Agent name (e.g., "qr-data-specialist").
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
    Parse arguments and run the Data Specialist.
    """
    parser = argparse.ArgumentParser(
        description="Run the Data Specialist for a research direction.",
    )
    parser.add_argument(
        "--direction", type=int, required=True,
        help="Direction number (e.g., 7).",
    )
    parser.add_argument(
        "--step", type=int, default=None, choices=[1, 2, 3],
        help="Run only this step (default: all three).",
    )
    parser.add_argument(
        "--project-dir", type=Path, default=Path("."),
        help="Path to the project directory (default: current directory).",
    )
    args = parser.parse_args()

    project = args.project_dir.resolve()

    if not (project / "project_description.md").exists():
        print(f"Error: project_description.md not found in {project}",
              file=sys.stderr)
        sys.exit(1)

    import os
    os.chdir(project)

    d = args.direction
    steps = [args.step] if args.step else [1, 2, 3]

    print("=" * 60)
    print("Data Specialist")
    print(f"Project:   {project}")
    print(f"Direction: {d}")
    print(f"Steps:     {', '.join(str(s) for s in steps)}")
    print("=" * 60)

    step_names = {1: "Requirements Mapping", 2: "Raw Data Acquisition",
                  3: "Data Preparation"}

    for s in steps:
        print(f"\n=== Step {s}: {step_names[s]} ===")
        run_claude("qr-data-specialist", f"Direction {d}, Step {s}.")

    print()
    print("=" * 60)
    print(f"Data Specialist complete for direction {d}.")
    print("=" * 60)


if __name__ == "__main__":
    main()
