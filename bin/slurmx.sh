#!/usr/bin/env bash
# slurmx — unified CLI. Dispatches subcommands via cli/slurmx.py (real
# argparse subparsers, not a bash case statement).
set -e
SCRIPT="$(readlink -f "$0")"
REPO="$(dirname "$(dirname "$SCRIPT")")"
cd "$REPO"
exec "$REPO/.venv/bin/python" -m cli.slurmx "$@"
