#!/usr/bin/env python3
"""Backing commands for SLURM preemption inspection and the guarded probe."""

from __future__ import annotations

import argparse

import slurm_mcp


def add_info_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the read-only preemption inspection command."""


def run_info(_args: argparse.Namespace) -> None:
    print(slurm_mcp.preemption_info())


def add_probe_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--real", action="store_true", help="Submit the disposable probe after every safety check.")
    parser.add_argument("--max-seconds", type=int, default=600, help="Bounded real-probe wait, 1-3600 seconds (default: 600).")


def run_probe(args: argparse.Namespace) -> None:
    print(slurm_mcp.probe_preemption(dry_run=not args.real, max_seconds=args.max_seconds))
