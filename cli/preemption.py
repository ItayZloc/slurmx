#!/usr/bin/env python3
"""Backing command for read-only SLURM preemption inspection."""

from __future__ import annotations

import argparse

import slurm_mcp


def add_info_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the read-only preemption inspection command."""


def run_info(_args: argparse.Namespace) -> None:
    print(slurm_mcp.preemption_info())
