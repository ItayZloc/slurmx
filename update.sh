#!/usr/bin/env bash
# Check the repo against origin and pull updates if available.
#   - Bails if the working tree is dirty (uncommitted changes).
#   - Fast-forward only — no merge commits.
#   - Re-runs 'uv sync' if pyproject.toml or uv.lock changed.
set -euo pipefail

REPO="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$REPO"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "==> Branch: $BRANCH"

# --- Dirty check ---
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: working tree has uncommitted changes. Commit or stash first."
    git status --short
    exit 1
fi

# --- Fetch ---
echo "==> Fetching origin"
git fetch origin "$BRANCH"

LOCAL="$(git rev-parse HEAD)"
REMOTE="$(git rev-parse "origin/$BRANCH")"
BASE="$(git merge-base HEAD "origin/$BRANCH")"

if [ "$LOCAL" = "$REMOTE" ]; then
    echo "==> Up to date with origin/$BRANCH"
    exit 0
fi

if [ "$LOCAL" != "$BASE" ]; then
    echo "ERROR: local branch has commits not in origin/$BRANCH (or has diverged)."
    echo "       Push or rebase manually."
    git log --oneline "$BASE..$LOCAL"
    exit 1
fi

# --- Track what changes before pulling so we can re-sync if needed ---
NEEDS_SYNC=0
if ! git diff --quiet "$LOCAL" "$REMOTE" -- pyproject.toml uv.lock; then
    NEEDS_SYNC=1
fi

echo "==> Pulling $REMOTE (fast-forward)"
git merge --ff-only "$REMOTE"

if [ "$NEEDS_SYNC" = "1" ]; then
    if command -v uv >/dev/null 2>&1; then
        echo "==> pyproject.toml or uv.lock changed — running 'uv sync'"
        uv sync
    else
        echo "WARNING: uv not on PATH — skipping 'uv sync'. Run it manually."
    fi
fi

echo "==> Updated $LOCAL -> $REMOTE"
