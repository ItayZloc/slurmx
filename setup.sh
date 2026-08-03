#!/usr/bin/env bash
# Bootstrap the slurmx environment.
#   - Runs `uv sync` to create .venv and install dependencies.
#   - Symlinks every bin/*.sh into ~/.local/bin/ (with .sh stripped) so the
#     CLI (slurmx) is callable from any shell.
#   - Registers the MCP server with Claude Code, if `claude` is on PATH.
# Idempotent — safe to re-run.
set -euo pipefail

REPO="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$REPO"

echo "==> Repo: $REPO"

# --- uv check ---
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not found on PATH. Install from https://docs.astral.sh/uv/"
    exit 1
fi

# --- venv + dependencies ---
echo "==> Running 'uv sync' (creates .venv if missing)"
uv sync

# --- config.py warning ---
if [ ! -f "$REPO/config.py" ]; then
    echo
    echo "WARNING: config.py is missing. Create it with:"
    echo "    slurmx config      # pick a template, then edit it in a form"
fi

# --- Symlink every bin/*.sh into ~/.local/bin/ (with .sh stripped) ---
LINKS_DIR="$HOME/.local/bin"
mkdir -p "$LINKS_DIR"
shopt -s nullglob
for src in "$REPO"/bin/*.sh; do
    cmd=$(basename "$src" .sh)
    link="$LINKS_DIR/$cmd"
    if [ -L "$link" ] && [ "$(readlink -f "$link")" = "$(readlink -f "$src")" ]; then
        echo "==> Symlink already in place: $link"
    else
        ln -sf "$src" "$link"
        echo "==> Linked: $link -> $src"
    fi
done
shopt -u nullglob

# --- PATH check ---
case ":$PATH:" in
    *":$HOME/.local/bin:"*)
        echo "==> ~/.local/bin is on PATH"
        ;;
    *)
        echo
        echo "WARNING: ~/.local/bin is NOT on your PATH."
        echo "Add this line to your ~/.bashrc or ~/.zshrc and restart your shell:"
        echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
        ;;
esac

# --- Register the MCP server with Claude Code ---
# Skipped when `claude` isn't installed (the CLI still works on its own) and
# when slurmx is already registered, so re-running never duplicates it.
if command -v claude >/dev/null 2>&1; then
    # `mcp get` over `mcp list`: list health-checks every configured server,
    # which on a machine with a few remote ones takes seconds.
    if claude mcp get slurmx 2>/dev/null | grep -q '^slurmx:'; then
        echo "==> MCP server already registered with Claude Code"
    elif claude mcp add slurmx "$REPO/.venv/bin/python" "$REPO/server.py" >/dev/null 2>&1; then
        echo "==> Registered the MCP server with Claude Code"
    else
        echo "==> Could not register the MCP server. Add it by hand:"
        echo "    claude mcp add slurmx $REPO/.venv/bin/python $REPO/server.py"
    fi
else
    echo "==> 'claude' not on PATH — skipping MCP registration (the CLI still works)."
fi

echo
if [ -f "$REPO/WELCOME.md" ]; then
    cat "$REPO/WELCOME.md"
else
    echo "Done. Try: slurmx --help"
fi
