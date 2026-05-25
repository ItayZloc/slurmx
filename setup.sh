#!/usr/bin/env bash
# Bootstrap the slurmx environment.
#   - Runs `uv sync` to create .venv and install dependencies.
#   - Symlinks every bin/*.sh into ~/.local/bin/ (with .sh stripped) so the
#     CLI (slurmx) is callable from any shell.
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
    echo "WARNING: config.py is missing. Copy an example and edit MAIL_USER:"
    echo "    cp config-examples/default.py config.py     # blank template"
    echo "    cp config-examples/yisroel.py config.py     # Yisroel's lab pre-filled"
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

echo
if [ -f "$REPO/WELCOME.md" ]; then
    cat "$REPO/WELCOME.md"
else
    echo "Done. Try: slurmx --help"
fi
