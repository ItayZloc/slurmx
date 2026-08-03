#!/usr/bin/env bash
# Bootstrap the slurmx environment.
#   - Installs uv if it isn't there, since nothing else works without it.
#   - Runs `uv sync` to create .venv and install dependencies.
#   - Symlinks every bin/*.sh into ~/.local/bin/ (with .sh stripped) so the
#     CLI (slurmx) is callable from any shell, and puts that directory on PATH
#     for good if it isn't already.
#   - Registers the MCP server with Claude Code, if `claude` is on PATH.
# Idempotent — safe to re-run.
set -euo pipefail

REPO="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$REPO"

LINKS_DIR="$HOME/.local/bin"
NOTES=()          # actionable lines, printed at the very end where they're read

echo "==> Repo: $REPO"

# --- PATH first ---------------------------------------------------------- #
# A non-login shell often has neither directory on PATH, so uv can be sitting
# in ~/.local/bin and still look missing. Add them before deciding anything.
for d in "$LINKS_DIR" "$HOME/.cargo/bin"; do
    case ":$PATH:" in
        *":$d:"*) ;;
        *) [ -d "$d" ] && PATH="$d:$PATH" ;;
    esac
done
export PATH

# --- uv ------------------------------------------------------------------ #
if command -v uv >/dev/null 2>&1; then
    echo "==> uv: $(command -v uv) ($(uv --version 2>/dev/null))"
else
    echo "==> uv not found — installing it into ~/.local/bin"
    if command -v curl >/dev/null 2>&1; then
        fetch=(curl -LsSf https://astral.sh/uv/install.sh)
    elif command -v wget >/dev/null 2>&1; then
        fetch=(wget -qO- https://astral.sh/uv/install.sh)
    else
        echo "ERROR: need curl or wget to install uv. Install it yourself, then re-run:"
        echo "    https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
    if ! "${fetch[@]}" | sh; then
        echo
        echo "ERROR: could not install uv — no network from this node, most likely."
        echo "Install it yourself and re-run this script:"
        echo "    https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
    # The installer writes to ~/.local/bin (or ~/.cargo/bin on older versions)
    # and ships an env file that puts it on PATH. Pick it up without a new shell.
    for env_file in "$HOME/.local/bin/env" "$HOME/.cargo/env"; do
        # shellcheck disable=SC1090
        [ -f "$env_file" ] && . "$env_file"
    done
    case ":$PATH:" in *":$LINKS_DIR:"*) ;; *) PATH="$LINKS_DIR:$PATH" ;; esac
    export PATH
    if ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: installed uv but still can't find it on PATH. Open a new shell and re-run."
        exit 1
    fi
    echo "==> uv installed: $(command -v uv)"
fi

# --- venv + dependencies ------------------------------------------------- #
echo "==> Running 'uv sync' (creates .venv if missing)"
uv sync

# --- Symlink every bin/*.sh into ~/.local/bin/ (with .sh stripped) -------- #
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

# --- Keep ~/.local/bin on PATH across shells ----------------------------- #
# Linking `slurmx` somewhere the user's shell never looks is a setup that
# reports success and leaves them with "command not found".
case "$(basename "${SHELL:-/bin/bash}")" in
    zsh)  RC="$HOME/.zshrc" ;;
    bash) RC="$HOME/.bashrc" ;;
    *)    RC="$HOME/.profile" ;;
esac
if [ -f "$RC" ] && grep -q '\.local/bin' "$RC"; then
    echo "==> $(basename "$RC") already puts ~/.local/bin on PATH"
elif [ -n "${SLURMX_NO_PATH_EDIT:-}" ]; then
    NOTES+=("Add this to your $(basename "$RC"):  export PATH=\"\$HOME/.local/bin:\$PATH\"")
else
    {
        echo ''
        echo '# added by slurmx setup'
        echo 'export PATH="$HOME/.local/bin:$PATH"'
    } >> "$RC"
    echo "==> Added ~/.local/bin to PATH in $RC"
    NOTES+=("Run 'source $RC' (or open a new shell) so 'slurmx' is on PATH.")
fi

# --- Register the MCP server with Claude Code ---------------------------- #
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
        echo "==> Could not register the MCP server"
        NOTES+=("Register the MCP server: claude mcp add slurmx $REPO/.venv/bin/python $REPO/server.py")
    fi
else
    echo "==> 'claude' not on PATH — skipping MCP registration (the CLI still works)"
    NOTES+=("Install Claude Code, then re-run this script to register the MCP server.")
fi

if [ ! -f "$REPO/config.py" ]; then
    NOTES+=("Run 'slurmx config' to create config.py — nothing else works until you do.")
fi

echo
if [ -f "$REPO/WELCOME.md" ]; then
    cat "$REPO/WELCOME.md"
else
    echo "Done. Try: slurmx --help"
fi

# Printed last, after WELCOME.md, because anything above it scrolls away.
if [ ${#NOTES[@]} -gt 0 ]; then
    echo
    echo "DO THIS NEXT"
    for i in "${!NOTES[@]}"; do
        echo "  $((i + 1)). ${NOTES[$i]}"
    done
fi
