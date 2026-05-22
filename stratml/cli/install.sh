#!/usr/bin/env bash
# =============================================
# StratML CLI Installer — Bash (Linux / macOS)
# =============================================
# Usage (from project root or stratml/cli/):
#   bash stratml/cli/install.sh
#   bash stratml/cli/install.sh --uninstall

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MAIN_PY="$SCRIPT_DIR/main.py"
BIN_DIR="$HOME/.local/bin"
WRAPPER="$BIN_DIR/stratml"

# ── Uninstall ────────────────────────────────
if [[ "${1:-}" == "--uninstall" ]]; then
    rm -f "$WRAPPER"
    echo "Removed $WRAPPER"
    echo "Uninstall complete."
    exit 0
fi

# ── Verify Python 3.12+ ───────────────────────
PYTHON=""
for candidate in python3 python python3.12 python3.13; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo 0)
        major=$("$candidate" -c "import sys; print(sys.version_info.major)" 2>/dev/null || echo 0)
        if [[ "$major" -eq 3 && "$ver" -ge 12 ]]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo "ERROR: Python 3.12+ not found. Install it and re-run."
    exit 1
fi
echo "Using: $PYTHON ($($PYTHON --version))"

# ── Install dependencies ──────────────────────
echo ""
echo "Installing dependencies from $PROJECT_ROOT ..."

if command -v uv &>/dev/null; then
    echo "Using uv ..."
    (cd "$PROJECT_ROOT" && uv sync)
elif [[ -f "$PROJECT_ROOT/requirements.txt" ]]; then
    echo "Using pip ..."
    "$PYTHON" -m pip install -r "$PROJECT_ROOT/requirements.txt" --quiet
else
    echo "Using pip (editable install) ..."
    "$PYTHON" -m pip install -e "$PROJECT_ROOT" --quiet
fi

# ── Create bin dir ────────────────────────────
mkdir -p "$BIN_DIR"

# ── Write wrapper script ──────────────────────
cat > "$WRAPPER" <<EOF
#!/usr/bin/env bash
exec "$PYTHON" "$MAIN_PY" "\$@"
EOF
chmod +x "$WRAPPER"
echo "Wrote wrapper: $WRAPPER"

# ── Ensure bin dir is on PATH ─────────────────
SHELL_RC=""
case "$SHELL" in
    */zsh)  SHELL_RC="$HOME/.zshrc" ;;
    */bash) SHELL_RC="$HOME/.bashrc" ;;
    *)      SHELL_RC="$HOME/.profile" ;;
esac

PATH_LINE='export PATH="$HOME/.local/bin:$PATH"'
if ! grep -qF '.local/bin' "$SHELL_RC" 2>/dev/null; then
    echo "" >> "$SHELL_RC"
    echo "# Added by StratML installer" >> "$SHELL_RC"
    echo "$PATH_LINE" >> "$SHELL_RC"
    echo "Added PATH entry to $SHELL_RC"
else
    echo "\$HOME/.local/bin already in $SHELL_RC"
fi

# ── Done ──────────────────────────────────────
echo ""
echo "Installation complete!"
echo "Reload your shell or run:"
echo "  source $SHELL_RC"
echo ""
echo "Then verify with:"
echo "  stratml doctor"
