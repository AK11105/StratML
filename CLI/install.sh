#!/usr/bin/env bash
# =========================
# StratML CLI Installer
# =========================

set -e

BIN_PATH="$HOME/.local/bin"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_PY="$SCRIPT_DIR/main.py"
SH_DEST="$BIN_PATH/stratml"

echo "Installing StratML CLI..."

# 1. Create bin folder if not exists
if [ ! -d "$BIN_PATH" ]; then
    mkdir -p "$BIN_PATH"
    echo "Created $BIN_PATH"
fi

# 2. Write wrapper with absolute path baked in (mirrors stratml.bat approach)
cat > "$SH_DEST" <<EOF
#!/usr/bin/env bash
python3 "$MAIN_PY" "\$@"
EOF
chmod +x "$SH_DEST"
echo "Installed stratml to $SH_DEST"

# 3. Add to PATH if not already present
SHELL_RC=""
if [ "$(basename "$SHELL")" = "zsh" ]; then
    SHELL_RC="$HOME/.zshrc"
else
    SHELL_RC="$HOME/.bashrc"
fi

if ! grep -q "$BIN_PATH" "$SHELL_RC" 2>/dev/null; then
    echo "export PATH=\"$BIN_PATH:\$PATH\"" >> "$SHELL_RC"
    echo "Added $BIN_PATH to PATH in $SHELL_RC"
else
    echo "$BIN_PATH already in PATH"
fi

echo ""
echo "Installation complete!"
echo "Run: source $SHELL_RC"
echo "Then confirm with: stratml init"
