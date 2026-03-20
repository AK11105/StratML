#!/usr/bin/env bash
# This file is a template. The installer (install.sh) generates the actual
# wrapper at ~/.local/bin/stratml with the correct absolute path to main.py.
#
# To install, run from the CLI/ directory:
#   chmod +x install.sh && ./install.sh
python3 "/path/to/CLI/main.py" "$@"
