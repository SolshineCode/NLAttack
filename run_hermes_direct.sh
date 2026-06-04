#!/bin/bash
# Direct-CLI Hermes review (NO kanban). Programmatic quiet mode.
export PATH="$HOME/.local/bin:$PATH"
PROMPT="$(cat /mnt/c/Users/caleb/nla-eval-harness/hermes_review_prompt.md)"
# -Q programmatic/quiet, --query passes the prompt as a single arg, --yolo so a
# non-TTY run never hangs on an approval prompt (task is pure analysis).
timeout 300 hermes chat -Q --yolo --query "$PROMPT" 2>&1
echo "=== HERMES EXIT: $? ==="
