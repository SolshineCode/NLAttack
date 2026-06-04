#!/bin/bash
# Create the Hermes kanban review task. Run inside WSL:
#   wsl.exe -d Ubuntu -- bash /mnt/c/Users/caleb/nla-eval-harness/mk_hermes_task.sh
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
BODY="$(cat /mnt/c/Users/caleb/nla-eval-harness/hermes_review_prompt.md)"
hermes kanban create \
  "Peer-review NLAttack: 20 NLA evaluation tests + weak-filter hypothesis" \
  --body "$BODY" \
  --assignee researcher \
  --priority 1 \
  --max-runtime 30m \
  --idempotency-key nlattack-review-20260604 \
  --json
