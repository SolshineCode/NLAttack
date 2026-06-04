#!/bin/bash
# Block until the Hermes review task leaves 'running', then print its result.
export PATH="$HOME/.local/bin:$PATH"
TASK=t_0d1d3bb9
for i in $(seq 1 120); do   # up to ~30 min at 15s cadence
  STATUS="$(hermes kanban show "$TASK" 2>/dev/null | grep -m1 'status:' | awk '{print $2}')"
  if [ "$STATUS" != "running" ] && [ "$STATUS" != "ready" ]; then
    echo "=== FINAL STATUS: $STATUS ==="
    hermes kanban show "$TASK" 2>&1
    exit 0
  fi
  sleep 15
done
echo "=== TIMED OUT waiting; last status: $STATUS ==="
hermes kanban show "$TASK" 2>&1 | tail -40
