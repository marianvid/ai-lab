#!/usr/bin/env bash
# Run the manager on this machine, replacing any copy already running.
#
# There is no deployment step locally, so the manager keeps running whatever
# code it was started with. That is easy to forget after an edit, and the
# symptom is confusing: the page looks wrong in a way the code says it cannot
# be. Stopping first makes the mistake impossible.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="${AI_LAB_CONFIG:-${HOME}/.ai-lab/config.json}"

if [ ! -f "${config}" ]; then
  echo "No configuration at ${config}" >&2
  exit 1
fi

if pgrep -f "ai_lab.main --config" >/dev/null; then
  echo "Stopping the manager that is already running"
  pkill -f "ai_lab.main --config"
  # Engines are this application's children on macOS and stop with it.
  sleep 1
fi

cd "${project_dir}"
echo "Configuration: ${config}"
exec python3 -m ai_lab.main --config "${config}" "$@"
