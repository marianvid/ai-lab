#!/usr/bin/env bash
# Remove the macOS service and menu bar app. Configuration and models stay.
set -euo pipefail

uid="$(id -u)"
domain="gui/${uid}"
agents_dir="${HOME}/Library/LaunchAgents"
manager_plist="${agents_dir}/com.ai-lab.manager.plist"
menu_plist="${agents_dir}/com.ai-lab.menu.plist"
app_dir="${HOME}/Applications/AI-Lab Menu.app"

launchctl bootout "${domain}" "${manager_plist}" 2>/dev/null || true
launchctl bootout "${domain}" "${menu_plist}" 2>/dev/null || true
rm -f "${manager_plist}" "${menu_plist}"
rm -rf "${app_dir}"

echo "AI-Lab service and menu icon removed. Configuration, logs and models were kept."
