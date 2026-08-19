#!/usr/bin/env bash
# Install the macOS manager service and its menu bar control for this user.
set -euo pipefail

if [ "$(uname -s)" != "Darwin" ]; then
  echo "This installer is for macOS only." >&2
  exit 1
fi

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="${AI_LAB_CONFIG:-${HOME}/.ai-lab/config.json}"
python="${AI_LAB_PYTHON:-$(command -v python3)}"
uid="$(id -u)"
domain="gui/${uid}"
menu_label="com.ai-lab.menu"
agents_dir="${HOME}/Library/LaunchAgents"
logs_dir="${HOME}/Library/Logs/AI-Lab"
app_dir="${HOME}/Applications/AI-Lab Menu.app"
manager_plist="${agents_dir}/com.ai-lab.manager.plist"
menu_plist="${agents_dir}/${menu_label}.plist"

if [ ! -f "${config}" ]; then
  echo "No configuration at ${config}" >&2
  exit 1
fi

if ! "${python}" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
  echo "AI-Lab needs Python 3.11 or newer; found ${python}" >&2
  exit 1
fi

mkdir -p "${agents_dir}" "${logs_dir}" \
  "${app_dir}/Contents/MacOS" "${app_dir}/Contents/Resources"

/usr/bin/swiftc "${project_dir}/system/macos/AI-LabMenu.swift" \
  -framework AppKit \
  -o "${app_dir}/Contents/MacOS/AI-Lab Menu"

"${python}" - "${app_dir}/Contents/Resources/settings.json" \
  "${project_dir}" "${config}" "${python}" <<'PY'
import json
import sys

target, project, config, python = sys.argv[1:]
with open(target, "w") as handle:
    json.dump({
        "projectDirectory": project,
        "configPath": config,
        "pythonPath": python,
    }, handle, indent=2)
PY

"${python}" - "${app_dir}/Contents/Info.plist" <<'PY'
import plistlib
import sys

payload = {
    "CFBundleDevelopmentRegion": "en",
    "CFBundleExecutable": "AI-Lab Menu",
    "CFBundleIdentifier": "com.ai-lab.menu",
    "CFBundleInfoDictionaryVersion": "6.0",
    "CFBundleName": "AI-Lab Menu",
    "CFBundlePackageType": "APPL",
    "CFBundleShortVersionString": "1.0",
    "LSMinimumSystemVersion": "13.0",
    "LSUIElement": True,
    "NSHighResolutionCapable": True,
}
with open(sys.argv[1], "wb") as handle:
    plistlib.dump(payload, handle, sort_keys=False)
PY

"${python}" - "${menu_plist}" "${app_dir}/Contents/MacOS/AI-Lab Menu" "${logs_dir}" <<'PY'
import plistlib
import sys

target, executable, logs = sys.argv[1:]
payload = {
    "Label": "com.ai-lab.menu",
    "ProgramArguments": [executable],
    "RunAtLoad": True,
    "KeepAlive": False,
    "AbandonProcessGroup": False,
    "StandardOutPath": f"{logs}/menu.log",
    "StandardErrorPath": f"{logs}/menu-error.log",
}
with open(target, "wb") as handle:
    plistlib.dump(payload, handle, sort_keys=False)
PY

# Remove the old two-service installation. The menu app now owns the manager.
pkill -f "ai_lab.main --config ${config}" 2>/dev/null || true

launchctl bootout "${domain}" "${manager_plist}" 2>/dev/null || true
launchctl bootout "${domain}" "${menu_plist}" 2>/dev/null || true
rm -f "${manager_plist}"
launchctl bootstrap "${domain}" "${menu_plist}"

read -r host port < <("${python}" -c '
import json, sys
config = json.load(open(sys.argv[1]))
host = config.get("host", "127.0.0.1")
print(("127.0.0.1" if host == "0.0.0.0" else host), config.get("port", 8090))
' "${config}")
manager_url="http://${host}:${port}"
ready=false
for _attempt in {1..15}; do
  if curl -fs --max-time 2 "${manager_url}/" >/dev/null; then
    ready=true
    break
  fi
  sleep 1
done

if [ "${ready}" != true ]; then
  echo "The service was installed but did not answer at ${manager_url}." >&2
  echo "Check ${logs_dir}/manager-error.log" >&2
  exit 1
fi

echo "AI-Lab menu controller installed."
echo "Manager: ${manager_url}"
echo "Logs: ${logs_dir}"
