#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "${project_dir}/opts/deploy.env"
target_host="${AI_LAB_HOST:?}"
container_id="${AI_LAB_CTID:-102}"
ssh_key="${AI_LAB_SSH_KEY:?}"
source_path="/mnt/corsair-4tb/test_models"
target_path="/test_models"

ssh -i "${ssh_key}" "${target_host}" bash -s -- "${container_id}" \
  "${source_path}" "${target_path}" <<'REMOTE'
set -euo pipefail
ctid="$1"; source_path="$2"; target_path="$3"
install -d -m 0755 "${source_path}"
config="$(pct config "${ctid}")"
if grep -Fq "${source_path},mp=${target_path}" <<<"${config}"; then
  echo "mount already configured; checking ownership"
else
  for index in $(seq 0 31); do
    if ! grep -q "^mp${index}:" <<<"${config}"; then
      pct set "${ctid}" "-mp${index}" "${source_path},mp=${target_path}"
      break
    fi
  done
fi
pct exec "${ctid}" -- test -d "${target_path}"
uid="$(pct exec "${ctid}" -- id -u ai-lab-manager)"
gid="$(pct exec "${ctid}" -- id -g ai-lab-manager)"
# Default unprivileged-LXC mapping. Refuse a custom mapping rather than guess.
if pct config "${ctid}" | grep -q '^lxc.idmap:'; then
  echo "custom LXC idmap: refusing to infer host ownership" >&2
  exit 1
fi
chown "$((100000 + uid)):$((100000 + gid))" "${source_path}"
pct exec "${ctid}" -- runuser -u ai-lab-manager -- test -w "${target_path}"
REMOTE
