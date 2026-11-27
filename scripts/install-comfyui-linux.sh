#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# As deploy.sh: the private half when it is checked out, else the environment (audit L67).
if [ -f "${project_dir}/opts/deploy.env" ]; then
  # shellcheck disable=SC1091
  . "${project_dir}/opts/deploy.env"
fi
target_host="${AI_LAB_HOST:?set AI_LAB_HOST, or check out the private half into opts/}"
container_id="${AI_LAB_CTID:?set AI_LAB_CTID to the Proxmox container id}"
ssh_key="${AI_LAB_SSH_KEY:?set AI_LAB_SSH_KEY to the private key for that host}"

ssh -i "${ssh_key}" "${target_host}" "pct exec '${container_id}' -- bash -s" <<'REMOTE'
set -euo pipefail
root=/opt/ai/comfyui
if [ -x "${root}/current/.venv/bin/python" ] && [ -f "${root}/current/ComfyUI/main.py" ]; then
  "${root}/current/.venv/bin/python" -c 'import torch; assert torch.cuda.is_available(); print(True)'
  exit 0
fi
install -d -o ai-lab-manager -g ai-lab-manager "${root}"
commit=$(runuser -u ai-lab-manager -- env HOME=/var/lib/ai-lab git ls-remote https://github.com/Comfy-Org/ComfyUI.git HEAD | awk '{print $1}')
test -n "${commit}"
release="${root}/.release-${commit:0:12}"
if [ ! -d "${release}" ]; then
runuser -u ai-lab-manager -- install -d "${release}"
runuser -u ai-lab-manager -- env HOME=/var/lib/ai-lab git clone https://github.com/Comfy-Org/ComfyUI.git "${release}/ComfyUI"
runuser -u ai-lab-manager -- env HOME=/var/lib/ai-lab git -C "${release}/ComfyUI" checkout --detach "${commit}"
# The newest Python 3.12 installed under /opt/ai/python, not a version written here.
python=$(ls -d /opt/ai/python/cpython-3.12*/bin/python3.12 2>/dev/null | sort -V | tail -1)
test -n "${python}" || { echo "no Python 3.12 under /opt/ai/python" >&2; exit 1; }
runuser -u ai-lab-manager -- env HOME=/var/lib/ai-lab UV_NO_CONFIG=1 /usr/local/bin/uv venv "${release}/.venv" --python "${python}"
runuser -u ai-lab-manager -- env HOME=/var/lib/ai-lab UV_NO_CONFIG=1 /usr/local/bin/uv pip install \
  --python "${release}/.venv/bin/python" -r "${release}/ComfyUI/requirements.txt"
runuser -u ai-lab-manager -- env HOME=/var/lib/ai-lab bash -lc \
  "cd '${release}/ComfyUI' && '${release}/.venv/bin/python' -c 'import torch; assert torch.cuda.is_available(); import server'"
printf '{"version":"%.12s","core_commit":"%s","created_at":%s,"components":{}}\n' "${commit}" "${commit}" "$(date +%s)" > "${release}/.ai-lab-release.json"
chown ai-lab-manager:ai-lab-manager "${release}/.ai-lab-release.json"
fi
ln -sfn "${release}" "${root}/.current-new"
mv -Tf "${root}/.current-new" "${root}/current"
REMOTE
