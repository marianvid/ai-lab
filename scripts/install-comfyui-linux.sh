#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "${project_dir}/opts/deploy.env"
target_host="${AI_LAB_HOST:?}"
container_id="${AI_LAB_CTID:-102}"
ssh_key="${AI_LAB_SSH_KEY:?}"

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
runuser -u ai-lab-manager -- env HOME=/var/lib/ai-lab UV_NO_CONFIG=1 /usr/local/bin/uv venv "${release}/.venv" --python /opt/ai/python/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12
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
