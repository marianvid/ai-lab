#!/usr/bin/env bash
# Test locally, ship, test again on the far side, then restart.
#
# The engine services are deliberately left alone: a deployment must not
# unload a model somebody is using. Only the web manager restarts.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Which machine to deploy to is not this repository's business, so there is no
# default. The private half keeps the real values; if it is checked out, they
# are read from there and this script takes no arguments.
#
#   opts/deploy.env      AI_LAB_HOST=root@... , AI_LAB_SSH_KEY=..., AI_LAB_CTID=...
#
# Without it, set them in the environment:
#   AI_LAB_HOST=root@proxmox.lan AI_LAB_SSH_KEY=~/.ssh/key ./scripts/deploy.sh
if [ -f "${project_dir}/opts/deploy.env" ]; then
  # shellcheck disable=SC1091
  . "${project_dir}/opts/deploy.env"
fi
target_host="${AI_LAB_HOST:?set AI_LAB_HOST, or check out the private half into opts/}"
container_id="${AI_LAB_CTID:-102}"
ssh_key="${AI_LAB_SSH_KEY:?set AI_LAB_SSH_KEY to the private key for that host}"
runtime_dir="/opt/ai-lab"

remote() {
  ssh -i "${ssh_key}" "${target_host}" "pct exec '${container_id}' -- $*"
}

echo "[1/6] Running tests locally"
(cd "${project_dir}" && python3 -m unittest discover -t . -s tests)

# The interface is tested here and not in the container, which has no node.
# The code is the same either way; only the runner is missing on the far side.
if command -v node >/dev/null && [ -d "${project_dir}/node_modules" ]; then
  (cd "${project_dir}" && node --test tests/ui/test_*.js)
else
  echo "  (skipping the interface tests: run npm install first)"
fi

echo "[2/6] Uploading to LXC ${container_id}"
# Remove the code tree first: tar only adds and overwrites, so a file deleted
# in the repository would otherwise stay on the server for ever — a stale
# module still importable, a stale script still served. The running manager
# holds its code in memory and is restarted at the end, so this is safe.
remote "rm -rf ${runtime_dir}/ai_lab ${runtime_dir}/tests"
# opts/ is the private half: machine addresses, an ssh key path, passwords.
# This script reads opts/deploy.env here, before anything is sent, so the far
# side never needs it — and the far side runs a web page with no authentication
# on it. node_modules is the test runner for the interface, which runs here
# too: the container has no node, so 25 MB crosses the wire for nothing.
COPYFILE_DISABLE=1 tar \
  --no-xattrs \
  --exclude=.git --exclude=.venv --exclude=.pytest_cache \
  --exclude='*.egg-info' --exclude='__pycache__' \
  --exclude='.DS_Store' --exclude='._*' --exclude=tmp \
  --exclude=opts --exclude=node_modules --exclude=research \
  -C "${project_dir}" -cf - . \
  | ssh -i "${ssh_key}" "${target_host}" \
      "pct exec '${container_id}' -- tar -C '${runtime_dir}' -xf -"

# tar keeps the owner it found on the Mac, and that user id means nothing here:
# the files land owned by a number with no account behind it. It works only
# because everything happens to be world-readable. Put the checkout back under
# the account the manager actually runs as.
remote "chown -R ai-lab-manager:ai-lab-manager ${runtime_dir}"

echo "[3/6] Installing configuration, helper and units"
# The configuration is seeded, never overwritten. Instances are created from
# the interface and live in /etc/ai-lab/config.json, so copying the repository
# version over it on every deployment would quietly delete them. The file in
# the repository is the starting point for a fresh machine, not the truth about
# a running one.
remote "sh -lc '\
  install -d -o ai-lab-manager -g ai-lab-manager -m 0755 \
    /etc/ai-lab /etc/ai-lab/image-workflows /var/lib/ai-lab \
    /var/lib/ai-lab/launch /var/lib/ai-lab/image-jobs \
    /opt/ai/paddleocr /opt/ai/comfyui \
    /models/images/ocr /models/images/generation /models/images/edit \
    /test_models/images/ocr /test_models/images/generation /test_models/images/edit && \
  test -f /etc/ai-lab/config.json || \
    install -o ai-lab-manager -g ai-lab-manager -m 0644 ${runtime_dir}/config.json /etc/ai-lab/config.json && \
  install -o root -g root -m 0755 ${runtime_dir}/system/ai-lab-control /usr/local/sbin/ai-lab-control && \
  install -o root -g root -m 0440 ${runtime_dir}/system/ai-lab-manager.sudoers /etc/sudoers.d/ai-lab && \
  install -o root -g root -m 0644 ${runtime_dir}/system/ai-lab.service /etc/systemd/system/ai-lab.service && \
  install -o root -g root -m 0644 ${runtime_dir}/system/ai-lab-engine@.service /etc/systemd/system/ai-lab-engine@.service && \
  visudo -cf /etc/sudoers.d/ai-lab && \
  usermod -aG systemd-journal ai-lab-manager && \
  systemctl daemon-reload'"

# Add new completion-phase keys without replacing the operator's existing
# instances, limits or paths. The migration is idempotent and keeps one backup.
remote "${runtime_dir}/.venv/bin/python ${runtime_dir}/scripts/migrate-completion-config.py /etc/ai-lab/config.json"
remote "sh -lc 'install -o ai-lab-manager -g ai-lab-manager -m 0644 \
  ${runtime_dir}/image_workflows/*.json /etc/ai-lab/image-workflows/'"

# ComfyUI uses a versioned checkout and environment. The installer exits
# immediately when a managed release already exists; on the first deployment
# after this feature it creates and verifies one before configuration starts
# pointing the engine at `current`.
"${project_dir}/scripts/install-comfyui-linux.sh"

# Read-only access to the journal, so that when an engine fails to load the
# manager can quote its own error instead of reporting that a process exited.

echo "[4/6] Installing the package"
remote "'${runtime_dir}/.venv/bin/pip' install --quiet --no-deps -e '${runtime_dir}'"

echo "[5/6] Running the same tests inside the container"
remote "sh -lc 'cd ${runtime_dir} && ${runtime_dir}/.venv/bin/python -m unittest discover -t . -s tests'"

echo "[6/6] Restarting the manager"
remote "systemctl restart ai-lab.service"
sleep 2
remote "systemctl is-active --quiet ai-lab.service"

echo "Deployment completed. Inference instances were not touched."
