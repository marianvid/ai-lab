#!/usr/bin/env bash
# Test locally, ship beside the running code, test there, swap, then restart.
#
# The engine services are deliberately left alone: a deployment must not
# unload a model somebody is using. Only the web manager restarts.
set -Eeuo pipefail   # -E: the ERR trap (restore) also fires inside remote()

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Which machine to deploy to is not this repository's business, so there is no
# default. The private half keeps the real values; if it is checked out, they
# are read from there and this script takes no arguments.
#
#   opts/deploy.env      AI_LAB_HOST=root@... , AI_LAB_SSH_KEY=..., AI_LAB_CTID=...
#
# Without it, set them in the environment:
#   AI_LAB_HOST=root@<proxmox-host> AI_LAB_SSH_KEY=<private-key> AI_LAB_CTID=<container-id> ./scripts/deploy.sh
if [ -f "${project_dir}/opts/deploy.env" ]; then
  # shellcheck disable=SC1091
  . "${project_dir}/opts/deploy.env"
fi
target_host="${AI_LAB_HOST:?set AI_LAB_HOST, or check out the private half into opts/}"
container_id="${AI_LAB_CTID:?set AI_LAB_CTID to the Proxmox container id}"
ssh_key="${AI_LAB_SSH_KEY:?set AI_LAB_SSH_KEY to the private key for that host}"
runtime_dir="/opt/ai-lab"

next_dir="${runtime_dir}.next"     # the new code, tested here before it replaces anything
prev_dir="${runtime_dir}.prev"     # the code it replaced, put back if the manager will not start

# AI_LAB_DRY_RUN=1: print what would be done on the far side, change nothing there.
remote() {
  if [ "${AI_LAB_DRY_RUN:-0}" = 1 ]; then
    echo "  would run: pct exec ${container_id} -- $*"
    return 0
  fi
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

echo "[2/6] Uploading to ${next_dir} in container ${container_id}, beside the running code"
# The far side is not touched until the new code has passed its tests there
# (audit m14). It lands in a fresh folder; a broken ssh or tar mid-way leaves
# the code in ${runtime_dir} as it was.
# One shell inside the container: a bare `a && b` would run b on the Proxmox host.
remote "sh -lc 'rm -rf ${next_dir} && install -d ${next_dir}'"
# opts/ is the private half: machine addresses, an ssh key path, passwords.
# This script reads opts/deploy.env here, before anything is sent, so the far
# side never needs it. node_modules is the test runner for the interface, which
# runs here: the container has no node, so 25 MB would cross the wire for nothing.
pack() {
  COPYFILE_DISABLE=1 tar \
    --no-xattrs \
    --exclude=.git --exclude=.venv --exclude=.pytest_cache \
    --exclude='*.egg-info' --exclude='__pycache__' \
    --exclude='.DS_Store' --exclude='._*' --exclude=tmp \
    --exclude=opts --exclude=benchmark --exclude=node_modules \
    --exclude=build --exclude=dist --exclude=.claude \
    --exclude=package-lock.json --exclude=research \
    -C "${project_dir}" -cf - .
}
if [ "${AI_LAB_DRY_RUN:-0}" = 1 ]; then
  echo "  would send $(pack | tar -tf - | wc -l | tr -d ' ') entries to ${next_dir}"
else
  pack | ssh -i "${ssh_key}" "${target_host}" \
      "pct exec '${container_id}' -- tar -C '${next_dir}' -xf -"
fi

echo "[3/6] Running the tests inside the container, on the new code in ${next_dir}"
# Run from ${next_dir}: its own ai_lab is found first, before the installed one.
remote "sh -lc 'cd ${next_dir} && ${runtime_dir}/.venv/bin/python -m unittest discover -t . -s tests'"

# From step 4 on, any failure puts the previous code back and restarts it
# (the reviewer, on m14): a half-done swap must not wait for the next restart.
restore() {
  trap - ERR
  echo "  putting the previous code back from ${prev_dir}" >&2
  remote "sh -lc 'for item in ai_lab tests; do [ ! -e ${prev_dir}/\$item ] || \
    { rm -rf ${runtime_dir}/\$item && mv ${prev_dir}/\$item ${runtime_dir}/; }; done && \
    chown -R ai-lab-manager:ai-lab-manager ${runtime_dir} && systemctl restart ai-lab.service'" || true
  exit 1
}

echo "[4/6] Putting the new code in place (the old one kept in ${prev_dir})"
# The code tree is replaced, not overlaid: tar only adds and overwrites, so a
# file deleted in the repository would otherwise stay on the server for ever.
# The running manager holds its code in memory and restarts at the end.
remote "sh -lc 'rm -rf ${prev_dir} && install -d ${prev_dir} && \
  for item in ai_lab tests; do [ ! -e ${runtime_dir}/\$item ] || mv ${runtime_dir}/\$item ${prev_dir}/; done && \
  cp -a ${next_dir}/. ${runtime_dir}/ && rm -rf ${next_dir}'" || restore
trap restore ERR
# tar keeps the owner it found on the Mac, and that user id means nothing here.
# Put the checkout back under the account the manager actually runs as.
remote "chown -R ai-lab-manager:ai-lab-manager ${runtime_dir}"

echo "[5/6] Installing configuration, helper and units"
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
    install -o ai-lab-manager -g ai-lab-manager -m 0644 ${runtime_dir}/config.example.json /etc/ai-lab/config.json && \
  install -o root -g root -m 0755 ${runtime_dir}/system/ai-lab-control /usr/local/sbin/ai-lab-control && \
  install -o root -g root -m 0440 ${runtime_dir}/system/ai-lab-manager.sudoers /etc/sudoers.d/ai-lab && \
  install -o root -g root -m 0644 ${runtime_dir}/system/ai-lab.service /etc/systemd/system/ai-lab.service && \
  install -o root -g root -m 0644 ${runtime_dir}/system/ai-lab-engine@.service /etc/systemd/system/ai-lab-engine@.service && \
  visudo -cf /etc/sudoers.d/ai-lab && \
  usermod -aG systemd-journal ai-lab-manager && \
  systemctl daemon-reload'"

# The journal group (above) lets the manager quote an engine's own error when
# it fails to load, instead of reporting only that a process exited.
#
# Existing configuration is owned by the private opts snapshot. Migrations
# are explicit operations: rerunning a historical migration on every deploy
# can reintroduce obsolete profiles and create drift from the active snapshot.
remote "sh -lc 'install -o ai-lab-manager -g ai-lab-manager -m 0644 \
  ${runtime_dir}/image_workflows/*.json /etc/ai-lab/image-workflows/'"

# ComfyUI uses a versioned checkout and environment. The installer exits
# immediately when a managed release already exists; on the first deployment
# after this feature it creates and verifies one before configuration starts
# pointing the engine at `current`.
if [ "${AI_LAB_DRY_RUN:-0}" = 1 ]; then
  echo "  would run scripts/install-comfyui-linux.sh"
else
  "${project_dir}/scripts/install-comfyui-linux.sh"
fi

remote "'${runtime_dir}/.venv/bin/pip' install --quiet --no-deps -e '${runtime_dir}'"

echo "[6/6] Restarting the manager, then waiting until it answers and stays up"
remote "systemctl restart ai-lab.service"
if [ "${AI_LAB_DRY_RUN:-0}" != 1 ]; then
  # A simple unit is "active" as soon as its process starts, so the manager
  # itself is asked (ai_lab.hosts.ready reads its port from the configuration), and
  # systemd must not have restarted it meanwhile.
  restarts=$(remote "systemctl show -p NRestarts --value ai-lab.service")
  remote "${runtime_dir}/.venv/bin/python -m ai_lab.hosts.ready --config /etc/ai-lab/config.json" || restore
  [ "$(remote "systemctl show -p NRestarts --value ai-lab.service")" = "${restarts}" ] || {
    echo "  systemd restarted the manager while it was being checked" >&2; restore; }
fi
trap - ERR

# The operator's active files and the private snapshots must remain identical.
# This verifier prints only status and counts, never configuration values.
if [ "${AI_LAB_DRY_RUN:-0}" != 1 ] && [ -f "${project_dir}/opts/verify_sync.py" ]; then
  python3 "${project_dir}/opts/verify_sync.py"
fi

if [ "${AI_LAB_DRY_RUN:-0}" = 1 ]; then
  echo "Dry run: nothing was changed on the far side."
else
  echo "Deployment completed. Inference instances were not touched."
fi
