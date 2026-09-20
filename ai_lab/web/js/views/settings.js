// Settings: the machine, its engines, and where models are kept.
// Everything here is read-only by decision.

import { api } from '../api.js';
import { machine } from './machine.js';
import { onLog } from '../events.js';
import { element } from '../format.js';
import { paths, enginePaths } from './settings_paths.js';
import { createEngineSection } from './settings_engines.js';

// Lines arriving while a build runs, kept here so switching tabs and coming
// back does not lose them.
const logs = new Map();
let subscribed = false;

function subscribeToBuildLog() {
  if (subscribed) return;
  subscribed = true;
  onLog((event) => {
    if (!logs.has(event.source)) logs.set(event.source, []);
    const lines = logs.get(event.source);
    lines.push(event.text);
    if (lines.length > 500) lines.shift();
    const pane = document.querySelector(`[data-log="${CSS.escape(event.source)}"]`);
    if (pane) {
      pane.textContent = lines.join('\n');
      pane.scrollTop = pane.scrollHeight;
    }
    // A build's own progress and its errors both belong in the pane above,
    // with the rest of that build's output. Splitting them across two places
    // meant reading a compile in two directions at once.
  });
}

// Installed package versions are refreshed on each render. Build logs remain
// in this module so changing tabs does not erase a running compile's output.
const byEngine = new Map();
const engines = createEngineSection(logs, byEngine);

export async function render(container) {
  subscribeToBuildLog();
  const refresh = () => render(container);
  // The installed folders of every package engine, fetched alongside so the
  // engine rows can show them. A machine with none answers with an empty list
  // rather than an error, so this never decides whether the page draws.
  const [settings, installed] = await Promise.all([
    api.settings(),
    api.allInstalls().catch(() => []),
  ]);
  byEngine.clear();
  (installed || []).forEach((item) => byEngine.set(item.engine, item));
  // Two balanced stacks. On a narrow screen they return to one column in this
  // order.
  container.replaceChildren(element('div', { class: 'columns settings-columns' }, [
    element('div', {}, [
      engines(settings.engines, refresh),
      enginePaths(settings, refresh),
    ].filter(Boolean)),
    element('div', {}, [
      machine(settings, refresh),
      paths(settings, refresh),
    ].filter(Boolean)),
  ]));
}
