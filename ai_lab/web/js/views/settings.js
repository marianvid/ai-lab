// Settings: repositories, free space, accelerator and engines.
// Everything here is read-only by decision.

import { api } from '../api.js';
import { showNotice } from '../confirm.js';
import { chooseFolder } from '../browse.js';
import { whileWorking } from '../working.js';
import { onLog } from '../events.js';
import { bytes, element, seconds } from '../format.js';

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

// A heading above its panel, the same shape for every section.
function section(title, children) {
  return element('div', { class: 'section' }, [
    element('h3', { text: title }),
    element('div', { class: 'card' }, children),
  ]);
}

function accelerator(snapshot, host) {
  const rows = [
    ['Name', snapshot.name || '—'],
    ['Kind', `${snapshot.kind} · ${snapshot.memory_kind} memory`],
    ['Memory', snapshot.memory_total_mb
      ? `${Math.round(snapshot.memory_used_mb)} / ${Math.round(snapshot.memory_total_mb)} MB`
      : '—'],
    ['Temperature', snapshot.temperature_c != null ? `${snapshot.temperature_c} °C` : '—'],
    ['Utilisation', snapshot.utilization_percent != null ? `${snapshot.utilization_percent} %` : '—'],
    ['Supervisor', host.supervisor],
  ];
  return section('Accelerator', [
    ...rows.map(([label, value]) => element('div', { class: 'row' }, [
      element('span', { class: 'muted', text: label }),
      element('span', { text: value }),
    ])),
    element('p', { class: 'muted',
                   text: 'Read-only: accelerator settings are changed '
                         + 'deliberately elsewhere.' }),
  ]);
}

// Each repository's path is editable, with a chooser beside it. A path that is
// wrong — pointing at a folder that was moved, or set up on another machine —
// makes every other screen useless, and fixing it should not mean editing a
// file over ssh.
function repositoryRow(item, refresh) {
  const field = element('input', { class: 'grow path', value: item.path });

  const save = async (path) => {
    field.value = path;
    try {
      await api.updateRepository(item.id, { path });
      // Nothing is said on success: the field now shows the new path, which
      // is the whole message.
      refresh();
    } catch (error) {
      await showNotice({ title: `Could not point ${item.name} at ${path}`,
                         body: error.message });
    }
  };

  field.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') save(field.value.trim());
  });

  // Only says something when something is wrong. A folder that is there and
  // writable needs no announcement.
  const trouble = !item.exists ? 'missing'
    : !item.writable ? 'read-only' : null;

  return element('div', { class: 'row tight' }, [
    element('span', { class: 'label', text: item.name }),
    field,
    trouble ? element('span', { class: 'error', text: trouble }) : null,
    element('button', {
      class: 'action', text: 'Browse…',
      onclick: async () => {
        const picked = await chooseFolder(item.exists ? item.path : null);
        if (picked) await save(picked);
      },
    }),
    element('button', { class: 'action', text: 'Save',
                        onclick: () => save(field.value.trim()) }),
  ].filter(Boolean));
}

function repositories(list, refresh) {
  return section('Model repositories',
                 list.map((item) => repositoryRow(item, refresh)));
}

function engineState(engine) {
  const source = engine.source;
  if (!engine.available) {
    return element('span', { class: 'pill off', text: engine.reason || 'not installed' });
  }
  if (source && source.state === 'running') {
    return element('span', { class: 'pill', text: 'building…' });
  }
  if (source && source.update_available) {
    return element('span', {
      class: 'pill', style: 'color:var(--warn);border-color:var(--warn)',
      text: `${source.latest} available`,
    });
  }
  return element('span', { class: 'pill on', text: 'available' });
}

function sourceControls(engine, source, refresh) {
  const check = element('button', {
    class: 'action', text: 'Check for updates',
    // The answer is the pill beside the engine's name: it becomes either
    // "available" or "<version> available". Saying the same thing in a
    // sentence as well was two places to keep in step.
    onclick: (event) => whileWorking(event.target, 'Checking…', async () => {
      try {
        await api.checkBuild(engine.id);
        refresh();
      } catch (error) {
        await showNotice({ title: `Could not check ${engine.name}`,
                           body: error.message });
      }
    }),
  });

  const update = source.update_available
    ? element('button', {
        class: 'action', text: `Update to ${source.latest}`,
        // The pill turns to "building…" and the log pane starts filling.
        onclick: (event) => whileWorking(event.target, 'Starting…', async () => {
          try {
            await api.updateBuild(engine.id);
            logs.set(engine.id, []);
            refresh();
          } catch (error) {
            await showNotice({ title: `Could not build ${engine.name}`,
                               body: error.message });
          }
        }),
      })
    : null;

  return element('div', { class: 'inline' }, [check, update].filter(Boolean));
}

function engineCard(engine, refresh) {
  const source = engine.source;
  const stored = logs.get(engine.id) || (source && source.log) || [];
  const version = source && source.installed ? ` · ${source.installed}` : '';

  const rows = [
    element('div', { class: 'row' }, [
      element('div', {}, [
        element('strong', { text: engine.name }),
        element('span', { class: 'muted',
                          text: `${version} · ${engine.formats.join(', ')}` }),
      ]),
      engineState(engine),
    ]),
    element('div', { class: 'row' }, [
      element('span', {
        class: 'path muted', text: engine.binary || '',
        title: source && source.exists
          ? `built from ${source.path}${source.commit ? ` (${source.commit})` : ''}`
          : '',
      }),
      source && source.exists ? sourceControls(engine, source, refresh) : null,
    ].filter(Boolean)),
  ];

  if (source && source.note) rows.push(element('div', { class: 'warn', text: source.note }));
  if (source && source.error) rows.push(element('div', { class: 'error', text: source.error }));
  if (stored.length || (source && source.state === 'running')) {
    rows.push(element('pre', { class: 'log', 'data-log': engine.id,
                               text: stored.join('\n') }));
  }

  return element('div', { class: 'card' }, rows);
}

function engines(list, refresh) {
  // Each engine gets its own panel, so the heading stands above the group.
  return element('div', { class: 'section' }, [
    element('h3', { text: 'Engines' }),
    ...list.map((engine) => engineCard(engine, refresh)),
  ]);
}

export async function render(container) {
  subscribeToBuildLog();
  const refresh = () => render(container);
  const settings = await api.settings();
  // Two columns: what you change on the left, what you can only read on the
  // right. The accelerator is a report, so it sits out of the way of the
  // things that have buttons.
  container.replaceChildren(element('div', { class: 'columns' }, [
    element('div', {}, [
      engines(settings.engines, refresh),
      repositories(settings.repositories, refresh),
    ]),
    element('div', {}, [
      accelerator(settings.accelerator, settings.host),
    ]),
  ]));
}
