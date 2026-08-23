// Settings: the machine, its engines, and where models are kept.
// Everything here is read-only by decision.

import { api } from '../api.js';
import { showNotice } from '../confirm.js';
import { chooseFolder } from '../browse.js';
import { whileWorking } from '../working.js';
import { reviewUpdate } from './whatchanges.js';
import { versions } from './versions.js';
import { machine } from './machine.js';
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

// One directory holds every model, with a folder in it per weight format.
//
// It used to be four editable paths, one per format, which let GGUF sit on one
// disk and NVFP4 on another — a state nothing else in this application expects
// and nobody chooses on purpose. `MODEL_STORAGE.md` has described the
// format-first tree as the layout all along; this now enforces it.
//
// A path that is wrong — pointing at a folder that was moved, or set up on
// another machine — makes every other screen useless, and fixing it should not
// mean editing a file over ssh. So the root is editable, with a chooser beside
// it, and the folders below it are shown as what they are: worked out, not
// set.
function repositories(root, items, refresh) {
  const field = element('input', { class: 'grow path', value: root || '' });

  const save = async (path) => {
    field.value = path;
    try {
      await api.updateModelsRoot(path);
      refresh();
    } catch (error) {
      await showNotice({ title: 'Could not move the model store', body: error.message });
    }
  };

  const rows = [
    element('div', { class: 'row' }, [
      element('strong', { text: 'Models root',
                          title: 'Every weight format is a folder in here.' }),
      field,
      element('button', {
        class: 'action', text: 'Browse…',
        onclick: async () => {
          const picked = await chooseFolder(root || null);
          if (picked) await save(picked);
        },
      }),
      element('button', {
        class: 'action', text: 'Save',
        onclick: (event) => whileWorking(event.target, 'Saving…',
                                         () => save(field.value)),
      }),
    ]),
    ...items.map(repositoryRow),
  ];
  return section('Model repositories', rows);
}


// One format, and where it therefore is. Read-only on purpose.
//
// No free space and no format pill: the name says which format it is, and
// free space belongs where a download chooses its destination rather than on
// four repeated lines. What is worth saying is when a folder is not there or
// cannot be written to, because that is the thing that will fail later.
function repositoryRow(item) {
  const trouble = !item.path ? 'no models root set'
    : !item.exists ? 'missing'
    : !item.writable ? 'read-only'
    : '';
  return element('div', { class: 'row tight derived' }, [
    element('span', { class: 'muted', text: item.name }),
    element('span', { class: 'path muted grow', text: item.path || '—' }),
    trouble
      ? element('span', { class: 'pill', style: 'color:var(--warn);border-color:var(--warn)',
                          text: trouble })
      : null,
  ].filter(Boolean));
}


// The installed folders of each package engine, from the last fetch.
const byEngine = new Map();

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

// What pressing the real Update button does, which depends on how this engine
// arrives. Null when neither applies, so the review shows what would change
// without offering a button it cannot honour.
function take(engine, buildable, installable, refresh) {
  if (installable) {
    return async () => {
      try {
        await api.installEngine(engine.id, '');
        logs.set(engine.id, []);
        refresh();
      } catch (error) {
        await showNotice({ title: `Could not install ${engine.name}`,
                           body: error.message });
      }
    };
  }
  if (buildable) {
    return async () => {
      try {
        await api.updateBuild(engine.id);
        logs.set(engine.id, []);
        refresh();
      } catch (error) {
        await showNotice({ title: `Could not build ${engine.name}`,
                           body: error.message });
      }
    };
  }
  return null;
}

function sourceControls(engine, source, refresh) {
  // An engine built here has a checkout to check and to rebuild. One installed
  // as packages — vLLM — has neither, and used to get no controls at all: no
  // way to see what a newer version would bring, because there was no button
  // to hang it on. Reading what would change needs no checkout, so it is
  // offered either way; only the two that act on a checkout are hidden.
  // Two ways an engine gets a new version, and an engine has exactly one of
  // them: a checkout is rebuilt in place, packages are installed into a new
  // folder beside the old one.
  const buildable = Boolean(source && source.exists);
  const installable = byEngine.has(engine.id);

  const check = buildable ? element('button', {
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
  }) : null;

  // Nothing here updates anything. Pressing this reads what the update would
  // bring — what changes, what upstream wrote, which packages would be
  // replaced — and the real Update button is at the foot of *that*. An update
  // taken without reading it is a hope, and this machine runs one card.
  const review = element('button', {
    class: 'action', text: (source && source.update_available)
      ? `Review ${source.latest}` : 'What would change',
    title: 'Read what this update brings before taking it',
    onclick: (event) => whileWorking(event.target, 'Reading…', () =>
      reviewUpdate(engine, take(engine, buildable, installable, refresh))),
  });

  return element('div', { class: 'inline' }, [check, review].filter(Boolean));
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
      engine.available ? sourceControls(engine, source, refresh) : null,
    ].filter(Boolean)),
  ];

  const installed = byEngine.get(engine.id);
  if (installed) rows.push(versions(installed, engine, refresh));
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
  // The installed folders of every package engine, fetched alongside so the
  // engine rows can show them. A machine with none answers with an empty list
  // rather than an error, so this never decides whether the page draws.
  const [settings, installed] = await Promise.all([
    api.settings(),
    api.allInstalls().catch(() => []),
  ]);
  byEngine.clear();
  (installed || []).forEach((item) => byEngine.set(item.engine, item));
  // Two columns. On the left the things that are worked on — engines that get
  // updated, folders that get pointed somewhere else. On the right what this
  // machine is, which is read far more often than it is changed.
  container.replaceChildren(element('div', { class: 'columns' }, [
    element('div', {}, [
      engines(settings.engines, refresh),
      repositories(settings.models_root, settings.repositories, refresh),
    ]),
    element('div', {}, [
      machine(settings, refresh),
    ].filter(Boolean)),
  ]));
}
