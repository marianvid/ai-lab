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

// Every path this installation depends on, in one place.
//
// It used to be two: the model folders here, and each engine's program down in
// its own card. They are the same kind of thing — somewhere on disk that has
// to be right or a screen stops working — and looking for them in two places
// was the only reason it took two.
//
// **One editable path per real choice.** The model store is one root with a
// folder per weight format, so there is one field and four consequences; a
// path per format let GGUF sit on one disk and NVFP4 on another, which nothing
// else in this application expects. Each engine's program is its own choice,
// so it gets its own field.
function paths(settings, refresh) {
  const rows = [
    editable({
      label: 'Models root',
      help: 'Every weight format is a folder in here.',
      value: settings.models_root || '',
      browse: true,
      save: (value) => api.updateModelsRoot(value),
      trouble: 'Could not move the model store',
      refresh,
    }),
    ...(settings.repositories || []).map(derived),
    // One row per engine this machine can use, plus any that already has a
    // program configured. An engine that cannot run here and was never set up
    // — vLLM on the Mac, which needs CUDA — is not worth a field: pointing it
    // at something would not make it work, and the engine card already says
    // why.
    ...(settings.engines || [])
      .filter((engine) => engine.available || engine.binary)
      .map((engine) => editable({
      label: engine.name,
      help: 'The program that serves this engine. Takes effect the next time a '
          + 'model starts; nothing already running is touched.',
      value: engine.binary || '',
      // The chooser lists folders, and a program is a file, so there is
      // nothing for it to pick here.
      browse: false,
      save: (value) => api.updateEngineBinary(engine.id, value),
      trouble: `Could not point ${engine.name} somewhere else`,
      refresh,
    })),
  ];
  return section('Paths', rows);
}


// A path somebody may change. Save sleeps until the field is actually
// different from what is in force, so pressing it always means something.
function editable({ label, help, value, browse, save, trouble, refresh }) {
  const field = element('input', { class: 'grow path', value });
  const button = element('button', {
    class: 'action', text: 'Save', disabled: 'disabled',
    onclick: (event) => whileWorking(event.target, 'Saving…', () => store(field.value)),
  });
  field.addEventListener('input', () => { button.disabled = field.value === value; });

  const store = async (chosen) => {
    field.value = chosen;
    try {
      await save(chosen);
      refresh();
    } catch (error) {
      button.disabled = false;
      await showNotice({ title: trouble, body: error.message });
    }
  };

  return element('div', { class: 'row' }, [
    element('strong', { text: label, title: help }),
    field,
    browse
      ? element('button', {
          class: 'action', text: 'Browse…',
          onclick: async () => {
            const picked = await chooseFolder(value || null);
            if (picked) await store(picked);
          },
        })
      : null,
    button,
  ].filter(Boolean));
}


// A folder that follows from the models root. Read-only on purpose.
//
// No free space and no format pill: the name says which format it is, and free
// space belongs where a download chooses its destination rather than on four
// repeated lines. What is worth saying is when a folder is not there or cannot
// be written to, because that is what will fail later.
function derived(item) {
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

// Whether this engine can run, and nothing else.
//
// It used to turn orange and announce a new version, which put the news in
// competition with the state: an engine working perfectly well looked like
// something was wrong with it, and both engines could not be read the same
// way. What there is to update to lives on the line below, beside the button
// that reads about it.
function engineState(engine) {
  const source = engine.source;
  if (!engine.available) {
    return element('span', { class: 'pill off', text: engine.reason || 'not installed' });
  }
  if (source && source.state === 'running') {
    return element('span', { class: 'pill', text: 'building…' });
  }
  return element('span', { class: 'pill on', text: 'available' });
}

// Which version is running, whichever way this engine arrives: a build number
// read from the checkout, or the version of the environment its packages are
// installed in. vLLM has no checkout and so had no version beside its name at
// all, which made it look like the one engine nobody could tell anything
// about.
function installedVersion(engine) {
  const source = engine.source;
  if (source && source.installed) return source.installed;
  const installed = byEngine.get(engine.id);
  const active = installed && (installed.environments || [])
    .find((item) => item.active);
  return active ? active.version : '';
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

// The update line: what is waiting, and the way to read about it.
//
// **No button here updates anything.** Every one opens what the update would
// bring — what changes, what upstream wrote, which packages would be replaced
// — and the real Update sits at the foot of that. An update taken without
// reading it is a hope, and this machine runs one card.
//
// "Check for updates" used to sit here and is gone. Upstream is asked on a
// timer anyway, so the button did what was already being done, and its only
// real effect was to make the page look like it needed pressing.
function sourceControls(engine, source, refresh) {
  // Two ways an engine gets a new version, and an engine has exactly one of
  // them: a checkout is rebuilt in place, packages are installed into a new
  // folder beside the old one.
  const buildable = Boolean(source && source.exists);
  const installable = byEngine.has(engine.id);
  const waiting = source && source.update_available ? source.latest : '';

  return element('div', { class: 'row update' }, [
    element('span', { class: 'muted', text: 'Update version' }),
    element('span', { class: 'grow' }),
    // What there is to move to, next to the way to read about it. Nothing at
    // all when there is none to announce — vLLM only finds out by asking, and
    // asking costs a package resolution, so it is asked when the button is
    // pressed rather than every time this page is drawn.
    waiting
      ? element('span', { class: 'pill',
                          style: 'color:var(--warn);border-color:var(--warn)',
                          text: `${waiting} available` })
      : null,
    // The ellipsis is the promise: this opens something to read. Nothing on
    // this page updates an engine directly.
    element('button', {
      class: 'action', text: 'Update…',
      title: 'Read what this update brings before taking it',
      onclick: (event) => whileWorking(event.target, 'Reading…', () =>
        reviewUpdate(engine, take(engine, buildable, installable, refresh))),
    }),
  ].filter(Boolean));
}


function engineCard(engine, refresh) {
  const source = engine.source;
  const stored = logs.get(engine.id) || (source && source.log) || [];
  const version = installedVersion(engine);

  // The name and what is running, then whether it can run at all. The weight
  // formats used to hang off the name — "llama.cpp · b10448 · gguf", "vLLM ·
  // awq, fp8, gptq, nvfp4, safetensors" — which said something a reader of
  // this page cannot act on and which the model list says better, by only
  // offering an entry the formats an engine can read.
  const rows = [
    element('div', { class: 'row' }, [
      element('div', {}, [
        element('strong', { text: engine.name }),
        version ? element('span', { class: 'muted', text: ` · ${version}` }) : null,
      ].filter(Boolean)),
      engineState(engine),
    ]),
    engine.available ? sourceControls(engine, source, refresh) : null,
  ].filter(Boolean);

  const installed = byEngine.get(engine.id);
  if (installed) rows.push(versions(installed, engine, refresh));
  if (source && source.note) rows.push(element('div', { class: 'warn', text: source.note }));
  if (source && source.error) rows.push(element('div', { class: 'error', text: source.error }));
  const building = Boolean(source && source.state === 'running');
  if (stored.length || building) rows.push(buildLog(engine, stored, building));

  return element('div', { class: 'card' }, rows);
}

// What a build said, while it says it and afterwards.
//
// A compile takes ten minutes and nothing on screen is worse than a wall of
// output, so while it runs the pane is open and follows itself. When it is
// over the same output is worth keeping — that is where "Finished at v0.2.0"
// is, and where a failure explains itself — but a page that opens into eight
// hundred lines of cmake every time is a page with a wall in it.
//
// So afterwards it folds. Available, closed, and it says which build it was
// from rather than making that another thing to remember.
function buildLog(engine, lines, building) {
  const pane = element('pre', { class: 'log', 'data-log': engine.id,
                                text: lines.join('\n') });
  if (building) return pane;
  const last = lines.filter(Boolean).slice(-1)[0] || '';
  return element('details', { class: 'fold build-log' }, [
    element('summary', { text: last.startsWith('Finished')
      ? `Last update log · ${last.replace(/^Finished at /, 'finished at ')}`
      : 'Last update log' }),
    pane,
  ]);
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
      paths(settings, refresh),
    ]),
    element('div', {}, [
      machine(settings, refresh),
    ].filter(Boolean)),
  ]));
}
