// Settings: the machine, its engines, and where models are kept.
// Everything here is read-only by decision.

import { api } from '../api.js';
import { showNotice } from '../confirm.js';
import { chooseFolder, chooseProgram } from '../browse.js';
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
// its own card. They are the same kind of thing — somewhere on disk that has to
// be right or a screen stops working — and looking for them in two places was
// the only reason it took two.
//
// **Nothing here is typed.** A path typed by hand is a path with a typo in it,
// and the failure arrives much later as a screen with no models on it. The one
// path that is a real choice — where the model store lives — is picked from a
// listing of what is actually there. The rest follow from it, or are where an
// engine was installed.
function paths(settings, refresh) {
  const rows = [
    chosen(settings.models_root, refresh),
    ...(settings.repositories || []).map((item) =>
      readOnly(item.name, item.path, trouble(item))),
    // An engine that cannot run here is not given a path. Pointing it at
    // something would not make it work — vLLM on the Mac needs CUDA, and no
    // path fixes that — and the engine card already says why.
    ...(settings.engines || [])
      .filter((engine) => engine.available && engine.binary)
      .map((engine) => program(engine, refresh)),
  ];
  return section('Paths', rows);
}


// The one path that is chosen rather than typed or followed. A folder, and a
// typo in it empties every screen, so it is picked from a listing of what is
// actually on the machine — and saved as soon as it is picked, because there
// is nothing half-typed to confirm.
function chosen(root, refresh) {
  return pathRow('Models root', root, {
    help: 'Every weight format is a folder in here.',
    choose: () => chooseFolder(root || null),
    save: (picked) => api.updateModelsRoot(picked),
    trouble: 'Could not move the model store',
    refresh,
  });
}


// One shape for every path somebody can change: a label, the path, and the
// button that changes it. The same shape for all of them so the labels line up
// down the left and the buttons down the right — a column that steps in and
// out because one label is longer than another is a column that reads as three
// unrelated things.
//
// Saved the moment something is picked. There is nothing half-typed to
// confirm, so there is no Save to leave switched off and forget.
function pathRow(label, path, { help, choose, save, trouble, refresh }) {
  return element('div', { class: 'row path-row' }, [
    element('strong', { text: label, title: help }),
    element('span', { class: 'path muted grow', text: path || '—' }),
    element('button', {
      class: 'action', text: 'Browse…',
      onclick: (event) => whileWorking(event.target, 'Choosing…', async () => {
        const picked = await choose();
        if (!picked) return;
        try {
          await save(picked);
        } catch (error) {
          await showNotice({ title: trouble, body: error.message });
        }
        refresh();
      }),
    }),
  ]);
}


// Which program serves an engine.
//
// Not expected to change — on a settled machine it never will — but two builds
// of llama.cpp on one box is ordinary, and being unable to say which one means
// editing a file over ssh.
//
// Picked, not typed, like the models root: a path typed by hand is a path with
// a typo in it, and the failure arrives later as an engine reporting itself
// missing. The chooser lists only files that can be launched, so finding a
// launcher in a build directory is not a hunt through source and licences.
function program(engine, refresh) {
  return pathRow(engine.name, engine.binary, {
    help: 'The program that serves this engine. Takes effect the next time a '
        + 'model starts; nothing already running is touched.',
    choose: () => chooseProgram(parentOf(engine.binary)),
    save: (picked) => api.updateEngineBinary(engine.id, picked),
    trouble: `Could not point ${engine.name} there`,
    refresh,
  });
}


// Start the chooser where this engine already is, rather than at the top of
// the disk.
function parentOf(path) {
  if (!path) return null;
  const cut = path.lastIndexOf('/');
  return cut > 0 ? path.slice(0, cut) : null;
}


// A path that follows from something else. No free space and no format pill:
// the name says which format it is, and free space belongs where a download
// chooses its destination rather than on four repeated lines. What is worth
// saying is when a folder is not there or cannot be written to, because that
// is what will fail later.
function readOnly(label, path, warning) {
  return element('div', { class: 'row tight derived path-row' }, [
    element('span', { class: 'muted', text: label }),
    element('span', { class: 'path muted grow', text: path || '—' }),
    warning
      ? element('span', { class: 'pill', style: 'color:var(--warn);border-color:var(--warn)',
                          text: warning })
      : null,
  ].filter(Boolean));
}


function trouble(item) {
  if (!item.path) return 'no models root set';
  if (!item.exists) return 'missing';
  if (!item.writable) return 'read-only';
  return '';
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
// Said only when there is something to say.
//
// A green "available" on an engine that is plainly working told nobody
// anything: the version beside its name already says it runs, and a badge that
// is always there is a badge nobody reads. What is left are the two states
// worth interrupting for — it cannot run here, or it is being rebuilt now.
function engineState(engine) {
  const source = engine.source;
  if (!engine.available) {
    return element('span', { class: 'pill off', text: engine.reason || 'not installed' });
  }
  if (source && source.state === 'running') {
    return element('span', { class: 'pill', text: 'building…' });
  }
  return null;
}

// What this engine could be updated to, and whether that is anything.
//
// A checkout is asked with git and a package engine is asked of its index,
// both on a timer, so this page already knows when it is opened. Empty when
// upstream could not be reached — an engine nobody could ask about must not
// look like one with nothing waiting.
function updateWaiting(engine) {
  const source = engine.source;
  if (source && source.update_available) return source.latest || '';
  const installed = byEngine.get(engine.id);
  return installed && installed.update_available ? installed.latest || '' : '';
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
  return (installed && installed.installed) || '';
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
// What is waiting, and the way to read about it. Nothing at all when there is
// nothing waiting: an engine at the newest version has no update to review, so
// a button offering to review one would open a page saying "no changes".
//
// **This never updates anything.** It opens what the update would bring — what
// changes, what upstream wrote, which packages would be replaced — and the real
// Update is at the foot of that. The ellipsis is the promise.
function updateControls(engine, source, refresh, waiting) {
  // Two ways an engine gets a new version, and an engine has exactly one of
  // them: a checkout is rebuilt in place, packages are installed into a new
  // folder beside the old one.
  const buildable = Boolean(source && source.exists);
  const installable = byEngine.has(engine.id);

  return [
    element('span', { class: 'pill',
                      style: 'color:var(--warn);border-color:var(--warn)',
                      text: `${waiting} available` }),
    element('button', {
      class: 'action', text: 'Update…',
      title: `Read what ${waiting} brings before taking it`,
      onclick: (event) => whileWorking(event.target, 'Reading…', () =>
        reviewUpdate(engine, take(engine, buildable, installable, refresh))),
    }),
  ];
}



function engineCard(engine, refresh) {
  const source = engine.source;
  const stored = logs.get(engine.id) || (source && source.log) || [];
  const version = installedVersion(engine);

  // One line: the name and what is running, then — pushed to the right — what
  // is waiting and the way to read about it. There is no second line, because
  // an engine at its newest version has nothing to say on one.
  //
  // The weight formats used to hang off the name — "llama.cpp · b10448 · gguf"
  // — which said something a reader of this page cannot act on, and which the
  // model list says better by only offering an entry the formats its engine
  // can read.
  const waiting = engine.available ? updateWaiting(engine) : '';
  const rows = [
    element('div', { class: 'row engine' }, [
      element('div', { class: 'grow' }, [
        element('strong', { text: engine.name }),
        version ? element('span', { class: 'muted', text: ` · ${version}` }) : null,
      ].filter(Boolean)),
      engineState(engine),
      ...(waiting ? updateControls(engine, source, refresh, waiting) : []),
    ].filter(Boolean)),
  ];

  // Only once there is a choice. With a single environment the list repeated
  // the version already beside the name and offered nothing to press; with two
  // it is the way back from an update that went badly, and the way to reclaim
  // the 8 GB the old one holds.
  const installed = byEngine.get(engine.id);
  if (installed && (installed.environments || []).length > 1) {
    rows.push(versions(installed, engine, refresh));
  }
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
