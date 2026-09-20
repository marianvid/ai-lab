// Model roots, derived repositories and engine launcher paths.

import { api } from '../api.js';
import { showNotice } from '../confirm.js';
import { chooseFolder, chooseProgram } from '../browse.js';
import { whileWorking } from '../working.js';
import { element } from '../format.js';
import { section } from './settings_section.js';

// The model store and the folders derived from it.
//
// **Nothing here is typed.** A path typed by hand is a path with a typo in it,
// and the failure arrives much later as a screen with no models on it. The one
// path that is a real choice — where the model store lives — is picked from a
// listing of what is actually there. The rest follow from it, or are where an
// engine was installed.
export function paths(settings, refresh) {
  const roots = settings.model_roots || [
    { id: 'core', name: 'Core', path: settings.models_root, enabled: true },
  ];
  const rows = [
    ...roots.map((item) => chosenRoot(item, refresh)),
    ...displayRepositories(settings).map((item) =>
      readOnly(item.name, item.path, trouble(item))),
  ];
  return section('Paths', rows);
}

function chosenRoot(root, refresh) {
  const isCore = root.id === 'core';
  return pathRow(isCore ? 'Production models' : 'Temporary / benchmark models', root.path, {
    help: isCore ? 'Approved models used by AI-Lab.'
                 : 'Models being evaluated before promotion to production.',
    choose: () => chooseFolder(root.path || null),
    // Core keeps the original endpoint: it is the one root every existing
    // deployment already has, and it can never be disabled, so it does not
    // need the newer per-root PATCH the benchmark tier uses.
    save: (picked) => isCore
      ? api.updateModelsRoot(picked)
      : api.updateModelRoot(root.id, { path: picked, enabled: true }),
    trouble: 'Could not configure ' + root.name + ' storage',
    refresh,
  });
}


// The catalogue needs a repository for every format an engine can load. The
// Settings page does not: a person needs to know where a kind of model lives,
// not that two engines read different formats below the same folder. Keep the
// detailed repositories in configuration and collapse audio here to its three
// useful roots.
function displayRepositories(settings) {
  const repositories = settings.repositories || [];
  const visible = repositories.filter((item) =>
    item.root_id === 'core'
    && (!item.task || item.task === 'text-generation'));
  const audio = [
    ['transcription', 'Audio transcription', 'audio/asr'],
    ['vad', 'Voice activity detection', 'audio/vad'],
    ['diarization', 'Speaker diarization', 'audio/diarization'],
  ];
  audio.forEach(([task, name, subpath]) => {
    const members = repositories.filter((item) =>
      item.root_id === 'core' && item.task === task);
    if (!members.length) return;
    visible.push({
      name,
      path: settings.models_root
        ? `${settings.models_root.replace(/\/+$/, '')}/${subpath}`
        : '',
      exists: members.some((item) => item.exists),
      writable: members.some((item) => item.writable),
    });
  });
  return visible;
}


// Programs are paths too, but they belong with the engines they launch rather
// than with the model store. Keeping them in their own card also keeps the
// engine status list compact and easy to scan.
export function enginePaths(settings, refresh) {
  const rows = (settings.engines || [])
    // An engine that cannot run here is not given a path. Pointing it at
    // something would not make it work — vLLM on the Mac needs CUDA, and no
    // path fixes that — and the engine row already says why.
    .filter((engine) => engine.available && engine.binary)
    .map((engine) => program(engine, refresh));
  if (!rows.length) return null;
  const found = section('Engine paths', rows);
  found.classList.add('engine-paths');
  return found;
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
    element('span', { class: 'path muted grow', text: path || '—',
                      ...(path ? { title: path } : {}) }),
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
