// What an update would bring, shown before the button that takes it.
//
// Pressing "Update" used to mean hoping. This is the screen in between: what
// is installed, what it would become, which changes matter on this machine,
// what the people upstream wrote about it, and — for an engine installed as
// packages — exactly which packages would be replaced and which of them would
// go *backwards*.
//
// That last one is the reason this exists. Release notes describe a version;
// they say nothing about what will happen to this particular machine. Moving
// vLLM from the installed nightly to 0.27.1 moves fifteen packages and takes
// eight of them backwards, because the nightly had pulled newer kernel
// libraries than the stable release pins. No release note anywhere says that.
//
// The real Update button is at the bottom, after all of it.

import { api } from '../api.js';
import { showNotice } from '../confirm.js';
import { element } from '../format.js';
import { whileWorking } from '../working.js';

// Long notes are folded. v0.27.0 ran to 24,000 characters and opening a dialog
// straight into that is not reading either.
const FOLD_AFTER = 1400;


export async function reviewUpdate(engine, onConfirm) {
  let found;
  try {
    found = await api.buildChanges(engine.id);
  } catch (error) {
    return showNotice({ title: `Could not read what would change in ${engine.name}`,
                        body: error.message });
  }
  return new Promise((resolve) => {
    const dialog = element('dialog', { class: 'confirm review' },
                           body(found, engine, onConfirm, () => dialog.close()));
    dialog.addEventListener('close', () => { dialog.remove(); resolve(); });
    document.body.append(dialog);
    dialog.showModal();
  });
}


function body(found, engine, onConfirm, close) {
  // `onConfirm` is null for an engine this machine cannot update by itself —
  // vLLM, which arrives as packages rather than as a checkout. Everything
  // above still applies and is worth reading; there is simply nothing here to
  // press, and pretending otherwise would be a button that lies.
  const nothing = !onConfirm || !found.latest || found.installed === found.latest;
  return [
    element('h3', { text: `${found.engine_name || engine.name}` }),
    versions(found),
    found.unreadable
      ? element('p', { class: 'warn small', text: found.unreadable })
      : null,
    packages(found.packages),
    areas(found),
    notes(found.notes),
    element('div', { class: 'row buttons' }, [
      element('button', { class: 'action', autofocus: 'autofocus', text: 'Close',
                          onclick: close }),
      onConfirm || !found.latest || found.installed === found.latest
        ? null
        : element('p', { class: 'muted small',
                         text: `Updating ${found.engine_name || engine.name} is `
                             + 'not done from here yet.' }),
      nothing
        ? null
        : element('button', {
            class: 'action primary', text: `Update to ${found.latest}`,
            onclick: (event) => whileWorking(event.target, 'Starting…', async () => {
              await onConfirm();
              close();
            }),
          }),
    ].filter(Boolean)),
  ].filter(Boolean);
}


function versions(found) {
  if (!found.latest) {
    return element('p', { class: 'muted', text: `Installed ${found.installed || 'unknown'}` });
  }
  if (found.installed === found.latest) {
    return element('p', { class: 'muted', text: `Up to date at ${found.installed}` });
  }
  return element('p', { class: 'versions' }, [
    element('strong', { text: found.installed || 'unknown' }),
    element('span', { class: 'muted', text: ' → ' }),
    element('strong', { text: found.latest }),
  ]);
}


// Which packages would be replaced. Only for an engine installed as packages;
// a checkout replaces nothing, it is rebuilt.
function packages(moves) {
  if (!moves || !moves.length) return null;
  const backwards = moves.filter((move) => move.backwards);
  return element('section', { class: 'moves' }, [
    element('h4', {}, [
      element('span', { text: `${moves.length} package${moves.length === 1 ? '' : 's'} replaced` }),
      backwards.length
        ? element('span', { class: 'pill warn',
                            text: `${backwards.length} going backwards`,
                            title: 'These would be replaced with older versions' })
        : null,
    ].filter(Boolean)),
    element('table', { class: 'moves' }, element('tbody', {},
      moves.map((move) => element('tr', { class: move.backwards ? 'backwards' : '' }, [
        element('td', {}, [
          element('span', { text: move.name }),
          move.heavy
            ? element('span', { class: 'pill', text: 'heavy',
                                title: 'Gigabytes move with this one' })
            : null,
        ].filter(Boolean)),
        element('td', { class: 'muted', text: move.was || '—' }),
        element('td', { class: 'muted', text: '→' }),
        element('td', { text: move.becomes || '—' }),
      ])))),
  ]);
}


// The changes, sorted by whether this machine uses that part of the engine.
// Nothing is thrown away: what does not apply here is counted and can be
// opened, because a summary that quietly drops things is one nobody trusts.
function areas(found) {
  const mine = Object.entries(found.by_area || {});
  const theirs = Object.entries(found.other_areas || {});
  if (!mine.length && !theirs.length) return null;
  return element('section', { class: 'areas' }, [
    element('h4', { text: `${count(mine)} changes here` }),
    tally(mine),
    list(found.yours),
    theirs.length
      ? folded(`${count(theirs)} for parts this machine does not use`,
               [tally(theirs), list(found.others)])
      : null,
  ].filter(Boolean));
}

function count(pairs) {
  return pairs.reduce((total, [, many]) => total + many, 0);
}

function tally(pairs) {
  return element('p', { class: 'tally' }, pairs.map(([name, many]) =>
    element('span', { class: 'pill', text: `${name} ${many}` })));
}

function list(changes) {
  if (!changes || !changes.length) return null;
  return element('ul', { class: 'changes' }, changes.map((change) =>
    element('li', {}, [
      element('span', { class: 'muted area', text: change.area }),
      element('span', { text: change.title }),
    ])));
}


// Prose written upstream. Shown as it was written — sifting a release note by
// guessing at prefixes would damage the one thing that was written for a
// reader.
function notes(text) {
  if (!text) return null;
  const long = text.length > FOLD_AFTER;
  const pane = element('pre', { class: 'notes', text });
  return element('section', {},
    long ? [folded(`Release notes (${text.length.toLocaleString()} characters)`, [pane])]
         : [element('h4', { text: 'Release notes' }), pane]);
}


function folded(summary, children) {
  return element('details', { class: 'fold' },
    [element('summary', { text: summary }), ...children.filter(Boolean)]);
}
