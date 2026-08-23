// Picking a path on the machine the server runs on.
//
// A web page cannot open a file dialog there — the browser only knows about
// the computer it is running on, which may not be the same one. So the server
// lists what is there and this walks through it.
//
// Two things get picked, and they end differently. A **folder** is chosen by
// standing in it and saying so, because the folder you are looking into is the
// one you mean. A **program** is chosen by clicking it, because clicking a file
// cannot mean "go inside". One walker, two endings.

import { confirmDestructive } from './confirm.js';
import { element } from './format.js';

async function listing(path, programs) {
  const parts = [];
  if (path) parts.push(`path=${encodeURIComponent(path)}`);
  if (programs) parts.push('programs=1');
  const query = parts.length ? `?${parts.join('&')}` : '';
  const response = await fetch(`/api/browse${query}`);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || 'Could not read that folder');
  return payload;
}

export function chooseFolder(startAt) {
  return pick({ startAt, programs: false, title: 'Choose a folder' });
}


// Pick something to run. The same walker; clicking a file settles it.
export function chooseProgram(startAt) {
  return pick({ startAt, programs: true, title: 'Choose a program' });
}


function pick({ startAt, programs, title }) {
  return new Promise((resolve) => {
    let chosen = null;
    const list = element('div', { class: 'browser' });
    const here = element('div', { class: 'muted path' });
    const newName = element('input', { placeholder: 'new folder name', size: 18 });

    const dialog = element('dialog', { class: 'confirm wide' }, [
      element('h3', { text: title }),
      here,
      list,
      element('div', { class: 'row' }, [
        // Making a folder belongs to choosing one. Nobody creates a program
        // by naming it.
        programs ? element('span', {}) : element('div', { class: 'inline' }, [
          newName,
          element('button', {
            class: 'action', type: 'button', text: 'Create',
            onclick: async () => {
              if (!newName.value.trim()) return;
              const target = `${chosen}/${newName.value.trim()}`;
              const response = await fetch('/api/directories', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: target }),
              });
              const payload = await response.json();
              if (!response.ok) { here.textContent = payload.error; return; }
              newName.value = '';
              await show(payload.path);
            },
          }),
        ]),
        element('div', { class: 'inline' }, [
          element('button', {
            class: 'action', type: 'button', autofocus: 'autofocus', text: 'Cancel',
            onclick: () => { chosen = null; dialog.close(); },
          }),
          // A program has none: clicking the file is the choice, and "use
          // this folder" would hand back the folder it sits in.
          programs ? null : element('button', {
            class: 'action', type: 'button', text: 'Use this folder',
            onclick: () => dialog.close(),
          }),
        ].filter(Boolean)),
      ]),
    ]);

    async function show(path) {
      let payload;
      try {
        payload = await listing(path, programs);
      } catch (error) {
        here.textContent = error.message;
        return;
      }
      chosen = payload.path;
      here.textContent = payload.path + (payload.writable ? '' : '  (read-only)');
      list.replaceChildren(...[
        payload.parent
          ? element('button', {
              class: 'action quiet', type: 'button', text: '↑ up',
              onclick: () => show(payload.parent),
            })
          : null,
        ...payload.entries.map((entry) => element('button', {
          class: 'action quiet', type: 'button',
          text: entry.kind === 'program'
            ? entry.name
            : entry.name + (entry.writable ? '' : ' (read-only)'),
          onclick: () => {
            if (entry.kind === 'program') { chosen = entry.path; dialog.close(); }
            else show(entry.path);
          },
        })),
        payload.entries.length ? null
          : element('p', { class: 'muted',
                           text: programs ? 'Nothing here that can be run.'
                                          : 'No folders in here.' }),
      ].filter(Boolean));
    }

    // Closing by any means other than "Use this folder" is a refusal, and the
    // answer is settled before the close event so nothing depends on ordering.
    dialog.addEventListener('close', () => { dialog.remove(); resolve(chosen); });
    document.body.append(dialog);
    dialog.showModal();
    show(startAt);
  });
}
