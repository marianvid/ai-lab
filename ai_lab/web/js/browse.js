// Picking a folder on the machine the server runs on.
//
// A web page cannot open a file dialog there — the browser only knows about
// the computer it is running on, which may not be the same one. So the server
// lists directories and this walks through them.

import { confirmDestructive } from './confirm.js';
import { element } from './format.js';

async function listing(path) {
  const query = path ? `?path=${encodeURIComponent(path)}` : '';
  const response = await fetch(`/api/browse${query}`);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || 'Could not read that folder');
  return payload;
}

export function chooseFolder(startAt) {
  return new Promise((resolve) => {
    let chosen = null;
    const list = element('div', { class: 'browser' });
    const here = element('div', { class: 'muted path' });
    const newName = element('input', { placeholder: 'new folder name', size: 18 });

    const dialog = element('dialog', { class: 'confirm wide' }, [
      element('h3', { text: 'Choose a folder' }),
      here,
      list,
      element('div', { class: 'row' }, [
        element('div', { class: 'inline' }, [
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
          element('button', {
            class: 'action', type: 'button', text: 'Use this folder',
            onclick: () => dialog.close(),
          }),
        ]),
      ]),
    ]);

    async function show(path) {
      let payload;
      try {
        payload = await listing(path);
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
          text: entry.name + (entry.writable ? '' : ' (read-only)'),
          onclick: () => show(entry.path),
        })),
        payload.entries.length ? null
          : element('p', { class: 'muted', text: 'No folders in here.' }),
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
