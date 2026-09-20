// Installed models and their storage actions.
import { api } from '../api.js';
import { confirmDestructive, showNotice } from '../confirm.js';
import { whileWorking } from '../working.js';
import { capabilities } from '../icons.js';
import { bytes, element } from '../format.js';

let redraw = () => {};
export function setInstalledRedraw(callback) { redraw = callback; }

async function remove(model) {
  const confirmed = await confirmDestructive({
    title: `Delete ${model.name}?`,
    body: `${model.file_count} file${model.file_count === 1 ? '' : 's'}, `
          + `${bytes(model.size_bytes)}, will be removed from disk. `
          + 'This cannot be undone.',
  });
  if (!confirmed) return;
  try {
    await api.deleteModel(model.id);
  } catch (error) {
    await showNotice({ title: `Could not delete ${model.name}`,
                       body: error.message });
  }
  await redraw();
}

async function move(model) {
  const target = model.storage_tier === 'core' ? 'benchmark' : 'core';
  try {
    await api.moveModel(model.id, target);
  } catch (error) {
    await showNotice({ title: 'Could not move ' + model.name,
                       body: error.message });
  }
  await redraw();
}

export async function refreshLibrary(button) {
  await whileWorking(button, 'Scanning…', async () => {
    try {
      await redraw();
    } catch (error) {
      await showNotice({ title: 'Could not refresh the library',
                         body: error.message });
    }
  });
}

function modelRow(model) {
  const state = model.complete
    ? element('span', { class: 'pill on', text: 'complete' })
    : element('span', {
        class: 'pill', style: 'color:var(--warn);border-color:var(--warn)',
        text: `missing ${model.missing.length}`, title: model.missing.join(', '),
      });
  return element('tr', {}, [
    // On disk there is no entry and so nothing to switch anything off: these
    // are what the weights themselves can do.
    element('td', {
      title: model.description || model.summary || '',
      class: model.description || model.summary ? 'model-curated' : '',
    }, element('span', { class: 'inline ident' }, [
      element('strong', { text: model.name,
                          title: model.description || model.summary || '' }),
      model.description || model.summary
        ? element('span', {
            class: 'model-info', text: 'ⓘ',
            title: model.description || model.summary,
            'aria-label': `About ${model.name}: ${model.description || model.summary}`,
          }) : null,
      ...capabilities(model.capabilities),
    ].filter(Boolean))),
    element('td', { class: 'muted', text: model.format }),
    element('td', { class: 'muted', text: model.task || 'text-generation' }),
    element('td', {}, element('span', {
      class: 'pill', text: model.storage_tier || 'core',
    })),
    element('td', { text: bytes(model.size_bytes) }),
    element('td', { class: 'muted', text: String(model.file_count) }),
    element('td', {}, state),
    element('td', {}, element('span', { class: 'inline model-actions' }, [
      element('button', { class: 'action',
                          text: model.storage_tier === 'core'
                            ? 'Move to benchmark' : 'Move to core',
                          onclick: (event) => whileWorking(
                            event.target, 'Moving…', () => move(model)) }),
      element('button', { class: 'action danger', text: 'Delete',
                          onclick: (event) => whileWorking(
                            event.target, 'Deleting…', () => remove(model)) }),
    ])),
  ]);
}

export function repositorySection(repository, models) {
  const heading = `${repository.name} · ${repository.format}`;
  return element('section', {}, [
    element('h3', { text: heading }),
    models.length
      ? element('table', { class: 'model-library' }, [
          element('colgroup', {}, [
            element('col', { class: 'model-column' }),
            element('col', { class: 'format-column' }),
            element('col', { class: 'task-column' }),
            element('col', { class: 'storage-column' }),
            element('col', { class: 'size-column' }),
            element('col', { class: 'files-column' }),
            element('col', { class: 'state-column' }),
            element('col', { class: 'actions-column' }),
          ]),
          element('thead', {}, element('tr', {}, [
            element('th', { text: 'Model' }),
            element('th', { text: 'Format' }),
            element('th', { text: 'Task' }),
            element('th', { text: 'Storage' }),
            element('th', { text: 'Size' }),
            element('th', { text: 'Files' }),
            element('th', { text: '' }),
            element('th', { text: '' }),
          ])),
          element('tbody', {}, models.map(modelRow)),
        ])
      : element('p', { class: 'muted', text: 'Empty.' }),
  ]);
}
