// Space that can be reclaimed without touching the model library.

import { api } from '../api.js';
import { confirmDestructive, showNotice } from '../confirm.js';
import { bytes, element } from '../format.js';
import { whileWorking } from '../working.js';
import { versions } from './versions.js';


function reclaimable(state, refresh) {
  // Older managers returned configured locations even when they did not
  // exist. Filter them here too, so upgrading the page immediately removes a
  // stale row while server and browser versions overlap during deployment.
  const rows = (state.items || []).filter((item) => item.exists !== false)
    .map((item) => element('div', { class: 'row' }, [
    element('div', { class: 'grow' }, [
      element('strong', { text: item.name }),
      item.description
        ? element('div', { class: 'muted', text: item.description }) : null,
      element('div', { class: 'path muted', text: item.path }),
    ].filter(Boolean)),
    element('span', { class: 'muted', text: bytes(item.size_bytes) }),
    element('button', {
      class: 'action danger', text: item.kind === 'cache' ? 'Clear' : 'Delete',
      onclick: (event) => whileWorking(event.target, 'Clearing…', async () => {
        const yes = await confirmDestructive({
          title: `${item.kind === 'cache' ? 'Clear' : 'Delete'} ${item.name}?`,
          body: `This frees ${bytes(item.size_bytes)}. It does not remove models. `
              + (item.kind === 'cache'
                ? 'Files needed again will be rebuilt or downloaded again.'
                : 'This leftover will be removed permanently.'),
          confirmLabel: item.kind === 'cache' ? 'Clear' : 'Delete',
        });
        if (!yes) return;
        try {
          await api.clearStorage(item.id);
        } catch (error) {
          await showNotice({ title: `Could not clear ${item.name}`, body: error.message });
        }
        refresh();
      }),
    }),
  ].filter(Boolean)));

  return element('div', { class: 'section' }, [
    element('h3', {}, [
      element('span', { text: 'Caches and leftovers' }),
      state.recoverable_bytes
        ? element('span', { class: 'pill', text: `${bytes(state.recoverable_bytes)} recoverable` })
        : null,
    ].filter(Boolean)),
    element('div', { class: 'card' }, rows.length ? rows : [
      element('div', { class: 'muted', text: 'Nothing reclaimable is present.' }),
    ]),
  ]);
}


function rollbacks(installs, engines, refresh) {
  const choices = installs.filter((item) => (item.environments || []).length > 1);
  return element('div', { class: 'section' }, [
    element('h3', { text: 'Engine rollback versions' }),
    ...(choices.length ? choices.map((state) => {
      const engine = engines.get(state.engine) || { id: state.engine, name: state.engine };
      return element('div', { class: 'card' }, [
        element('strong', { text: engine.name }),
        versions(state, engine, refresh),
      ]);
    }) : [element('div', { class: 'card muted',
                           text: 'No engine has a rollback version at the moment.' })]),
  ]);
}


export async function render(container) {
  const refresh = () => render(container);
  const [storage, installs, builds, settings] = await Promise.all([
    api.storage(), api.allInstalls(), api.builds(), api.settings(),
  ]);
  const engines = new Map((settings.engines || []).map((item) => [item.id, item]));
  container.replaceChildren(element('div', { class: 'columns' }, [
    reclaimable(storage, refresh),
    rollbacks([...installs, ...builds], engines, refresh),
  ]));
}
