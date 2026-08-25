// The installed versions of an engine, whether compiled or packaged.
//
// A new version goes in a *new* folder, so the one that works is never written
// over — which is the only reason going back is possible at all. Packaged
// engines may disappear from an index; source engines may take a long time to
// compile again. Neither should lose the known-good version during an update.
//
// So this shows the folders. One is in use; the others are the way back. The
// old one is deleted when somebody decides the new one has proved itself,
// which is a judgement no timer can make — there is no automatic tidying here
// and there should not be.

import { api } from '../api.js';
import { confirmDestructive, showNotice } from '../confirm.js';
import { bytes, element } from '../format.js';
import { whileWorking } from '../working.js';


export function versions(state, engine, refresh) {
  if (!state || !state.environments || !state.environments.length) return null;
  const spare = state.environments.filter((item) => !item.active);
  return element('section', { class: 'versions' }, [
    element('h4', {}, [
      element('span', { text: 'Installed versions' }),
      spare.length
        ? element('span', {
            class: 'pill', text: `${bytes(state.spare_bytes)} recoverable`,
            title: 'Held by the versions not in use — they are the way back '
                 + 'from an update that goes wrong',
          })
        : null,
    ].filter(Boolean)),
    element('div', { class: 'stack' },
      state.environments.map((item) => row(item, state, engine, refresh))),
  ]);
}


function row(environment, state, engine, refresh) {
  const busy = state.state === 'running';
  const source = state.kind === 'source';
  return element('div', { class: `row version${environment.active ? ' active' : ''}` }, [
    element('div', { class: 'inline ident' }, [
      element('strong', { text: environment.version }),
      environment.active
        ? element('span', { class: 'pill on', text: 'in use' })
        : null,
      element('span', { class: 'muted', text: bytes(environment.size_bytes) }),
    ].filter(Boolean)),
    element('div', { class: 'inline' }, [
      environment.active ? null : element('button', {
        class: 'action', text: 'Use this one',
        title: 'Point the engine at this version. Nothing is downloaded and '
             + 'nothing is deleted.',
        ...(busy ? { disabled: 'disabled' } : {}),
        onclick: (event) => whileWorking(event.target, 'Switching…', async () => {
          try {
            if (source) await api.activateBuild(engine.id, environment.name);
            else await api.activateInstall(engine.id, environment.name);
          } catch (error) {
            await showNotice({ title: `Could not switch to ${environment.version}`,
                               body: error.message });
          }
          refresh();
        }),
      }),
      environment.active ? null : element('button', {
        class: 'action danger', text: 'Delete',
        title: 'Free the space. This is the version you would go back to.',
        ...(busy ? { disabled: 'disabled' } : {}),
        onclick: (event) => whileWorking(event.target, 'Deleting…',
                                         () => remove(environment, state, engine, refresh)),
      }),
    ].filter(Boolean)),
  ]);
}


async function remove(environment, state, engine, refresh) {
  const source = state.kind === 'source';
  const confirmed = await confirmDestructive({
    title: `Delete ${engine.name} ${environment.version}?`,
    body: `This frees ${bytes(environment.size_bytes)}. It is the version the `
        + 'engine would go back to if the one in use turned out badly. '
        + (source
          ? 'It can be compiled again while its source tag remains available.'
          : 'It may not be reinstallable — a build that has left its index '
            + 'cannot be fetched again.'),
    confirmLabel: 'Delete',
  });
  if (!confirmed) return;
  try {
    if (source) await api.removeBuild(engine.id, environment.name);
    else await api.removeInstall(engine.id, environment.name);
  } catch (error) {
    await showNotice({ title: `Could not delete ${environment.version}`,
                       body: error.message });
  }
  refresh();
}
