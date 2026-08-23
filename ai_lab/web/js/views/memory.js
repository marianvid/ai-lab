// How much of this machine models may use.
//
// The card is used whole: nothing else on this machine wants it, so there is
// no setting for it. The machine's own memory is shared with the browser, the
// editor and the operating system, so a part of it is held back — and that is
// the one number here.
//
// It is set as a **reserve**, not as an allowance. A reserve stays right when
// the machine changes: a container grown from 48 GB to 64 GB should offer
// models the extra 16, and an allowance of "40 GB" would sit on it. The figure
// somebody actually wants to read — how much that leaves — is shown beside it
// rather than typed.
//
// On Apple silicon there is one pool. The chip and everything else share the
// same memory, so showing a card figure and a machine figure would be counting
// one thing twice.

import { api } from '../api.js';
import { showNotice } from '../confirm.js';
import { element } from '../format.js';
import { whileWorking } from '../working.js';

const POOLS = {
  card: 'Card',
  machine: 'This machine',
};


export function memory(state, refresh) {
  // No pools means the machine could not be read — not that it has no room.
  // Drawing "0 MB available for models" there would be a claim, and the wrong
  // one.
  if (!state || !state.pools || !state.pools.length) return null;
  const field = element('input', {
    type: 'number', class: 'number small', min: '0', step: '1024',
    value: String(Math.round(reserveOf(state))),
  });
  const save = element('button', {
    class: 'action', text: 'Save', disabled: 'disabled',
    onclick: () => whileWorking(save, 'Saving…', async () => {
      try {
        await api.updateMemory({ reserve_mb: Number(field.value) });
      } catch (error) {
        await showNotice({ title: 'Could not change the reserve',
                           body: error.message });
      }
      refresh();
    }),
  });
  const before = field.value;
  field.addEventListener('input', () => { save.disabled = field.value === before; });

  return element('section', { class: 'card memory' }, [
    element('h3', { text: 'Memory for models' }),
    ...state.pools.map(pool),
    element('div', { class: 'row' }, [
      element('span', { class: 'muted', title:
        'Held back for the operating system and everything else you run. '
        + 'Set as a reserve rather than as an allowance so it stays right if '
        + 'this machine is given more memory.',
        text: 'Reserved for the system (MB)' }),
      element('span', { class: 'inline' }, [field, save]),
    ]),
    element('div', { class: 'row total' }, [
      element('span', { text: 'Available for models' }),
      element('strong', { text: `${Math.round(state.for_models_mb)} MB` }),
    ]),
  ]);
}


function pool(item) {
  const share = item.total_mb
    ? Math.round((100 * item.used_mb) / item.total_mb) : 0;
  return element('div', { class: 'row pool' }, [
    element('span', { class: 'muted', text: POOLS[item.name] || item.name }),
    element('span', {
      text: `${Math.round(item.used_mb)} / ${Math.round(item.total_mb)} MB`,
      title: item.name === 'card' && item.kind === 'dedicated'
        ? `${share}% of the card. All of it is available to models — nothing `
          + 'else on this machine wants it.'
        : `${share}% in use, by models and by everything else. On unified `
          + 'memory this is the same pool the chip draws from.',
    }),
  ]);
}


// The reserve as the server reported it, from whichever pool carries one. Read
// back rather than remembered, so the field shows what is actually in force.
function reserveOf(state) {
  const held = (state.pools || []).find((item) => item.reserve_mb > 0);
  return held ? held.reserve_mb
              : ((state.pools || []).slice(-1)[0] || {}).reserve_mb || 0;
}
