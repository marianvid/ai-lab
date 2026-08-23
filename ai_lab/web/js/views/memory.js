// How much memory this machine has for models.
//
// A capacity, not a reading. What is *free right now* is on the Gateway page,
// where the rest of what is happening lives; this says how big the machine is,
// and it does not move when a model loads. Two different questions, and mixing
// them makes a number that answers neither.
//
// One line per kind of memory, because they are not interchangeable — a model
// cannot spill from the card into the machine unless it was started to. And
// one line for the only thing here that is set rather than read.
//
// The card is listed at its full size: nothing else on this machine wants it,
// so all of it is for models. The machine's own memory is shared with the
// browser, the editor and the operating system, so it is listed less what is
// held back for them.
//
// Held back as a **reserve** rather than as an allowance. A reserve stays right
// when the machine changes: a container grown from 48 GB to 64 GB should offer
// models the extra 16, and an allowance of "40 GB" would sit on it.

import { api } from '../api.js';
import { showNotice } from '../confirm.js';
import { element } from '../format.js';
import { whileWorking } from '../working.js';

// What each pool is called, and what it is worth saying about it. On Apple
// silicon the chip and the rest of the machine draw on one pool, which is why
// there is no card line there — it would be the same memory under two names.
const KINDS = {
  card: {
    label: 'VRAM',
    help: 'The whole card. Nothing else on this machine uses it, so all of it '
        + 'is for models. What is free right now is on the Gateway page.',
  },
  machine: {
    label: 'RAM',
    help: 'The machine’s own memory, less what is held back below. Where '
        + 'the part of a model that does not fit on the card goes, and where '
        + 'everything an engine keeps outside the card lives.',
  },
  unified: {
    label: 'Unified memory',
    help: 'Apple silicon shares one pool between the chip and the rest of the '
        + 'machine, so this is both. Shown less what is held back below.',
  },
};


export function memory(state, refresh) {
  // No pools means the machine could not be read — not that it has no room.
  // A row of zeroes there would be a claim, and the wrong one.
  if (!state || !state.pools || !state.pools.length) return null;

  return element('section', { class: 'card memory' }, [
    element('h3', { text: 'Available memory' }),
    element('div', { class: 'rows' }, [
      ...state.pools.map(kindRow),
      reserveRow(state, refresh),
    ]),
  ]);
}


function kindRow(pool) {
  const kind = KINDS[pool.kind === 'unified' ? 'unified' : pool.name] || {
    label: pool.name, help: '',
  };
  return element('div', { class: 'row' }, [
    element('span', { class: 'muted', text: kind.label, title: kind.help }),
    element('span', { text: `${Math.round(pool.capacity_mb)} MB` }),
  ]);
}


// The one thing on this card that is set rather than read, so it looks
// different: a field and a Save, below the figures it explains.
function reserveRow(state, refresh) {
  const field = element('input', {
    type: 'number', class: 'number small', min: '0', step: '1024',
    value: String(Math.round(state.reserve_mb || 0)),
  });
  const before = field.value;
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
  field.addEventListener('input', () => { save.disabled = field.value === before; });

  return element('div', { class: 'row reserve' }, [
    element('span', {
      class: 'muted', text: 'Reserved for the system',
      title: 'Kept for the operating system and everything else you run, so '
           + 'models are never offered it. Taken out of the figures above.',
    }),
    element('span', { class: 'inline' }, [field, save]),
  ]);
}
