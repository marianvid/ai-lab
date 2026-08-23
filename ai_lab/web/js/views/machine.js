// What this machine is, and how much of it models may use.
//
// One card, because it is one subject. It used to be two — the chip in one and
// the memory in another — and half of what the first showed was memory anyway,
// said differently.
//
// **Nothing here moves on its own.** How much is used, how warm the card is,
// how many requests are running: those are facts about right now, and right
// now is the Gateway page. This says what the machine *is*, so a figure read
// here is still true an hour later.
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

// What each pool is called. On Apple silicon the chip and the rest of the
// machine draw on one pool, which is why there is no card line there — it
// would be the same memory under two names.
const KINDS = {
  card: {
    label: 'VRAM',
    help: 'The whole card. Nothing else on this machine uses it, so all of it '
        + 'is for models. What is free right now is on the Gateway page.',
  },
  machine: {
    label: 'RAM',
    help: 'All of this machine’s own memory. Where the part of a model that '
        + 'does not fit on the card goes, and where everything an engine keeps '
        + 'outside the card lives. Models are offered it less the reserve '
        + 'below.',
  },
  unified: {
    label: 'Unified memory',
    help: 'All of it. Apple silicon shares one pool between the chip and the '
        + 'rest of the machine, so this is both. Models are offered it less '
        + 'the reserve below.',
  },
};

export function machine(settings, refresh) {
  const chip = settings.accelerator || {};
  const memory = settings.memory;
  const rows = [];

  if (chip.name) {
    rows.push(row('Accelerator', chip.name,
                  'What the engines compute on.'));
  }
  // No pools means the machine could not be read — not that it has no room.
  // A row of zeroes there would be a claim, and the wrong one.
  const pools = (memory && memory.pools) || [];
  if (pools.length) {
    pools.forEach((pool, index) => rows.push(poolRow(pool, index === 0)));
    rows.push(reserveRow(memory, refresh));
  }

  if (!rows.length) return null;
  // Heading above the panel rather than inside it, which is how every section
  // on this page is built: the headings line up down the column that way.
  return element('div', { class: 'section machine' }, [
    element('h3', { text: 'Machine' }),
    element('div', { class: 'card' }, rows),
  ]);
}


function poolRow(pool, first) {
  const kind = KINDS[pool.kind === 'unified' ? 'unified' : pool.name]
    || { label: pool.name, help: '' };
  // What this machine has, whole. The reserve is a line of its own
  // directly below, so subtracting it here as well would show a number
  // that matches neither the hardware nor anything printed beside it.
  return row(kind.label, `${Math.round(pool.total_mb)} MB`, kind.help,
             first ? 'memory' : '');
}


// The one thing on this card that is set rather than read, so it looks
// different: a field and a Save, below the figures it comes out of.
function reserveRow(memory, refresh) {
  const field = element('input', {
    type: 'number', class: 'number small', min: '0', step: '1024',
    value: String(Math.round(memory.reserve_mb || 0)),
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


function row(label, value, help, extra = '') {
  return element('div', { class: `row${extra ? ` ${extra}` : ''}` }, [
    element('span', { class: 'muted', text: label, ...(help ? { title: help } : {}) }),
    element('span', { text: value }),
  ]);
}
