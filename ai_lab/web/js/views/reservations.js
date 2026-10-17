// The Reservations page (ADR 0005): who holds the card, who waits for it and
// in what order, what ended recently and why, and the settings.
//
// A team reserves the whole card for a run; while it holds it, only its calls
// reach this machine's models. The page refreshes itself on a timer, like the
// Gateway page: reservations change through requests that produce no events.

import { api } from '../api.js';
import { element } from '../format.js';
import { showNotice } from '../confirm.js';

const EVERY_MS = 5000;
let timer = null;

// The settings, shown in minutes; the server keeps seconds.
const SETTINGS = [
  ['heartbeat_timeout_s', 'Silent holder ends after',
   'A holder sends a heartbeat while it works. No heartbeat for this long ends its reservation and frees the card.'],
  ['unused_s', 'Granted but unused ends after',
   'Counted only until the reservation’s first call; after that only the heartbeat counts.'],
  ['default_max_s', 'Longest hold when not asked',
   'How long a reservation may hold the card when it does not say.'],
  ['ceiling_s', 'Longest hold ever',
   'The most a reservation may ask for, extensions included.'],
  ['aging_s', 'Waiting raises priority every',
   'A waiting reservation rises one priority level this often, so nothing waits forever. 0: never.'],
  ['unreserved_wait_s', 'A call without a reservation waits up to',
   'While the card is reserved, a call without a reservation waits this long, then is refused.'],
];

function section(title, children) {
  return element('div', { class: 'section' }, [
    element('h3', { text: title }),
    element('div', { class: 'card' }, children),
  ]);
}

function line(label, value, title) {
  return element('div', { class: 'row', ...(title ? { title } : {}) }, [
    element('span', { class: 'muted', text: label }),
    element('span', { text: value }),
  ]);
}

function minutes(seconds) {
  const value = Number(seconds);
  if (!Number.isFinite(value)) return '—';
  return value < 120 ? `${Math.round(value)} s` : `${Math.round(value / 60)} min`;
}

function since(stamp) {
  const value = Number(stamp);
  return Number.isFinite(value) && value > 0 ? new Date(value * 1000).toLocaleTimeString() : '—';
}

// "ingest @ ct109 · collect · run 20261017-…"
function who(item) {
  return [item.team + (item.instance ? ` @ ${item.instance}` : ''),
          item.purpose, item.run_id ? `run ${item.run_id}` : '']
    .filter(Boolean).join(' · ');
}

function cancelButton(item, redraw) {
  const button = element('button', {
    class: 'action', text: 'Cancel',
    onclick: async () => {
      button.disabled = true;
      try {
        await api.cancelReservation(item.id);
      } catch (error) {
        await showNotice({ title: 'Could not cancel', body: error.message });
      }
      redraw();
    },
  });
  return button;
}

function holderSection(holder, redraw) {
  if (!holder) {
    return section('Holder', [element('p', { class: 'muted', text: 'The card is not reserved.' })]);
  }
  return section('Holder', [
    line('Held by', who(holder)),
    line('Priority', String(holder.priority)),
    line('Since', since(holder.granted_at)),
    line('Time left', minutes(holder.time_left_s),
         'Until its maximum duration; the holder may extend it up to the ceiling.'),
    line('Last heartbeat', since(holder.last_beat)),
    line('First call', holder.first_call_at ? since(holder.first_call_at) : 'not yet'),
    holder.restored ? line('Restored', 'after a manager restart') : null,
    element('div', { class: 'row' }, [element('span', {}), cancelButton(holder, redraw)]),
  ].filter(Boolean));
}

function queueSection(queue, redraw) {
  if (!queue.length) {
    return section('Waiting', [element('p', { class: 'muted', text: 'Nobody is waiting.' })]);
  }
  return section('Waiting', queue.map((item) => element('div', { class: 'row' }, [
    element('span', { text: `${item.position}. ${who(item)}` }),
    element('span', { class: 'inline' }, [
      element('span', { class: 'muted', text:
        `priority ${item.priority}${item.effective_priority !== item.priority
          ? ` → ${item.effective_priority}` : ''} · since ${since(item.requested_at)}` }),
      cancelButton(item, redraw),
    ]),
  ])));
}

function historySection(history) {
  const rows = history.slice(0, 10).map((item) => line(
    `${since(item.ended_at)} · ${who(item)}`, item.reason || 'ended'));
  return section('Recently ended', rows.length ? rows
    : [element('p', { class: 'muted', text: 'Nothing yet.' })]);
}

function settingsSection(settings, redraw) {
  const inputs = {};
  const rows = SETTINGS.map(([key, label, title]) => {
    inputs[key] = element('input', { type: 'number', min: '0', step: '1',
                                     value: String(Math.round(settings[key] / 60)) });
    return element('label', { class: 'row', title }, [
      element('span', { class: 'muted', text: `${label} (min)` }), inputs[key]]);
  });
  const required = element('input', { type: 'checkbox' });
  required.checked = Boolean(settings.required);
  rows.push(element('label', { class: 'row',
    title: 'Refuse every call to this machine’s models that carries no reservation.' }, [
    element('span', { class: 'muted', text: 'Reservation required' }), required]));
  const save = element('button', { class: 'action', text: 'Save', onclick: async () => {
    const changes = { required: required.checked };
    for (const [key] of SETTINGS) changes[key] = Math.round(Number(inputs[key].value) * 60);
    try {
      await api.updateReservationSettings(changes);
    } catch (error) {
      await showNotice({ title: 'Could not save', body: error.message });
    }
    redraw();
  } });
  rows.push(element('div', { class: 'row' }, [element('span', {}), save]));
  return section('Settings', rows);
}

export function stopRefreshing() {
  if (timer) { window.clearInterval(timer); timer = null; }
}

export async function render(container) {
  stopRefreshing();
  let book;
  try {
    book = await api.reservations();
  } catch (error) {
    container.replaceChildren(element('p', { class: 'error', text: error.message }));
    return;
  }
  const redraw = () => render(container).catch(() => {});
  container.replaceChildren(element('div', { class: 'lanes' }, [
    element('p', { class: 'muted', text:
      'A team reserves the whole card for a run. While it holds it, only its calls reach '
      + 'this machine’s models; the others wait, highest priority first. Load and Unload '
      + 'on the Models page wait too — cancel the reservation here first.' }),
    holderSection(book.holder, redraw),
    queueSection(book.queue || [], redraw),
    historySection(book.history || []),
    settingsSection(book.settings || {}, redraw),
  ]));
  // The settings form must not be redrawn under the person typing in it.
  timer = window.setInterval(() => {
    if (!container.contains(document.activeElement)) redraw();
  }, EVERY_MS);
}
