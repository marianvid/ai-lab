// The one address an agent talks to, and what it has been doing.
//
// The numbers here answer one question: is the workflow running, or is it
// spending its time changing models? A switch is an unload, a wait for the
// card to go quiet, and a load — tens of seconds. When switches are close to
// requests, almost every step changes model, and reordering the workflow so
// that steps using the same model sit together is worth more than any setting
// on this page.
//
// This is the only view that redraws on a timer. Everywhere else a change
// comes from something the server did and arrives as an event; here the
// traffic comes from outside the manager entirely and produces no events at
// all, so a page that never redrew would sit at whatever the numbers were when
// the tab was opened.

import { api } from '../api.js';
import { element, seconds } from '../format.js';
import { setStatus } from '../status.js';

// Slower than it looks like it should be, on purpose. Reading these numbers
// means asking systemd about every configured instance and probing each one
// that is up: measured on the container with eleven instances, one call costs
// about 125 ms and spawns 28 processes. And the thing being watched moves far
// more slowly than that — a model switch takes between twelve and forty-five
// seconds. A two-second poll was finer than its own subject and cost fourteen
// processes a second for the privilege.
const EVERY_MS = 5000;
let timer = null;

// A heading above its panel, the same shape the Settings page uses.
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

// Where to point an agent. Built from the page being looked at rather than
// from the server's idea of itself: the manager may be reached by name, by
// address, or through a tunnel, and the answer has to be the address that
// actually worked to get here.
function address() {
  const base = `${window.location.protocol}//${window.location.host}/v1`;
  return section('Where to point an agent', [
    line('Base URL', base),
    line('API key', 'not checked — any value will do'),
    element('p', { class: 'muted',
                   text: 'Name any configured model in the request. If it is '
                         + 'not the one on the card, it is loaded first and '
                         + 'that one request simply takes longer.' }),
  ]);
}

function state(stats) {
  const answering = stats.in_flight
    ? `${stats.in_flight} of ${stats.places} places in use`
    : stats.switching ? 'loading' : 'idle';
  const queue = stats.waiting
    ? `${stats.waiting} waiting, longest ${stats.longest_wait_s} s`
    : 'nobody waiting';
  return section('Right now', [
    line('On the card', stats.current || 'nothing loaded'),
    line('Answering', answering,
         'Requests to the model on the card run together, up to the number '
         + 'the engine was started to serve. Requests for another model wait.'),
    line('Queue', queue,
         'Waiting for a model that is not loaded. The oldest decides which is '
         + 'loaded next.'),
    ...stats.waiting_for.map((row) => line(
      `  wanting ${row.instance_id}`,
      `${row.waiting}, longest ${row.longest_wait_s} s`)),
    stats.last_error
      ? element('p', { class: 'error', text: `Last error: ${stats.last_error}` })
      : null,
  ].filter(Boolean));
}

// The three numbers that decide how patient the front door is, and how much it
// will hold. Editable here rather than on the Settings page: they are about
// the gateway, and they sit beside the figures you would read before deciding
// to change them.
function limits(stats, redraw) {
  const fields = [
    ['first_byte_s', 'Wait for the first byte', stats.first_byte_s, 's',
     'How long to wait for an engine to start answering. It covers reading '
     + 'the prompt — and the whole answer for a request that did not ask for '
     + 'streaming, since such an engine sends nothing until it has finished. '
     + 'A large prompt on a slow machine is the case to size this for.'],
    ['between_bytes_s', 'Wait between bytes', stats.between_bytes_s, 's',
     'How long a silence in the middle of an answer means the engine has '
     + 'stopped rather than slowed. At the slowest generation measured here, '
     + '17 tokens a second, the gap between them is 59 milliseconds.'],
    ['max_waiting', 'Requests held in the queue', stats.max_waiting, '',
     'How many may wait for a model at once. Beyond it a request is refused '
     + 'rather than queued, because each one waiting occupies a thread. A '
     + 'workflow that hits this is asking for more at once than one card can '
     + 'answer.'],
  ];

  const inputs = new Map();
  const rows = fields.map(([key, label, value, unit, help]) => {
    const input = element('input', {
      type: 'number', value: String(value), min: '1', size: 8,
    });
    inputs.set(key, input);
    return element('label', { class: 'field explained', title: help }, [
      element('span', {}, element('span', { text: unit ? `${label} (${unit})` : label })),
      input,
    ]);
  });

  const save = element('button', {
    class: 'action', text: 'Save',
    onclick: async () => {
      const changes = {};
      inputs.forEach((input, key) => { changes[key] = Number(input.value); });
      save.disabled = true;
      try {
        await api.updateGateway(changes);
        setStatus('Gateway limits saved', 'ok');
      } catch (error) {
        setStatus(error.message, 'error');
      }
      save.disabled = false;
      redraw();
    },
  });

  return section('Limits', [
    ...rows,
    element('div', { class: 'row' }, [element('span', {}), save]),
    element('p', { class: 'muted',
                   text: 'Limits of safety, not of patience: in normal work '
                         + 'nothing comes near them. They take effect at once.' }),
  ]);
}

// The ratio is the number worth reading, so it is stated rather than left for
// the reader to divide two figures in their head.
function traffic(stats) {
  const share = stats.requests
    ? Math.round((stats.switches / stats.requests) * 100) : 0;
  const verdict = !stats.requests ? 'nothing yet'
    : share >= 60 ? 'changing model on most requests — reorder the workflow so '
                    + 'steps sharing a model run together'
    : share >= 25 ? 'changing model fairly often'
    : 'mostly staying on one model';
  return section('Traffic', [
    line('Requests', String(stats.requests)),
    line('Switches', `${stats.switches} (${share}% of requests)`),
    line('Verdict', verdict),
    line('Average wait before answering', `${stats.average_wait_s} s`,
         'From the request arriving to the model being ready for it: the '
         + 'queue in front of it, plus a switch if one was needed.'),
    line('Average switch', `${stats.average_switch_s} s`),
    line('Total spent switching', `${stats.total_switch_s} s`),
  ]);
}

function recent(stats) {
  if (!stats.recent || !stats.recent.length) {
    return section('Recent switches', [
      element('p', { class: 'muted', text: 'No switches yet.' }),
    ]);
  }
  return section('Recent switches', stats.recent.map((entry) => {
    const out = entry.unloaded && entry.unloaded.length
      ? entry.unloaded.join(', ') : 'nothing';
    // A tidy-up is not a switch: the model asked for was already up, and
    // something else was unloaded from beside it. Saying "loaded X" there
    // would claim a load that never happened.
    const what = entry.tidied
      ? `${out} unloaded from beside ${entry.loaded}`
      : `${entry.loaded} in, ${out} out`;
    return element('div', { class: 'row' }, [
      element('span', { text: what }),
      element('span', { class: 'muted',
                        text: entry.tidied ? 'tidy-up'
                              : `${entry.took_s} s · load ${seconds(entry.load_ms)}` }),
    ]);
  }));
}

export async function render(container) {
  // One timer at a time. Leaving the old one running would make the page
  // redraw twice as often for every visit to this tab.
  if (timer) { window.clearInterval(timer); timer = null; }

  let stats;
  try {
    stats = await api.gateway();
  } catch (error) {
    container.replaceChildren(element('p', { class: 'error', text: error.message }));
    return;
  }

  container.replaceChildren(...[
    address(), state(stats), traffic(stats), limits(stats, () => render(container)),
    recent(stats),
  ]);

  // Stop as soon as the tab is left: `container` is emptied and refilled by
  // whichever view is drawn next, so its contents no longer being ours is the
  // signal that this view is gone.
  // window.setInterval rather than the bare global, so closing the page
  // cancels it. A timer that outlives the window keeps the whole thing alive.
  const mine = container.firstChild;
  timer = window.setInterval(() => {
    if (container.firstChild !== mine) { window.clearInterval(timer); timer = null; return; }
    render(container).catch(() => {});
  }, EVERY_MS);
}
