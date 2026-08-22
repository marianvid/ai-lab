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

// The same, for something that is not a measurement. An address and a key are
// things to read and copy, not figures to compare, so they do not take the
// emphasis the numbers get.
function plain(label, value, title) {
  return element('div', { class: 'row', ...(title ? { title } : {}) }, [
    element('span', { class: 'muted', text: label }),
    element('span', { class: 'muted', text: value }),
  ]);
}

// Where to point an agent. Built from the page being looked at rather than
// from the server's idea of itself: the manager may be reached by name, by
// address, or through a tunnel, and the answer has to be the address that
// actually worked to get here.
function address(stats) {
  const base = `${window.location.protocol}//${window.location.host}`;
  const all = stats.shapes.reduce((most, row) =>
    Math.max(most, row.models.length), 0);

  // Two shapes of request are in circulation, and listing only the address
  // would be half the answer — the wrong half for anybody whose tool speaks
  // the other one. Grouped by which models answer each, because that is the
  // part that differs: every engine answers the usual shape, and only some
  // answer Anthropic's.
  const shapes = stats.shapes.map((row) => {
    const everyone = row.models.length === all;
    // The engine, not the models. A list of names grows every time an entry is
    // added and is stale by the next one; "vLLM models" says it once and stays
    // true.
    const who = everyone ? 'every model'
      : `${row.engines.join(' and ')} models`;
    return element('div', { class: 'row tight', title: everyone
      ? 'Every configured model answers this'
      : `${row.models.length} of ${all}: ${row.models.join(', ')}` }, [
      element('span', { class: 'muted', text: row.path }),
      element('span', { class: 'muted', text: who }),
    ]);
  });

  return section('Server address', [
    plain('Base URL', `${base}/v1`),
    ...shapes,
  ]);
}

// What the accelerator says about itself. Memory is the binding constraint on
// a machine that holds one model at a time; the other two arrive in the same
// reading, so they cost nothing extra.
function cardLines(card) {
  if (!card.total_mb) return [];
  const share = Math.round((100 * card.used_mb) / card.total_mb);
  const lines = [
    line('Card memory', `${card.used_mb} / ${card.total_mb} MB`,
         card.kind === 'unified'
           ? 'Apple silicon shares one pool between the chip and everything '
             + 'else, so this is what the engines hold against the machine.'
           : `${share}% of the card`),
  ];
  if (card.busy_percent !== null && card.busy_percent !== undefined) {
    lines.push(line('Card busy', `${card.busy_percent}%`,
                    'A model can hold memory and do nothing. This says whether '
                    + 'it is working.'));
  }
  if (card.temperature_c !== null && card.temperature_c !== undefined) {
    lines.push(line('Temperature', `${card.temperature_c} °C`));
  }
  if (card.ram_total_mb) {
    lines.push(line('System memory', `${card.ram_used_mb} / ${card.ram_total_mb} MB`,
                    'The machine\u2019s own memory. A model split between the '
                    + 'card and here lives in it, and so does everything an '
                    + 'engine keeps outside the card.'));
  }
  return lines;
}

function activity(stats) {
  const status = stats.switching ? 'loading a model'
    : stats.in_flight && stats.waiting ? 'working, with a queue'
    : stats.in_flight ? 'working'
    : stats.waiting ? 'waiting to swap'
    : 'idle';
  return section('Activity', [
    line('Status', status),
    line('Current model', stats.current || 'nothing loaded'),
    line('Processing', `${stats.in_flight} from ${stats.places}`,
         'Requests running together on the model that is loaded, against the '
         + 'number the engine was started to serve.'),
    line('Queue size', String(stats.waiting),
         'Requests waiting for a model that is not loaded.'),
    line('Requests per minute', String(stats.requests_per_minute),
         'In the last sixty seconds.'),
    line('Time to first token', `${stats.average_first_token_s} s`,
         'Averaged over requests that asked for streaming. Without it an '
         + 'engine sends nothing until the answer is finished, so its first '
         + 'byte is the whole generation.'),
    line('Switches', String(stats.switches)),
    line('Average switch', `${stats.average_switch_s} s`),
    line('Time spent switching', `${stats.switching_share}%`,
         'Of the time this was working — answering or loading — how much went '
         + 'on loading.'),
    ...cardLines(stats.card || {}),
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
    ['first_byte_s', 'Max TTFT', stats.first_byte_s, 's',
     'How long to wait for an engine to start answering. It covers reading the '
     + 'prompt — and the whole answer for a request that did not ask for '
     + 'streaming, since such an engine sends nothing at all until it has '
     + 'finished writing. A large prompt on a slow machine is the case to size '
     + 'this for.'],
    ['between_bytes_s', 'Max idle time', stats.between_bytes_s, 's',
     'How long a silence in the middle of an answer means the engine has '
     + 'stopped rather than slowed. At the slowest generation measured here, '
     + '17 tokens a second, the gap between them is 59 milliseconds. It only '
     + 'applies to a request that asked for streaming: without it the answer '
     + 'arrives in one piece and there are no gaps to measure.'],
    ['max_waiting', 'Max queue size', stats.max_waiting, '',
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

  // Nothing to save until something is different. A button that is always
  // ready invites a press that does nothing, and then you cannot tell whether
  // the last press took.
  const asShown = () => {
    const now = {};
    inputs.forEach((input, key) => { now[key] = Number(input.value); });
    return now;
  };
  const unchanged = () => fields.every(([key, , value]) =>
    Number(asShown()[key]) === Number(value));

  const save = element('button', {
    class: 'action', text: 'Save', disabled: 'disabled',
    onclick: async () => {
      const changes = asShown();
      save.disabled = true;
      try {
        await api.updateGateway(changes);
        setStatus('Gateway limits saved', 'ok');
      } catch (error) {
        setStatus(error.message, 'error');
      }
      redraw();
    },
  });

  inputs.forEach((input) => {
    input.addEventListener('input', () => { save.disabled = unchanged(); });
  });

  return section('Limits', [
    ...rows,
    element('div', { class: 'row' }, [element('span', {}), save]),
  ]);
}


// The queue, as the order it will actually be served in.
//
// It is served in order and requests next to each other wanting the same model
// go in together, so it is not a list of requests — it is a list of turns. Read
// down it and you have the schedule of model changes that is about to happen,
// which is the thing worth knowing before it does.
function coming(stats) {
  const runs = stats.queue_runs || [];
  const rows = [];
  if (stats.current || stats.in_flight) {
    rows.push(element('div', { class: 'row tight now' }, [
      element('span', { text: stats.current || 'nothing loaded' }),
      element('span', { class: 'muted', text: stats.in_flight
        ? `${stats.in_flight} running` : 'idle' }),
    ]));
  }
  runs.forEach((run, index) => {
    rows.push(element('div', { class: 'row tight',
                               title: `longest has waited ${run.longest_wait_s} s` }, [
      element('span', { class: 'swap',
                        text: `${index === 0 ? '→ ' : '   '}${run.instance_id}` }),
      element('span', { class: 'muted', text: `${run.requests}` }),
    ]));
  });
  if (!runs.length) {
    rows.push(element('p', { class: 'muted', text: 'Nothing waiting.' }));
  }
  return section('Queue', rows);
}

export function stopRefreshing() {
  if (timer) { window.clearInterval(timer); timer = null; }
}

export async function render(container) {
  // One timer at a time. Leaving the old one running would make the page
  // redraw twice as often for every visit to this tab.
  stopRefreshing();

  let stats;
  try {
    stats = await api.gateway();
  } catch (error) {
    container.replaceChildren(element('p', { class: 'error', text: error.message }));
    return;
  }

  // The grid is its own element rather than the shared page container: a
  // class put on that one stays behind when another tab is opened, and the
  // Models and Library pages inherited a layout meant for this one.
  container.replaceChildren(element('div', { class: 'gateway-grid' }, [
    element('div', { class: 'at-activity' }, activity(stats)),
    element('div', { class: 'at-address' }, address(stats)),
    element('div', { class: 'at-limits' }, limits(stats, () => render(container))),
    element('div', { class: 'at-queue' }, coming(stats)),
  ]));

  // window.setInterval rather than the bare global, so closing the page
  // cancels it. A timer that outlives the window keeps the whole thing alive.
  const mine = container.firstChild;
  timer = window.setInterval(() => {
    if (container.firstChild !== mine) { window.clearInterval(timer); timer = null; return; }
    render(container).catch(() => {});
  }, EVERY_MS);
}
