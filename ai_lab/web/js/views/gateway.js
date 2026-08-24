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
import { showNotice } from '../confirm.js';

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
// the binding constraint on how many models fit; the other two arrive in the same
// reading, so they cost nothing extra.
// What the machine has room for, and how warm it is.
//
// The pools come from the same reading that decides whether a model may be
// loaded, so this page cannot disagree with the thing that says no. On a
// dedicated card there are two — the card, used whole, and the machine, which
// keeps a reserve. On Apple silicon there is one, because the chip and
// everything else draw on the same memory.
const POOL_NAMES = { card: 'Card mem', machine: 'System mem' };

// A figure, or a dash. Every number here comes from a manager that may be
// older than this file — they are not replaced in the same instant — and
// `String(undefined)` reaches the screen as the word "undefined". A dash says
// "no answer", which is true, and is what the rest of this interface uses.
function figure(value, suffix = '') {
  const found = Number(value);
  return Number.isFinite(found) ? `${found}${suffix}` : '—';
}

function memoryLines(memory, card) {
  const lines = [];
  if (card && card.temperature_c !== null && card.temperature_c !== undefined) {
    lines.push(line('GPU temp', figure(card.temperature_c, ' °C')));
  }
  (memory && memory.pools ? memory.pools : []).forEach((pool) => {
    const share = pool.total_mb
      ? Math.round((100 * pool.used_mb) / pool.total_mb) : 0;
    lines.push(line(POOL_NAMES[pool.name] || pool.name,
                    `${figure(pool.used_mb)} / ${figure(pool.total_mb)} MB`,
                    pool.kind === 'unified'
                      ? `${share}% in use, by models and by everything else. `
                        + 'Apple silicon shares one pool between the chip and '
                        + 'the rest of the machine.'
                      : pool.name === 'card'
                        ? `${share}% of the card. All of it is for models.`
                        : `${share}% in use. ${pool.reserve_mb} MB of the rest `
                          + 'is held back for the machine itself.'));
  });
  if (memory && memory.pools && memory.pools.length) {
    lines.push(line('Available mem', figure(memory.available_mb, ' MB'),
                    'What is free, less what is held back for the machine. A '
                    + 'model has to fit in one pool, so this total is the '
                    + 'ceiling rather than the test.'));
  }
  return lines;
}

function activity(stats) {
  const loaded = stats.loaded || [];
  const status = stats.switching ? 'loading a model'
    : stats.in_flight && stats.waiting ? 'working, with a queue'
    : stats.in_flight ? 'working'
    : stats.waiting ? 'waiting for room'
    : 'idle';
  const places = loaded.reduce((total, item) => total + (Number(item.places) || 0), 0);
  return section('Activity', [
    line('Status', status),
    line('Loaded', figure(loaded.length),
         'How many models are on the machine. Which ones, and what each is '
         + 'doing, is below.'),
    line('Processing', `${figure(stats.in_flight)} from ${figure(places)}`,
         'Requests running right now, against the places every loaded model '
         + 'offers between them. Places are the engine\u2019s own number and '
         + 'differ per model.'),
    line('Queue size', figure(stats.waiting),
         'Requests waiting for a model that is not loaded.'),
    line('Requests per minute', figure(stats.requests_per_minute),
         'In the last sixty seconds.'),
    line('Switches', figure(stats.switches)),
    line('Time spent switching', figure(stats.switching_share, '%'),
         'Of the time this was working — answering or loading — how much went '
         + 'on loading.'),
    ...memoryLines(stats.memory, stats.card),
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
        // Nothing is said on success: the fields keep the saved values and
        // Save goes back to sleep, which is the confirmation.
      } catch (error) {
        await showNotice({ title: 'Could not save the gateway limits',
                           body: error.message });
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
  const loaded = stats.loaded || [];
  const rows = [];

  // What is on the machine, one row each: what it is answering now and how
  // many are queued for it. Both, because they say different things — running
  // is whether it is busy, waiting is whether there is pressure on it.
  loaded.forEach((item) => {
    rows.push(element('div', { class: 'row tight now' }, [
      element('span', { class: 'swap', text: `→ ${item.instance_id}` }),
      element('span', { class: 'muted', text:
        `${figure(item.in_flight)} running · ${figure(item.waiting)} waiting` }),
    ]));
  });
  if (!loaded.length) {
    rows.push(element('div', { class: 'row tight now' },
                      element('span', { class: 'muted', text: 'nothing loaded' })));
  }

  // Everything queued for a model that is already there. It is waiting for a
  // place, not for a load.
  const onLoaded = loaded.reduce((total, item) => total + (Number(item.waiting) || 0), 0);
  rows.push(element('div', { class: 'row tight tally' }, [
    element('span', { class: 'muted', text: 'Waiting' }),
    element('span', { class: 'muted', text: figure(onLoaded) }),
  ]));

  // The next model that has to be loaded, and everything held up behind it.
  // Read together they are the cost of the change: this many requests are
  // stopped until that one is on.
  const next = runs[0];
  const behind = next
    ? runs.reduce((total, run) => total + (Number(run.requests) || 0), 0)
      - (Number(next.requests) || 0)
    : 0;
  rows.push(element('div', { class: 'row tight next',
                             ...(next ? { title: 'longest has waited '
                                                 + `${next.longest_wait_s} s` } : {}) }, [
    element('span', { class: 'muted', text: 'Next change' }),
    next
      ? element('span', { class: 'swap grow', text: `${next.instance_id}  ←` })
      : element('span', { class: 'muted grow', text: '—' }),
    next
      ? element('span', { class: 'muted', text: `${figure(next.requests)} waiting` })
      : null,
  ].filter(Boolean)));
  rows.push(element('div', { class: 'row tight tally' }, [
    element('span', { class: 'muted', text: 'Remaining' }),
    element('span', { class: 'muted', text: figure(behind) }),
  ]));
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
