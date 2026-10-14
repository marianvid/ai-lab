// Models that never touch the card: subscriptions answered in the cloud, and
// models run on the processor.
//
// They have a page of their own because nothing on the Gateway page applies
// to them. They take no place on the card, never wait for a model to be
// loaded, and each has a queue of its own — one per vendor for the
// subscriptions, one per model for the processor. Reading their numbers next
// to the card's queue would suggest they share it.
//
// Like the Gateway page, this one redraws on a timer: the traffic comes from
// outside the manager and produces no events.

import { api } from '../api.js';
import { element } from '../format.js';
import { showNotice } from '../confirm.js';

const EVERY_MS = 5000;
let timer = null;

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

// A figure, or a dash: a manager older than this page may not send it.
function figure(value, suffix = '') {
  const found = Number(value);
  return Number.isFinite(found) ? `${found}${suffix}` : '—';
}

function words(value) {
  return value === undefined || value === null || value === '' ? '—' : String(value);
}

// One subscription vendor: its models, how many calls it is answering against
// its places, any pause after a rate limit, and today's counts.
function vendor(name, stats) {
  const today = stats.today || {};
  const failures = Object.entries(today.failures || {})
    .map(([label, count]) => `${label} ×${count}`).join(', ');
  const paused = Number(stats.paused_for_s) > 0;
  return section(`Subscription · ${name}`, [
    line('Models', (stats.models || []).join(', ') || '—',
         'Names to put in "model". Answered by the vendor through its CLI, '
         + 'which runs on this machine’s processor.'),
    line('Status', paused ? `paused for ${figure(stats.paused_for_s, ' s')}`
                          : (Number(stats.in_flight) ? 'working' : 'idle'),
         'Paused means the vendor said its allowance is spent: every call to '
         + 'this vendor waits, then tries again.'),
    line('Processing', `${figure(stats.in_flight)} from ${figure(stats.concurrency)}`,
         'Calls running now, against how many may run at once for this vendor. '
         + 'The rest wait their turn here, not in the card’s queue.'),
    line('Default effort', words(stats.default_effort),
         'How long the model thinks when a request names no reasoning_effort.'),
    line('Today', `${figure(today.requests ?? 0)} calls · ${figure(today.ok ?? 0)} answered`,
         'Since midnight on this machine, every model of this vendor together.'),
    failures ? line('Failures today', failures,
                    'Counted by the reason the CLI gave.') : null,
  ].filter(Boolean));
}

// One processor model: whether its server is up, its queue, and the buttons to
// start or stop it by hand.
function processorModel(name, stats, redraw) {
  const running = Boolean(stats.running);
  const act = element('button', {
    class: 'action', text: running ? 'Stop' : 'Start',
    onclick: async () => {
      act.disabled = true;
      try {
        await (running ? api.stopCpuModel(name) : api.startCpuModel(name));
      } catch (error) {
        await showNotice({ title: `Could not ${running ? 'stop' : 'start'} ${name}`,
                           body: error.message });
      }
      redraw();
    },
  });
  const idleStop = Number(stats.idle_stop_s);
  return section(`Processor · ${name}`, [
    line('Status', running ? 'running' : 'stopped',
         'Started by the first request that names it, stopped when nothing '
         + 'has used it for the idle limit. Runs from RAM; the card is never used.'),
    line('Processing', `${figure(stats.in_flight)} from ${figure(stats.places)}`,
         'Requests at the model now, against how many may be at once.'),
    line('Queued', figure(stats.waiting),
         'Requests waiting for a place on this model. Its own queue, not the card’s.'),
    line('Requests', `${figure(stats.requests)} · ${figure(stats.failures)} failed`,
         'Since the manager started.'),
    line('Idle', stats.idle_s === null || stats.idle_s === undefined
      ? 'not used yet' : figure(stats.idle_s, ' s'),
         'Seconds since a request last used it.'),
    line('Stops after', idleStop ? figure(Math.round(idleStop / 60), ' min idle') : 'never',
         'How long it may sit unused before it is stopped to give its RAM back.'),
    element('div', { class: 'row' }, [element('span', {}), act]),
  ]);
}

// How to call them: the same address as every other model, with these paths.
function howToCall(providers, cpu) {
  const base = `${window.location.protocol}//${window.location.host}/v1`;
  const rows = [line('Base URL', base)];
  if (Object.keys(providers).length) {
    rows.push(line('Chat', 'POST /chat/completions',
                   'Subscription models: text only, no streaming.'));
  }
  if (Object.keys(cpu).length) {
    rows.push(line('Embeddings', 'POST /embeddings',
                   'Processor models: send many short texts in one call.'));
  }
  return section('Server address', rows);
}

export function stopRefreshing() {
  if (timer) { window.clearInterval(timer); timer = null; }
}

export async function render(container) {
  stopRefreshing();
  let stats;
  try {
    stats = await api.gateway();
  } catch (error) {
    container.replaceChildren(element('p', { class: 'error', text: error.message }));
    return;
  }
  const providers = stats.providers || {};
  const cpu = stats.cpu_models || {};
  const redraw = () => render(container).catch(() => {});
  const panels = [
    ...Object.entries(providers).map(([name, item]) => vendor(name, item || {})),
    ...Object.entries(cpu).map(([name, item]) => processorModel(name, item || {}, redraw)),
  ];
  container.replaceChildren(element('div', { class: 'lanes' }, [
    element('p', { class: 'muted', text:
      'Models that never use the card: subscriptions answered in the cloud, and '
      + 'models run on the processor. Each has its own queue.' }),
    howToCall(providers, cpu),
    ...(panels.length ? panels
      : [element('p', { class: 'muted', text: 'None configured on this machine.' })]),
  ]));

  const mine = container.firstChild;
  timer = window.setInterval(() => {
    if (container.firstChild !== mine) { window.clearInterval(timer); timer = null; return; }
    redraw();
  }, EVERY_MS);
}
