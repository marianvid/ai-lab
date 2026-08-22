// The engine's own output, in a panel pinned to the bottom of the window.
//
// Pinned rather than placed in the page, because the page is long: with a
// dozen entries, a panel that scrolled with them would be somewhere else by
// the time you had read a line of it. This stays where it is and the list
// scrolls behind it.
//
// One panel, one model. Opening it for another replaces what is in it, since
// two of these would leave no room for the list they are about.

import { api } from './api.js';
import { element } from './format.js';

// New lines arrive while a model runs, so the panel asks again on a timer.
// Slower than it could be: reading the journal is a command on the far side,
// and nobody reads a log four times a second.
const EVERY_MS = 3000;

let open = null;          // instance id, or null
let timer = null;
let panel = null;
let output = null;
let onToggle = () => {};

export function watching() {
  return open;
}

// Told when the panel opens or closes, so the row's button can show which it
// is without this file reaching into the list.
export function onPanelChange(callback) {
  onToggle = callback;
}

export function toggleLogs(instance) {
  if (open === instance.id) return closeLogs();
  return openLogs(instance);
}

export function closeLogs() {
  open = null;
  if (timer) { window.clearInterval(timer); timer = null; }
  if (panel) { panel.remove(); panel = null; output = null; }
  onToggle();
}

async function openLogs(instance) {
  open = instance.id;
  if (!panel) build();
  panel.querySelector('.log-title').textContent =
    `${instance.id} — engine output`;
  output.textContent = 'Reading…';
  await refresh();
  if (timer) window.clearInterval(timer);
  timer = window.setInterval(() => { refresh().catch(() => {}); }, EVERY_MS);
  onToggle();
}

// Whether the reader is at the end. Somebody who has scrolled up is reading
// something, and yanking them back to the bottom on every refresh would make
// that impossible.
function atEnd(node) {
  return node.scrollHeight - node.scrollTop - node.clientHeight < 40;
}

async function refresh() {
  if (!open || !output) return;
  const wanted = open;
  let answer;
  try {
    answer = await api.logs(wanted);
  } catch (error) {
    if (open === wanted) output.textContent = error.message;
    return;
  }
  if (open !== wanted || !output) return;      // switched or closed meanwhile
  const wasAtEnd = atEnd(output);
  output.textContent = answer.lines.length
    ? answer.lines.join('\n')
    : (answer.running
        ? 'The engine has printed nothing yet.'
        : 'Nothing running. The panel shows what a model is saying while it runs.');
  if (wasAtEnd) output.scrollTop = output.scrollHeight;
}

function build() {
  output = element('pre', { class: 'log-output' });
  panel = element('aside', { class: 'logpane' }, [
    element('div', { class: 'row log-head' }, [
      element('strong', { class: 'log-title' }),
      element('button', { class: 'action', text: 'Close', onclick: closeLogs }),
    ]),
    output,
  ]);
  document.body.append(panel);
}
