// Entry point: draws the tabs, renders the selected view, keeps them fresh.
//
// Views are refreshed on a timer rather than on every event, because the
// event stream carries progress — which the runtime view draws directly — and
// a full redraw would fight with it.

import { api } from './api.js';
import { onChange, startEventStream } from './events.js';
import { element, pageTitle } from './format.js';
import { installTheme } from './theme.js';

import { render as renderSettings } from './views/settings.js';
import { render as renderLibrary } from './views/library.js';
import { render as renderRuntime } from './views/runtime.js';
import { render as renderGateway } from './views/gateway.js';

// "Models" is the list you run; "Library" is what is on disk. They were both
// called models before, which was the same confusion as calling a row a slot.
// Downloading, listing and deleting are one subject seen from three angles, so
// they share the Library page rather than making you know which tab a model is
// on before you can act on it.
//
// Each view says which kinds of change concern it. Nothing is drawn on a
// timer: a page where nothing is happening stays exactly as it is, including
// whatever is half-typed in it.
const VIEWS = [
  { id: 'runtime', label: 'Models', render: renderRuntime,
    topics: ['instances', 'models'] },
  { id: 'library', label: 'Library', render: renderLibrary,
    topics: ['models', 'downloads'] },
  // The one address an agent talks to. Its numbers come from traffic the
  // manager only forwards, which produces no events, so this view keeps
  // itself fresh on a timer of its own rather than waiting to be told.
  { id: 'gateway', label: 'Gateway', render: renderGateway, topics: ['instances'] },
  { id: 'settings', label: 'Settings', render: renderSettings,
    topics: ['engines', 'models'] },
];

const view = document.getElementById('view');
const tabs = document.getElementById('tabs');

installTheme(document.getElementById('theme'));

let current = VIEWS[0];
let pending = null;

async function draw() {
  try {
    await current.render(view);
  } catch (error) {
    view.replaceChildren(element('p', { class: 'error', text: error.message }));
  }
}

function select(entry) {
  current = entry;
  [...tabs.children].forEach((button) =>
    button.classList.toggle('active', button.dataset.id === entry.id));
  draw();
}

// Somebody is in the middle of something: a field has the focus, or a menu is
// open. Redrawing would take it away mid-word, so the redraw waits.
function busy() {
  const active = document.activeElement;
  if (!active || active === document.body) return false;
  return view.contains(active) || document.querySelector('dialog[open]') !== null;
}

function refreshLater() {
  if (pending) return;
  pending = setTimeout(function attempt() {
    if (busy()) { pending = setTimeout(attempt, 1000); return; }
    pending = null;
    draw();
  }, 150);   // a moment's wait, so a burst of notices redraws once
}

onChange((event) => {
  if (current.topics.includes(event.topic)) refreshLater();
});

function drawTabs() {
  tabs.replaceChildren(...VIEWS.map((entry) =>
    element('button', { 'data-id': entry.id, text: entry.label, onclick: () => select(entry) })));
}

async function drawHostSummary() {
  const settings = await api.settings();
  document.title = pageTitle(settings);
  const accelerator = settings.accelerator;
  const memory = accelerator.memory_total_mb
    ? `${Math.round(accelerator.memory_total_mb / 1024)} GB ${accelerator.memory_kind}`
    : 'no accelerator';
  document.getElementById('host-summary').textContent =
    `${settings.host.operating_system || 'unknown OS'} · `
    + `${accelerator.name || 'unknown'} · ${memory} · ${settings.host.supervisor}`;
}

drawTabs();
select(current);
drawHostSummary().catch(() => {});
startEventStream();
