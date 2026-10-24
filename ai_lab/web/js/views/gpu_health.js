// The GPU health page (docs/gpu-health.md): the card now, the alarms in force,
// its reference values and their trend, the last day as a chart, the events.
//
// The monitor reads the card every 15 seconds in the manager; this page only
// shows what it found, and refreshes itself on a timer like the Gateway page.

import { api } from '../api.js';
import { element } from '../format.js';
import { showNotice } from '../confirm.js';

const EVERY_MS = 15000;
const SVG = 'http://www.w3.org/2000/svg';
let timer = null;

// NVML's reasons for holding the clocks down. Power capping is normal work.
const THROTTLE = { 0x1: 'idle', 0x2: 'application clocks', 0x4: 'power cap', 0x8: 'HW slowdown',
                   0x10: 'sync boost', 0x20: 'SW thermal', 0x40: 'HW thermal',
                   0x80: 'HW power brake', 0x100: 'display clock' };

const NOW_ROWS = [
  ['temp_gpu', 'Chip temperature', (v) => `${v} °C`],
  ['temp_mem', 'Memory temperature', (v) => `${v} °C`],
  ['temp_hotspot', 'Hotspot', (v) => `${v} °C`],
  ['power_w', 'Power', (v, row) => `${v} W of ${row.power_limit_w ?? '—'} W`],
  ['fan_pct', 'Fan', (v) => `${v} %`],
  ['clock_sm', 'Clocks (chip / memory)', (v, row) => `${v} / ${row.clock_mem ?? '—'} MHz`],
  ['util_pct', 'Use', (v) => `${v} %`],
  ['throttle', 'Clocks held down by', (v) => throttleText(v)],
  ['pcie_gen', 'PCIe link', (v, row) => `Gen ${v} x${row.pcie_width ?? '—'}`],
  ['pcie_replays', 'PCIe replays (since driver start)', (v) => String(v)],
];

export function throttleText(mask) {
  const names = Object.entries(THROTTLE).filter(([bit]) => mask & Number(bit))
    .map(([, name]) => name);
  return names.length ? names.join(', ') : 'nothing';
}

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

function empty(text) {
  return element('p', { class: 'muted', text });
}

function when(stamp) {
  const value = Number(stamp);
  return Number.isFinite(value) && value > 0 ? new Date(value * 1000).toLocaleString() : 'never';
}

function nowSection(state) {
  const row = state.current;
  if (!row) return section('Now', [empty('No reading yet.')]);
  const missing = new Set(state.not_available || []);
  const rows = NOW_ROWS.map(([key, label, show]) => line(label,
    missing.has(key) || row[key] === null || row[key] === undefined
      ? 'N/A on this card' : show(row[key], row)));
  rows.push(line('Model on the card', row.model || 'none known'));
  rows.push(line('A request running', row.running ? 'yes' : 'no'));
  rows.push(line('Monitor', `${state.running ? 'running' : 'STOPPED'} · last reading ${when(state.last_ok)}`));
  return section('Now', rows);
}

function alarmsSection(alarms) {
  if (!alarms.length) return section('Alarms in force', [empty('None.')]);
  return section('Alarms in force', alarms.map((item) => element('div', { class: 'row' }, [
    element('span', { class: item.level === 'critical' || item.level === 'alarm' ? 'error' : '',
                      text: `${item.level.toUpperCase()} · ${item.message}` }),
    element('span', { class: 'muted', text: `since ${when(item.since)}` }),
  ])));
}

function referenceRows(scope, base, recent) {
  const label = scope === 'global' ? 'All models' : scope;
  const rows = [];
  if (base) {
    rows.push(line(`${label}: reference R`, `${base.r.toFixed(3)} °C/W`,
      `Median of ${base.windows} stable windows, ${base.period_start} – ${base.period_end}`
      + (base.note ? ` (${base.note})` : '')));
  } else {
    rows.push(line(`${label}: reference R`, 'not formed yet',
      'Formed after the first 21 days, from at least 10 stable load windows.'));
  }
  if (recent) {
    const change = base ? ` (${((recent.r - base.r) / base.r * 100).toFixed(0)} % vs reference)` : '';
    rows.push(line(`${label}: last 14 days`, `${recent.r.toFixed(3)} °C/W${change}`,
      `${recent.windows} stable windows`));
  }
  return rows;
}

function trendSection(analysis) {
  const idle = analysis.idle_today;
  const rows = [line('Resting temperature today',
    idle ? `${idle.temp.toFixed(1)} °C${idle.from_day ? ` (from ${idle.from_day})` : ''}` : 'no reference',
    'Median chip temperature while the card rests (below 25 W, below 5 % use, 10 min or more).')];
  const scopes = new Set(['global', ...Object.keys(analysis.baselines || {}),
                          ...Object.keys(analysis.trend || {})]);
  for (const scope of scopes) {
    rows.push(...referenceRows(scope, (analysis.baselines || {})[scope],
                               (analysis.trend || {})[scope]));
  }
  return section('Thermal resistance (R = temperature above rest ÷ power)', rows);
}

export function chart(points, field, colour) {
  const svg = document.createElementNS(SVG, 'svg');
  svg.setAttribute('viewBox', '0 0 360 80');
  svg.setAttribute('class', 'gpu-chart');
  const values = points.map((point) => point[field]).filter((value) => value !== null);
  if (values.length < 2) return svg;
  const low = Math.min(...values);
  const high = Math.max(...values);
  const span = high - low || 1;
  const step = 360 / Math.max(points.length - 1, 1);
  const path = points.map((point, index) => (point[field] === null ? null
    : `${(index * step).toFixed(1)},${(76 - (point[field] - low) / span * 72).toFixed(1)}`))
    .filter(Boolean).join(' ');
  const polyline = document.createElementNS(SVG, 'polyline');
  polyline.setAttribute('points', path);
  polyline.setAttribute('fill', 'none');
  polyline.setAttribute('stroke', colour);
  polyline.setAttribute('stroke-width', '1.5');
  svg.append(polyline);
  return svg;
}

function chartSection(history) {
  const points = history?.points || [];
  if (!points.length) return section('Last 24 hours', [empty('No readings in this period.')]);
  const range = (field) => {
    const values = points.map((point) => point[field]).filter((value) => value !== null);
    return values.length ? `${Math.min(...values).toFixed(0)} – ${Math.max(...values).toFixed(0)}` : '—';
  };
  return section('Last 24 hours', [
    line('Chip temperature (°C)', range('temp_gpu')), chart(points, 'temp_gpu', 'var(--error)'),
    line('Power (W)', range('power_w')), chart(points, 'power_w', 'var(--accent)'),
    line('Fan (%)', range('fan_pct')), chart(points, 'fan_pct', 'var(--muted)'),
  ]);
}

function memorySection(state) {
  const slow = state.slow;
  const rows = [];
  if (slow) {
    rows.push(line('Driver', slow.driver));
    for (const [name, value] of Object.entries(slow.ecc || {})) {
      rows.push(line(`Memory errors, ${name.replace('_', ' ')}`, String(value)));
    }
    rows.push(line('Replaced memory rows', slow.rows
      ? `${slow.rows.correctable} correctable · ${slow.rows.uncorrectable} uncorrectable · `
        + `${slow.rows.pending} pending · ${slow.rows.failure} failed` : 'N/A on this card'));
  } else {
    rows.push(empty('Read every hour; not read yet.'));
  }
  const helper = state.helper || {};
  rows.push(line('Kernel log (Xid, PCIe AER)', helper.installed
    ? `helper heartbeat ${when(helper.last_heartbeat)}`
    : 'not watched: the host helper is not installed'));
  return section('Memory and kernel log', rows);
}

function eventsSection(events) {
  if (!events.length) return section('Events', [empty('Nothing yet.')]);
  return section('Events', events.slice(0, 40).map((event) => line(
    `${when(event.ts)} · ${event.level}`, event.message)));
}

function actionsSection(redraw) {
  const note = element('input', { type: 'text', placeholder: 'Why (for example: new thermal paste)' });
  const act = (label, run) => element('button', { class: 'action', text: label, onclick: async () => {
    try {
      await run();
    } catch (error) {
      await showNotice({ title: `Could not ${label.toLowerCase()}`, body: error.message });
    }
    redraw();
  } });
  return section('Actions', [
    element('div', { class: 'row', title: 'Sends one warning through the alarm channel.' }, [
      element('span', { class: 'muted', text: 'See an alarm arrive' }),
      act('Send test alarm', () => api.gpuTestAlarm())]),
    element('label', { class: 'row', title: 'Forget the reference values; new ones form over the next 21 days.' }, [
      note, act('Reset reference', () => api.resetGpuBaseline(note.value))]),
  ]);
}

export function stopRefreshing() {
  if (timer) { window.clearInterval(timer); timer = null; }
}

export async function render(container) {
  stopRefreshing();
  let state;
  let history = null;
  try {
    state = await api.gpuHealth();
    if (state.available) history = await api.gpuHealthHistory(24);
  } catch (error) {
    container.replaceChildren(element('p', { class: 'error', text: error.message }));
    return;
  }
  if (!state.available) {
    container.replaceChildren(empty(`GPU health is not watched here. ${state.reason || ''}`));
    return;
  }
  const redraw = () => render(container).catch(() => {});
  container.replaceChildren(element('div', { class: 'lanes' }, [
    empty('The card read every 15 seconds, for years: what it does now, what is wrong, and how '
          + 'its cooling compares with when it was new. Read only — nothing here changes the card.'),
    alarmsSection(state.alarms || []),
    nowSection(state),
    trendSection(state.analysis || {}),
    chartSection(history),
    memorySection(state),
    eventsSection(state.events || []),
    actionsSection(redraw),
  ]));
  timer = window.setInterval(() => {
    if (!container.contains(document.activeElement)) redraw();
  }, EVERY_MS);
}
