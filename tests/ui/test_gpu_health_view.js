// The GPU health page: the card now, N/A where the card has no sensor, the
// alarms, the reference and trend, the chart, the buttons.

import assert from 'node:assert/strict';
import { afterEach, describe, it } from 'node:test';

import { button, installDom, settle } from './support/dom.js';

const STATE = {
  available: true, running: true, last_ok: 1761300000,
  not_available: ['temp_hotspot', 'temp_mem'],
  current: { temp_gpu: 28, temp_mem: null, temp_hotspot: null, power_w: 9.6, power_limit_w: 200,
             fan_pct: 30, clock_sm: 180, clock_mem: 405, util_pct: 0, throttle: 0x5,
             pcie_gen: 1, pcie_width: 4, pcie_replays: 0, model: 'qwen38-nvfp4', running: 1 },
  slow: { driver: '610.57.04', ecc: { corrected_aggregate: 0 },
          rows: { correctable: 0, uncorrectable: 0, pending: 0, failure: 0 } },
  helper: { installed: false, last_heartbeat: null },
  alarms: [{ level: 'alarm', message: 'Chip at 90 °C', since: 1761300000 }],
  analysis: { idle_today: { temp: 28.4, from_day: '2026-10-24' },
              baselines: { global: { r: 0.2, windows: 12, period_start: '2026-10-01',
                                     period_end: '2026-10-21', note: '' } },
              trend: { global: { r: 0.23, windows: 30 }, 'qwen38-nvfp4': { r: 0.21, windows: 20 } } },
  events: [{ ts: 1761300000, level: 'info', message: 'Reference formed for global' }],
};
const HISTORY = { points: [{ ts: 1, temp_gpu: 28, power_w: 10, fan_pct: 30 },
                           { ts: 2, temp_gpu: 60, power_w: 150, fan_pct: 50 },
                           { ts: 3, temp_gpu: null, power_w: null, fan_pct: null }] };

let open = null;

afterEach(() => {
  if (!open) return;
  open.view.stopRefreshing();
  open.context.window.close();
  open = null;
});

async function renderPage(responses) {
  const context = installDom(responses);
  const view = await import(`../../ai_lab/web/js/views/gpu_health.js?${Math.random()}`);
  open = { view, context };
  await view.render(context.view);
  await settle();
  return context;
}

describe('the GPU health page', () => {
  it('shows the card now, with N/A where it has no sensor', async () => {
    const { view } = await renderPage({ '/api/gpu-health': STATE,
                                        '/api/gpu-health/history': HISTORY });
    const text = view.textContent;
    assert.match(text, /Chip temperature28 °C/);
    assert.match(text, /HotspotN\/A on this card/);
    assert.match(text, /9.6 W of 200 W/);
    assert.match(text, /idle, power cap/);
    assert.match(text, /Gen 1 x4/);
    assert.match(text, /qwen38-nvfp4/);
    assert.doesNotMatch(text, /null|undefined|NaN/);
  });

  it('shows the alarms, the reference and the trend', async () => {
    const { view } = await renderPage({ '/api/gpu-health': STATE,
                                        '/api/gpu-health/history': HISTORY });
    const text = view.textContent;
    assert.match(text, /ALARM · Chip at 90 °C/);
    assert.match(text, /28.4 °C \(from 2026-10-24\)/);
    assert.match(text, /0.200 °C\/W/);
    assert.match(text, /0.230 °C\/W \(15 % vs reference\)/);
    assert.match(text, /qwen38-nvfp4: reference Rnot formed yet/);
    assert.match(text, /not watched: the host helper is not installed/);
    assert.equal(view.querySelectorAll('svg.gpu-chart polyline').length, 3);
  });

  it('says why when there is no card to watch', async () => {
    const { view } = await renderPage({ '/api/gpu-health':
      { available: false, reason: 'No NVIDIA driver on this machine.' } });
    assert.match(view.textContent, /not watched here. No NVIDIA driver/);
  });

  it('says so when nothing was read yet', async () => {
    const { view } = await renderPage({ '/api/gpu-health':
      { ...STATE, current: null, slow: null, alarms: [], events: [],
        analysis: { idle_today: null, baselines: {}, trend: {} } },
      '/api/gpu-health/history': { points: [] } });
    const text = view.textContent;
    assert.match(text, /No reading yet/);
    assert.match(text, /None\./);
    assert.match(text, /no reference/);
    assert.match(text, /not read yet/);
    assert.match(text, /No readings in this period/);
  });

  it('sends a test alarm and resets the reference with a note', async () => {
    const context = await renderPage({ '/api/gpu-health': STATE,
                                       '/api/gpu-health/history': HISTORY });
    button(context.view, 'Send test alarm').click();
    await settle();
    context.view.querySelector('input[type=text]').value = 'new paste';
    button(context.view, 'Reset reference').click();
    await settle();
    assert.ok(context.calls.some((call) => call.path === '/api/gpu-health/test-alarm'));
    const reset = context.calls.find((call) => call.path === '/api/gpu-health/baseline/reset');
    assert.equal(JSON.parse(reset.body).note, 'new paste');
  });

  it('shows the error when the status cannot be read', async () => {
    const { view } = await renderPage({ '/api/gpu-health': { __status: 500, error: 'down' } });
    assert.match(view.textContent, /down/);
  });
});
