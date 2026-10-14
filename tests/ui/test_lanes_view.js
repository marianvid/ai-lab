// The Cloud & CPU page: models that never touch the card, each with its own queue.

import assert from 'node:assert/strict';
import { afterEach, describe, it } from 'node:test';

import { button, installDom, settle } from './support/dom.js';

const STATS = {
  loaded: [], shapes: [], queue_runs: [], memory: { pools: [] },
  providers: {
    anthropic: {
      in_flight: 2, concurrency: 6, paused_for_s: 0, rate_limit_strikes: 0,
      default_effort: 'medium',
      today: { requests: 14, ok: 13, failures: { 'not logged in': 1 } },
      models: ['claude/haiku', 'claude/sonnet'],
    },
  },
  cpu_models: {
    'bge-m3': { running: true, port: 8390, places: 4, in_flight: 1, waiting: 3,
                requests: 40, failures: 0, idle_s: 0.4, idle_stop_s: 1800 },
  },
};

// Every page drawn here keeps itself fresh on a timer, and a click redraws it,
// which starts a new one. Each test stops the timer of the copy it drew and
// closes its window, so nothing keeps the test run alive.
let open = null;

afterEach(() => {
  if (!open) return;
  open.view.stopRefreshing();
  open.context.window.close();
  open = null;
});

async function renderPage(responses) {
  const context = installDom(responses);
  const view = await import(`../../ai_lab/web/js/views/lanes.js?${Math.random()}`);
  open = { view, context };
  await view.render(context.view);
  await settle();
  return context;
}

describe('the Cloud & CPU page', () => {
  it('shows each vendor and processor model with its own queue', async () => {
    const { view } = await renderPage({ '/api/gateway': STATS });
    const text = view.textContent;
    assert.match(text, /Subscription · anthropic/);
    assert.match(text, /claude\/haiku, claude\/sonnet/);
    assert.match(text, /2 from 6/);
    assert.match(text, /not logged in ×1/);
    assert.match(text, /Processor · bge-m3/);
    assert.match(text, /1 from 4/);
    assert.match(text, /30 min idle/);
    assert.match(text, /POST \/embeddings/);
  });

  it('stops a running processor model by hand', async () => {
    const context = await renderPage({ '/api/gateway': STATS });
    button(context.view, 'Stop').click();
    await settle();
    assert.ok(context.calls.some((call) => call.method === 'POST'
      && call.path === '/api/cpu-models/bge-m3/stop'));
  });

  it('starts a stopped one, and says why when it cannot', async () => {
    const stopped = { ...STATS, providers: {},
                      cpu_models: { 'bge-m3': { ...STATS.cpu_models['bge-m3'],
                                                running: false, idle_s: null } } };
    const context = await renderPage({
      '/api/gateway': stopped,
      'POST /api/cpu-models/bge-m3/start': { __status: 500, error: 'no binary' },
    });
    assert.match(context.view.textContent, /not used yet/);
    button(context.view, 'Start').click();
    await settle(12);
    assert.match(context.document.body.textContent, /Could not start bge-m3/);
    const dialog = context.document.querySelector('dialog[open]');
    if (dialog) dialog.close();
    await settle();
  });

  it('says so when nothing is configured', async () => {
    const { view } = await renderPage({ '/api/gateway': { loaded: [] } });
    assert.match(view.textContent, /None configured/);
  });

  it('shows the error when the gateway cannot be read', async () => {
    const { view } = await renderPage({ '/api/gateway': { __status: 500, error: 'down' } });
    assert.match(view.textContent, /down/);
  });
});
