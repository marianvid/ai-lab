// The Reservations page: the holder, the queue, recent history, the settings.

import assert from 'node:assert/strict';
import { afterEach, describe, it } from 'node:test';

import { button, installDom, settle } from './support/dom.js';

const SETTINGS = { heartbeat_timeout_s: 180, unused_s: 600, default_max_s: 1800,
                   ceiling_s: 14400, aging_s: 600, unreserved_wait_s: 1800, required: false };
const BOOK = {
  holder: { id: 'h1', team: 'ingest', instance: 'ct109', purpose: 'collect',
            run_id: '20261017-110327', priority: 4, effective_priority: 4,
            granted_at: 1760700000, last_beat: 1760700100, first_call_at: null,
            time_left_s: 1500, restored: true },
  queue: [{ id: 'q1', team: 'ingest', instance: 'mac', purpose: 'benchmark', run_id: '',
            priority: 2, effective_priority: 3, requested_at: 1760700050, position: 1 }],
  history: [{ id: 'e1', team: 'music', instance: '', purpose: '', run_id: '',
              ended_at: 1760690000, reason: 'no heartbeat for 180 s' }],
  settings: SETTINGS,
};

let open = null;

afterEach(() => {
  if (!open) return;
  open.view.stopRefreshing();
  open.context.window.close();
  open = null;
});

async function renderPage(responses) {
  const context = installDom(responses);
  const view = await import(`../../ai_lab/web/js/views/reservations.js?${Math.random()}`);
  open = { view, context };
  await view.render(context.view);
  await settle();
  return context;
}

describe('the Reservations page', () => {
  it('shows who holds the card, who waits and what ended', async () => {
    const { view } = await renderPage({ '/api/reservations': BOOK });
    const text = view.textContent;
    assert.match(text, /ingest @ ct109 · collect · run 20261017-110327/);
    assert.match(text, /25 min/);
    assert.match(text, /after a manager restart/);
    assert.match(text, /1\. ingest @ mac · benchmark/);
    assert.match(text, /priority 2 → 3/);
    assert.match(text, /no heartbeat for 180 s/);
  });

  it('says so when the card is free', async () => {
    const { view } = await renderPage({ '/api/reservations':
      { holder: null, queue: [], history: [], settings: SETTINGS } });
    assert.match(view.textContent, /The card is not reserved/);
    assert.match(view.textContent, /Nobody is waiting/);
  });

  it('counts the calls without a reservation that wait at the door', async () => {
    const { view } = await renderPage({ '/api/reservations': { ...BOOK, queue: [],
      at_the_door: { calls: 11, models: { 'qwen38-nvfp4': 11 }, oldest_since: 1760700060 } } });
    const text = view.textContent;
    assert.match(text, /11 calls without a reservation/);
    assert.match(text, /qwen38-nvfp4 ×11/);
    assert.doesNotMatch(text, /Nobody is waiting/);
  });

  it('cancels a reservation by hand', async () => {
    const context = await renderPage({ '/api/reservations': BOOK });
    button(context.view, 'Cancel').click();
    await settle();
    assert.ok(context.calls.some((call) => call.method === 'POST'
      && call.path === '/api/reservations/h1/cancel'));
  });

  it('saves the settings in seconds', async () => {
    const context = await renderPage({ '/api/reservations': BOOK });
    const input = context.view.querySelector('input[type=number]');
    assert.equal(input.value, '3');
    input.value = '5';
    button(context.view, 'Save').click();
    await settle();
    const saved = context.calls.find((call) => call.method === 'PATCH');
    assert.equal(JSON.parse(saved.body).heartbeat_timeout_s, 300);
    assert.equal(JSON.parse(saved.body).ceiling_s, 14400);
  });

  it('shows the error when the book cannot be read', async () => {
    const { view } = await renderPage({ '/api/reservations': { __status: 500, error: 'down' } });
    assert.match(view.textContent, /down/);
  });
});
