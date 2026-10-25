// The Access page: On/Off, the clients and their maps, the log, Security.

import assert from 'node:assert/strict';
import { afterEach, describe, it } from 'node:test';

import { button, installDom, settle } from './support/dom.js';

const CLIENTS = { clients: [
  { name: 'scheduler-linux', registered_at: 1, last_call_at: 1760700000,
    heartbeat_at: 1760700000, down: false,
    map: { cards: [{ card: 'linux', reservations: [
      { start: '21:15', end: '21:45', task: 'ingest collect' }] }] } },
  { name: 'voice-studio', registered_at: 1, last_call_at: null, heartbeat_at: null,
    down: false, map: null },
] };
const BASE = {
  '/api/power': { on: true, since: null },
  '/api/clients': CLIENTS,
  '/api/journal': { lines: [{ id: 1, at: 1760700000, text: 'qwen36: starting' }],
                            last_id: 1 },
  '/api/security': { require_token: false, login_required: false, password_set: false },
};

let open = null;

afterEach(() => {
  if (!open) return;
  open.view.stopRefreshing();
  open.context.window.close();
  open = null;
});

async function renderPage(responses = BASE) {
  const context = installDom(responses);
  const view = await import(`../../ai_lab/web/js/views/access.js?${Math.random()}`);
  open = { view, context };
  await view.render(context.view);
  await settle();
  return context;
}

describe('the Access page', () => {
  it('shows the clients, the announced reservations and the log', async () => {
    const { view } = await renderPage();
    const text = view.textContent;
    assert.match(text, /scheduler-linux/);
    assert.match(text, /heartbeat/);
    assert.match(text, /calls only/);
    assert.match(text, /ingest collect/);
    assert.match(text, /qwen36: starting/);
    assert.match(text, /Set password/);
  });

  it('switches AI-Lab Off', async () => {
    const { view, calls } = await renderPage({ ...BASE, 'POST /api/power': { on: false } });
    button(view, 'Switch Off').click();
    await settle();
    const sent = calls.find((call) => call.method === 'POST' && call.path === '/api/power');
    assert.deepEqual(JSON.parse(sent.body), { on: false });
  });

  it('says when nothing is announced and when AI-Lab is Off', async () => {
    const { view } = await renderPage({ ...BASE, '/api/power': { on: false, since: 1760700000 },
                                        '/api/clients': { clients: [] } });
    assert.match(view.textContent, /No reservation announced/);
    assert.match(view.textContent, /Off since/);
    assert.ok(button(view, 'Switch On'));
  });
});
