// The Gateway page, from the user's side.
//
// It answers one question: is the workflow running, or is it spending its
// time changing models? Every test here is about that question being
// answerable at a glance rather than by dividing two numbers in your head.

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { installDom, settle } from './support/dom.js';

const STATS = {
  current: 'coder',
  current_settings: { context_size: 32768 },
  busy: false,
  holder: null,
  in_flight: 0,
  places: 8,
  switching: false,
  waiting: 0,
  waiting_for: [],
  longest_wait_s: 0.0,
  max_waiting: 150,
  first_byte_s: 120.0,
  between_bytes_s: 30.0,
  requests: 20,
  switches: 2,
  average_wait_s: 1.4,
  average_switch_s: 31.2,
  total_switch_s: 62.4,
  last_error: '',
  recent: [
    { at: 1, loaded: 'coder', unloaded: ['reviewer'], took_s: 31.2, load_ms: 28100 },
  ],
};

async function renderPage(overrides = {}) {
  const context = installDom({ '/api/gateway': { ...STATS, ...overrides } });
  const { render } = await import(`../../ai_lab/web/js/views/gateway.js?${Math.random()}`);
  await render(context.view);
  await settle();
  // The view keeps itself fresh on a timer; closing the window cancels it so
  // the test run can finish.
  context.window.close();
  return context;
}

describe('the Gateway page', () => {
  it('never puts the word null on the page', async () => {
    const { view } = await renderPage();
    assert.equal(view.textContent.includes('null'), false, view.textContent);
  });

  it('gives the address to point an agent at', async () => {
    // Without this the address has to be guessed from the browser bar, and
    // the /v1 on the end is exactly the part that gets guessed wrong.
    const { view } = await renderPage();
    assert.match(view.textContent, /http:\/\/localhost:8090\/v1/);
  });

  it('says which model is on the card', async () => {
    const { view } = await renderPage();
    assert.match(view.textContent, /coder/);
  });

  it('says the card is idle when nothing is running on it', async () => {
    const { view } = await renderPage();
    assert.match(view.textContent, /idle/);
    assert.match(view.textContent, /nobody waiting/);
  });

  it('counts the answers in progress against the places there are', async () => {
    // Requests to the model on the card run together now. How many are in
    // flight, and how many could be, is the figure that says whether the
    // concurrency you configured is being used.
    const { view } = await renderPage({ in_flight: 3, places: 8, busy: true });
    assert.match(view.textContent, /3 of 8 places in use/);
  });

  it('says how many are waiting and how long the longest has', async () => {
    const { view } = await renderPage({
      waiting: 4, longest_wait_s: 12.5, busy: true,
      waiting_for: [{ instance_id: 'reviewer', waiting: 4, longest_wait_s: 12.5 }],
    });
    assert.match(view.textContent, /4 waiting, longest 12.5 s/);
  });

  it('says which models the queue is waiting for', async () => {
    // Two models fighting over one card is a different fault from too little
    // concurrency, and the fix is different too.
    const { view } = await renderPage({
      waiting: 5, longest_wait_s: 9,
      waiting_for: [
        { instance_id: 'reviewer', waiting: 3, longest_wait_s: 9 },
        { instance_id: 'coder', waiting: 2, longest_wait_s: 2 },
      ],
    });
    assert.match(view.textContent, /wanting reviewer/);
    assert.match(view.textContent, /wanting coder/);
  });

  it('says a model is loading rather than answering', async () => {
    const { view } = await renderPage({ switching: true, busy: true });
    assert.match(view.textContent, /loading/);
  });

  it('states switching as a share of requests rather than leaving the division', async () => {
    const { view } = await renderPage();
    assert.match(view.textContent, /10%/);
  });

  it('says plainly when the workflow is thrashing', async () => {
    // The whole point of the page. A count of switches means nothing without
    // the count of requests beside it, and even then it wants saying.
    const { view } = await renderPage({ requests: 10, switches: 9 });
    assert.match(view.textContent, /reorder the workflow/);
  });

  it('does not accuse a well-behaved workflow of thrashing', async () => {
    const { view } = await renderPage({ requests: 100, switches: 2 });
    assert.match(view.textContent, /mostly staying on one model/);
  });

  it('does not divide by zero before any traffic', async () => {
    const { view } = await renderPage({ requests: 0, switches: 0, recent: [] });
    assert.equal(view.textContent.includes('NaN'), false, view.textContent);
    assert.match(view.textContent, /nothing yet/);
  });

  it('says what was unloaded for what', async () => {
    const { view } = await renderPage();
    assert.match(view.textContent, /coder in, reviewer out/);
  });

  it('does not report a tidy-up as a load that happened', async () => {
    // The model asked for was already up and something else was unloaded from
    // beside it. Calling that "coder in" would claim a load that never ran.
    const { view } = await renderPage({
      recent: [{ at: 1, loaded: 'coder', unloaded: ['reviewer'],
                 took_s: 0, load_ms: 0, tidied: true }],
    });
    assert.match(view.textContent, /reviewer unloaded from beside coder/);
    assert.equal(view.textContent.includes('coder in,'), false);
  });

  it('shows the last error rather than hiding a failure behind good averages', async () => {
    const { view } = await renderPage({
      last_error: 'The card still holds 9000 MB 60 seconds after everything was unloaded',
    });
    assert.match(view.textContent, /still holds 9000 MB/);
  });
});

describe('the limits, on the page that shows what they cost', () => {
  // Editable here rather than on Settings: they are about the gateway, and
  // they sit beside the figures you would read before deciding to change them.

  function field(view, label) {
    const found = [...view.querySelectorAll('label.field')]
      .find((node) => node.textContent.includes(label));
    if (!found) throw new Error(`no field for "${label}"`);
    return found;
  }

  it('shows what each limit is set to', async () => {
    const { view } = await renderPage();
    assert.equal(field(view, 'first byte').querySelector('input').value, '120');
    assert.equal(field(view, 'between bytes').querySelector('input').value, '30');
    assert.equal(field(view, 'Requests held').querySelector('input').value, '150');
  });

  it('explains each one where hovering finds it', async () => {
    // They trade against each other and against the machine. A number with no
    // explanation is a number nobody dares change.
    const { view } = await renderPage();
    assert.match(field(view, 'first byte').getAttribute('title'), /streaming/);
    assert.match(field(view, 'between bytes').getAttribute('title'), /17 tokens/);
    assert.match(field(view, 'Requests held').getAttribute('title'), /thread/);
  });

  it('sends every limit when one is saved', async () => {
    const context = installDom({ '/api/gateway': STATS,
                                 'PATCH /api/gateway': STATS });
    const { render } = await import(`../../ai_lab/web/js/views/gateway.js?${Math.random()}`);
    await render(context.view);
    await settle();
    field(context.view, 'between bytes').querySelector('input').value = '45';
    [...context.view.querySelectorAll('button')]
      .find((node) => node.textContent.trim() === 'Save').click();
    await settle();
    context.window.close();
    const sent = context.calls.find((call) => call.method === 'PATCH');
    assert.ok(sent, 'nothing was saved');
    assert.deepEqual(JSON.parse(sent.body),
                     { first_byte_s: 120, between_bytes_s: 45, max_waiting: 150 });
  });

  it('says so when the server refuses a number', async () => {
    const context = installDom({
      '/api/gateway': STATS,
      'PATCH /api/gateway': { __status: 400,
                              error: 'between_bytes_s must be between 1.0 and 3600.0' },
    });
    const { attachStatus } = await import('../../ai_lab/web/js/status.js');
    attachStatus(context.document.getElementById('status'));
    const { render } = await import(`../../ai_lab/web/js/views/gateway.js?${Math.random()}`);
    await render(context.view);
    await settle();
    [...context.view.querySelectorAll('button')]
      .find((node) => node.textContent.trim() === 'Save').click();
    await settle();
    const told = context.document.getElementById('status').textContent;
    context.window.close();          // after reading it: closing empties it
    assert.match(told, /must be between/);
  });

  it('says they are limits of safety, not settings to tune', async () => {
    const { view } = await renderPage();
    assert.match(view.textContent, /safety, not of patience/);
  });
});
