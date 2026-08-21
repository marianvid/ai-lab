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
  busy: false,
  holder: null,
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

  it('says the card is free when nothing holds it', async () => {
    const { view } = await renderPage();
    assert.match(view.textContent, /free/);
  });

  it('names the model that is working, not just that something is', async () => {
    // "Busy" alone cannot tell you whether the thing you are about to stop is
    // the thing that is working.
    const { view } = await renderPage({
      busy: true, holder: { instance_id: 'reviewer', answering: true },
    });
    assert.match(view.textContent, /reviewer is answering a request/);
  });

  it('distinguishes a model that is loading from one that is answering', async () => {
    const { view } = await renderPage({
      busy: true, holder: { instance_id: 'reviewer', answering: false },
    });
    assert.match(view.textContent, /reviewer is loading/);
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
