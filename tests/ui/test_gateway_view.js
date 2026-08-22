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
  requests_per_minute: 12,
  average_first_token_s: 0.42,
  switches: 2,
  average_wait_s: 1.4,
  average_switch_s: 31.2,
  switching_share: 18.5,
  last_error: '',
  shapes: [
    { path: '/v1/chat/completions', models: ['coder', 'fast'],
      engines: ['llama.cpp', 'vLLM'] },
    { path: '/v1/completions', models: ['coder', 'fast'],
      engines: ['llama.cpp', 'vLLM'] },
    { path: '/v1/embeddings', models: ['coder', 'fast'],
      engines: ['llama.cpp', 'vLLM'] },
    { path: '/v1/messages', models: ['fast'], engines: ['vLLM'] },
    { path: '/v1/messages/count_tokens', models: ['fast'], engines: ['vLLM'] },
  ],
  recent: [
    { at: 1, loaded: 'coder', unloaded: ['reviewer'], took_s: 31.2, load_ms: 28100 },
  ],
};

async function renderPage(overrides = {}) {
  const context = installDom({ '/api/gateway': { ...STATS, ...overrides } });
  const { render } = await import(`../../ai_lab/web/js/views/gateway.js?${Math.random()}`);
  await render(context.view);
  await settle();
  // The view keeps itself fresh on a timer. Stopping it explicitly rather than
  // relying on the window closing: the timer would otherwise fire into a
  // window that is gone.
  const { stopRefreshing } = await import('../../ai_lab/web/js/views/gateway.js');
  stopRefreshing();
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

  it('reports a rate rather than a total that only grows', async () => {
    const { view } = await renderPage();
    assert.match(view.textContent, /Requests per minute\s*12/);
  });

  it('reports the time to the first token', async () => {
    const { view } = await renderPage();
    assert.match(view.textContent, /Time to first token\s*0.42 s/);
  });

  it('reports how many are waiting', async () => {
    const { view } = await renderPage({ waiting: 7 });
    assert.match(view.textContent, /Queue size\s*7/);
  });

  it('states the share of working time spent loading', async () => {
    const { view } = await renderPage();
    assert.match(view.textContent, /18.5%/);
  });

  it('writes a switch as the move it was, with the time on the right', async () => {
    const { view } = await renderPage();
    const row = [...view.querySelectorAll('.row')]
      .find((node) => node.textContent.includes('→'));
    assert.match(row.textContent, /reviewer → coder/);
    assert.match(row.textContent, /31.2 s/);
  });

  it('says what was unloaded even when it was nothing', async () => {
    const { view } = await renderPage({
      recent: [{ at: 1, loaded: 'coder', unloaded: [], took_s: 4.1, load_ms: 3851 }],
    });
    assert.match(view.textContent, /nothing → coder/);
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

  // Setting .value does not tell the page anything; a person typing does.
  function type(input, value) {
    input.value = value;
    input.dispatchEvent(new input.ownerDocument.defaultView.Event('input'));
  }

  function field(view, label) {
    const found = [...view.querySelectorAll('label.field')]
      .find((node) => node.textContent.includes(label));
    if (!found) throw new Error(`no field for "${label}"`);
    return found;
  }

  it('shows what each limit is set to', async () => {
    const { view } = await renderPage();
    assert.equal(field(view, 'first token').querySelector('input').value, '120');
    assert.equal(field(view, 'between tokens').querySelector('input').value, '30');
    assert.equal(field(view, 'Requests held').querySelector('input').value, '150');
  });

  it('explains each one where hovering finds it', async () => {
    // They trade against each other and against the machine. A number with no
    // explanation is a number nobody dares change.
    const { view } = await renderPage();
    assert.match(field(view, 'first token').getAttribute('title'), /streaming/);
    assert.match(field(view, 'between tokens').getAttribute('title'), /17 tokens/);
    assert.match(field(view, 'Requests held').getAttribute('title'), /thread/);
  });

  it('sends every limit when one is saved', async () => {
    const context = installDom({ '/api/gateway': STATS,
                                 'PATCH /api/gateway': STATS });
    const { render } = await import(`../../ai_lab/web/js/views/gateway.js?${Math.random()}`);
    await render(context.view);
    await settle();
    type(field(context.view, 'between tokens').querySelector('input'), '45');
    [...context.view.querySelectorAll('button')]
      .find((node) => node.textContent.trim() === 'Save').click();
    await settle();
    const { stopRefreshing } = await import('../../ai_lab/web/js/views/gateway.js');
    stopRefreshing();
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
    type(field(context.view, 'between tokens').querySelector('input'), '45');
    [...context.view.querySelectorAll('button')]
      .find((node) => node.textContent.trim() === 'Save').click();
    await settle();
    const { stopRefreshing } = await import('../../ai_lab/web/js/views/gateway.js');
    stopRefreshing();
    const told = context.document.getElementById('status').textContent;
    context.window.close();          // after reading it: closing empties it
    assert.match(told, /must be between/);
  });

  it('says nothing under the fields that the fields do not say', async () => {
    // What each is for is in its tooltip, where somebody deciding about that
    // one setting looks. A paragraph under all three is read once and then
    // never again.
    const { view } = await renderPage();
    assert.equal(view.textContent.includes('wedged engine'), false);
  });
});

describe('what can be sent to the front door', () => {
  // Two shapes of request are in circulation. Listing only the address is half
  // the answer, and the wrong half for anybody whose tool speaks the other.

  it('lists the usual shape', async () => {
    const { view } = await renderPage();
    assert.match(view.textContent, /\/v1\/chat\/completions/);
  });

  it('lists the Anthropic shape too', async () => {
    // vLLM serves it; llama.cpp does not. A client written against Anthropic's
    // own library sends this and nothing else.
    const { view } = await renderPage();
    assert.match(view.textContent, /\/v1\/messages/);
  });

  it('says a shape every model answers is answered by every model', async () => {
    const { view } = await renderPage();
    assert.match(view.textContent, /every model/);
  });

  it('names the engine when only some answer a shape', async () => {
    // The engine, not the models: a list of names grows every time an entry is
    // added and is stale by the next one. "vLLM models" says it once and stays
    // true. The names are in the tooltip for anyone who wants them.
    const { view } = await renderPage();
    const row = [...view.querySelectorAll('.row')]
      .find((node) => node.textContent.includes('/v1/messages'));
    assert.match(row.textContent, /vLLM models/);
    assert.match(row.getAttribute('title'), /1 of 2: fast/);
  });

  it('answers the key question with one word', async () => {
    // It said what is not checked, which is not what somebody needs to type.
    const { view } = await renderPage();
    const row = [...view.querySelectorAll('.row')]
      .find((node) => node.textContent.includes('API key'));
    assert.equal(row.textContent.replace('API key', '').trim(), 'none');
  });

  it('lays the page out, and in its own element', async () => {
    // The class used to go on the page container, which every tab shares — so
    // Models and Library inherited a layout meant for this page.
    const { view } = await renderPage();
    assert.equal(view.classList.contains('columns'), false,
                 'it wrote a layout onto the shared container');
    assert.ok(view.querySelector('.gateway-grid'));
  });

  it('gives each section its place in the grid', async () => {
    const { view } = await renderPage();
    for (const place of ['at-address', 'at-side', 'at-limits', 'at-recent']) {
      assert.ok(view.querySelector(`.${place}`), `nothing at ${place}`);
    }
  });
});

describe('saving a limit', () => {
  function field(view, label) {
    return [...view.querySelectorAll('label.field')]
      .find((node) => node.textContent.includes(label));
  }
  function saveButton(view) {
    return [...view.querySelectorAll('button')]
      .find((node) => node.textContent.trim() === 'Save');
  }
  function type(input, value) {
    input.value = value;
    input.dispatchEvent(new input.ownerDocument.defaultView.Event('input'));
  }

  it('offers nothing to save until something differs', async () => {
    // A button that is always ready invites a press that does nothing, and
    // then you cannot tell whether the last press took.
    const { view } = await renderPage();
    assert.equal(saveButton(view).disabled, true);
  });

  it('wakes when a value is changed', async () => {
    const { view } = await renderPage();
    type(field(view, 'between tokens').querySelector('input'), '45');
    assert.equal(saveButton(view).disabled, false);
  });

  it('sleeps again when the value is typed back', async () => {
    const { view } = await renderPage();
    const input = field(view, 'between tokens').querySelector('input');
    type(input, '45');
    type(input, '30');
    assert.equal(saveButton(view).disabled, true);
  });
});

describe('what is happening, in one word', () => {
  it('says idle when nothing is running or waiting', async () => {
    const { view } = await renderPage();
    assert.match(view.textContent, /Status\s*idle/);
  });

  it('says working when answers are in progress', async () => {
    const { view } = await renderPage({ in_flight: 3, places: 8, busy: true });
    assert.match(view.textContent, /working/);
  });

  it('says so when there is a queue behind the work', async () => {
    // Different from working: something is being held up, and by what is on
    // the line below.
    const { view } = await renderPage({
      in_flight: 3, places: 8, waiting: 2, busy: true,
      waiting_for: [{ instance_id: 'reviewer', waiting: 2, longest_wait_s: 4 }],
    });
    assert.match(view.textContent, /working, with a queue/);
  });

  it('says a model is loading while it loads', async () => {
    const { view } = await renderPage({ switching: true, busy: true });
    assert.match(view.textContent, /loading a model/);
  });

  it('says it is waiting to swap when the card is empty but somebody wants it', async () => {
    const { view } = await renderPage({
      in_flight: 0, waiting: 1, busy: true,
      waiting_for: [{ instance_id: 'reviewer', waiting: 1, longest_wait_s: 1 }],
    });
    assert.match(view.textContent, /waiting to swap/);
  });
});
