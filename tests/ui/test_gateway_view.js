// The Gateway page, from the user's side.
//
// It answers one question: is the workflow running, or is it spending its
// time changing models? Every test here is about that question being
// answerable at a glance rather than by dividing two numbers in your head.

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { installDom, settle } from './support/dom.js';

const STATS = {
  // A list, always. What is on the machine is a set — one model today, more
  // when the budget allows it — so the page reads it the same either way.
  loaded: [{ instance_id: 'coder', engine: 'vLLM',
             settings: { context_size: 32768 }, in_flight: 0, places: 8,
             requests_per_minute: 12, first_token_s: 0.42 }],
  busy: false,
  holder: null,
  in_flight: 0,
  switching: false,
  waiting: 0,
  waiting_for: [],
  longest_wait_s: 0.0,
  max_waiting: 150,
  first_byte_s: 120.0,
  between_bytes_s: 30.0,
  requests_per_minute: 12,
  switches: 2,
  average_wait_s: 1.4,
  switching_share: 18.5,
  last_error: '',
  shapes: [
    { path: '/v1/chat/completions', models: ['coder', 'fast'],
      engines: ['llama.cpp', 'vLLM'] },
    { path: '/v1/completions', models: ['coder', 'fast'],
      engines: ['llama.cpp', 'vLLM'] },
    { path: '/v1/messages', models: ['fast'], engines: ['vLLM'] },
    { path: '/v1/messages/count_tokens', models: ['fast'], engines: ['vLLM'] },
  ],
  card: { temperature_c: 32 },
  memory: {
    unified: false, capacity_mb: 32623 + 55808,
    available_mb: 29077 + 47608, held_by_models_mb: 3546,
    pools: [
      { name: 'card', kind: 'dedicated', total_mb: 32623, used_mb: 3546,
        used_by_models_mb: 3546, reserve_mb: 0, free_mb: 29077,
        capacity_mb: 32623, available_mb: 29077 },
      { name: 'machine', kind: 'dedicated', total_mb: 64000, used_mb: 8200,
        used_by_models_mb: 0, reserve_mb: 8192, free_mb: 55800,
        capacity_mb: 55808, available_mb: 47608 },
    ],
  },
  queue_runs: [],
  recent: [],
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

  it('counts what is loaded rather than naming them', async () => {
    // Two names on one line was a heading that grew with the machine. Which
    // ones is the queue panel's business.
    const { view } = await renderPage({
      loaded: [
        { instance_id: 'coder', engine: 'vLLM', in_flight: 2, places: 8,
          requests_per_minute: 14, first_token_s: 0.31 },
        { instance_id: 'reviewer', engine: 'llama.cpp', in_flight: 0, places: 1,
          requests_per_minute: 3, first_token_s: 0.08 },
      ],
      in_flight: 2, busy: true,
    });
    assert.match(view.textContent, /Loaded\s*2/);
    const activity = [...view.querySelectorAll('.section')]
      .find((item) => item.textContent.startsWith('Activity'));
    assert.equal(activity.textContent.includes('rpm'), false,
                 'per-model figures are back in Activity');
  });

  it('adds the places up, because they are the engines own numbers', async () => {
    const { view } = await renderPage({
      loaded: [
        { instance_id: 'coder', engine: 'vLLM', in_flight: 2, places: 8,
          requests_per_minute: 0, first_token_s: 0 },
        { instance_id: 'reviewer', engine: 'llama.cpp', in_flight: 1, places: 1,
          requests_per_minute: 0, first_token_s: 0 },
      ],
      in_flight: 3, busy: true,
    });
    assert.match(view.textContent, /Processing\s*3 from 9/);
  });

  it('never averages the time to first token across models', async () => {
    // A 3B and a 35B differ by an order of magnitude, so one average across
    // both describes neither. With a single model it was right by accident,
    // and Activity is a column of figures about the machine.
    const { view } = await renderPage();
    assert.equal(view.textContent.includes('Time to first token'), false,
                 'a machine-wide first-token average is back');
  });

  it('says nothing per model when nothing is loaded', async () => {
    const { view } = await renderPage({ loaded: [] });
    assert.match(view.textContent, /Loaded\s*0/);
    assert.equal(view.querySelector('.row.bymodel'), null);
  });

  it('has no average switch, which said nothing anybody acted on', async () => {
    const { view } = await renderPage();
    assert.equal(view.textContent.includes('Average switch'), false);
  });

  it('says the card is idle when nothing is running on it', async () => {
    const { view } = await renderPage();
    assert.match(view.textContent, /idle/);
  });

  it('does not repeat the status in the lines below it', async () => {
    // Answering said "idle" and Waiting said "nobody waiting", both under a
    // Status that had just said idle. They are counts; Status is the state.
    const { view } = await renderPage();
    assert.match(view.textContent, /Processing\s*0 from 8/);
    assert.match(view.textContent, /Queue size\s*0/);
    assert.equal(view.textContent.includes('nobody waiting'), false);
  });

  it('counts the answers in progress against the places there are', async () => {
    // Requests to the model on the card run together now. How many are in
    // flight, and how many could be, is the figure that says whether the
    // concurrency you configured is being used.
    const { view } = await renderPage({
      loaded: [{ instance_id: 'coder', engine: 'vLLM', settings: {},
                 in_flight: 3, places: 8 }],
      in_flight: 3, busy: true,
    });
    assert.match(view.textContent, /Processing\s*3 from 8/);
  });

  it('says how many are waiting, and for what', async () => {
    const { view } = await renderPage({
      waiting: 4, longest_wait_s: 12.5, busy: true,
      waiting_for: [{ instance_id: 'reviewer', waiting: 4, longest_wait_s: 12.5 }],
    });
    assert.match(view.textContent, /Queue size\s*4/);
  });

  it('says a model is loading rather than answering', async () => {
    const { view } = await renderPage({ switching: true, busy: true });
    assert.match(view.textContent, /loading/);
  });

  it('reports a rate rather than a total that only grows', async () => {
    const { view } = await renderPage();
    assert.match(view.textContent, /Requests per minute\s*12/);
  });

  it('reports how many are waiting', async () => {
    const { view } = await renderPage({ waiting: 7 });
    assert.match(view.textContent, /Queue size\s*7/);
  });

  it('states the share of working time spent loading', async () => {
    const { view } = await renderPage();
    assert.match(view.textContent, /18.5%/);
  });

  it('shows what is loaded, what it is doing, and what waits for it', async () => {
    // Running and waiting say different things: running is whether the model
    // is busy, waiting is whether there is pressure on it.
    const { view } = await renderPage({
      loaded: [
        { instance_id: 'coder', engine: 'vLLM', in_flight: 2, places: 8,
          waiting: 3, requests_per_minute: 0, first_token_s: 0 },
        { instance_id: 'gemma', engine: 'llama.cpp', in_flight: 0, places: 1,
          waiting: 0, requests_per_minute: 0, first_token_s: 0 },
      ],
      in_flight: 2,
    });
    const rows = [...view.querySelectorAll('.row.now')]
      .map((row) => [...row.children].map((cell) => cell.textContent.trim()));
    assert.deepEqual(rows, [['→ coder', '2 running · 3 waiting'],
                            ['→ gemma', '0 running · 0 waiting']]);
  });

  it('adds up what is waiting for models already there', async () => {
    const { view } = await renderPage({
      loaded: [
        { instance_id: 'coder', engine: 'vLLM', in_flight: 8, places: 8,
          waiting: 3, requests_per_minute: 0, first_token_s: 0 },
        { instance_id: 'gemma', engine: 'llama.cpp', in_flight: 1, places: 1,
          waiting: 2, requests_per_minute: 0, first_token_s: 0 },
      ],
      in_flight: 9,
    });
    const tally = [...view.querySelectorAll('.row.tally')]
      .map((row) => [...row.children].map((cell) => cell.textContent.trim()));
    assert.deepEqual(tally[0], ['Waiting for these', '5']);
  });

  it('names the next model to be loaded, and what is held up behind it', async () => {
    // Read together they are the cost of the change: this many requests are
    // stopped until that one is on.
    const { view } = await renderPage({
      waiting: 6,
      queue_runs: [
        { instance_id: 'reviewer', requests: 3, longest_wait_s: 9 },
        { instance_id: 'coder', requests: 2, longest_wait_s: 4 },
        { instance_id: 'glm-flash', requests: 1, longest_wait_s: 1 },
      ],
    });
    const next = view.querySelector('.row.next');
    assert.match(next.textContent, /reviewer\s*←/);
    assert.match(next.textContent, /3 waiting/);
    assert.match(next.title, /longest has waited 9 s/);
    const tally = [...view.querySelectorAll('.row.tally')]
      .map((row) => [...row.children].map((cell) => cell.textContent.trim()));
    assert.deepEqual(tally[1], ['Waiting behind it', '3'],
                     'the 2 for coder and the 1 for glm-flash');
  });

  it('says so plainly when no change is coming', async () => {
    const { view } = await renderPage();
    assert.match(view.textContent, /Nothing waiting for a change/);
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
    assert.equal(field(view, 'Max TTFT').querySelector('input').value, '120');
    assert.equal(field(view, 'Max idle time').querySelector('input').value, '30');
    assert.equal(field(view, 'Max queue size').querySelector('input').value, '150');
  });

  it('explains each one where hovering finds it', async () => {
    // They trade against each other and against the machine. A number with no
    // explanation is a number nobody dares change.
    const { view } = await renderPage();
    assert.match(field(view, 'Max TTFT').getAttribute('title'), /streaming/);
    assert.match(field(view, 'Max idle time').getAttribute('title'), /17 tokens/);
    assert.match(field(view, 'Max queue size').getAttribute('title'), /thread/);
  });

  it('sends every limit when one is saved', async () => {
    const context = installDom({ '/api/gateway': STATS,
                                 'PATCH /api/gateway': STATS });
    const { render } = await import(`../../ai_lab/web/js/views/gateway.js?${Math.random()}`);
    await render(context.view);
    await settle();
    type(field(context.view, 'Max idle time').querySelector('input'), '45');
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
    const { render } = await import(`../../ai_lab/web/js/views/gateway.js?${Math.random()}`);
    await render(context.view);
    await settle();
    type(field(context.view, 'Max idle time').querySelector('input'), '45');
    [...context.view.querySelectorAll('button')]
      .find((node) => node.textContent.trim() === 'Save').click();
    await settle();
    const { stopRefreshing } = await import('../../ai_lab/web/js/views/gateway.js');
    stopRefreshing();
    // A refusal takes the page, rather than whispering at the bottom of it.
    const dialog = context.document.querySelector('dialog');
    const told = dialog ? dialog.textContent : '';
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

  it('says nothing about a key, because there is nothing to say', async () => {
    const { view } = await renderPage();
    assert.equal(view.textContent.includes('API key'), false);
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
    for (const place of ['at-activity', 'at-address', 'at-limits', 'at-queue']) {
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
    type(field(view, 'Max idle time').querySelector('input'), '45');
    assert.equal(saveButton(view).disabled, false);
  });

  it('sleeps again when the value is typed back', async () => {
    const { view } = await renderPage();
    const input = field(view, 'Max idle time').querySelector('input');
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

  it('says it is waiting for room when nothing runs but somebody wants in', async () => {
    // "Waiting to swap" said there was one place and it was taken. With a
    // budget there may be room for another model instead of a swap, so the
    // word is what is actually true: waiting for room.
    const { view } = await renderPage({
      loaded: [], in_flight: 0, waiting: 1, busy: true,
      waiting_for: [{ instance_id: 'reviewer', waiting: 1, longest_wait_s: 1 }],
    });
    assert.match(view.textContent, /waiting for room/);
  });
});

describe('what the machine says about its memory', () => {
  it('shows each pool, and how much of it is used', async () => {
    // The binding constraint, and now the thing that decides how many models
    // fit at once.
    const { view } = await renderPage();
    assert.match(view.textContent, /Card mem\s*3546 \/ 32623 MB/);
    assert.match(view.textContent, /System mem\s*8200 \/ 64000 MB/);
  });

  it('ends with what a model could actually have', async () => {
    const { view } = await renderPage();
    assert.match(view.textContent, /Available mem\s*76685 MB/);
  });

  it('reads the same numbers the admission decision would', async () => {
    // Not a second reading computed here. Two figures meaning the same thing
    // and worked out twice eventually disagree, and the one on screen is the
    // one nobody checks.
    const { view } = await renderPage({
      memory: { unified: false, available_mb: 1234, capacity_mb: 32623,
                held_by_models_mb: 0,
                pools: [{ name: 'card', kind: 'dedicated', total_mb: 32623,
                          used_mb: 31389, used_by_models_mb: 31389,
                          reserve_mb: 0, free_mb: 1234, capacity_mb: 32623,
                          available_mb: 1234 }] },
    });
    assert.match(view.textContent, /Available mem\s*1234 MB/);
  });

  it('shows the temperature', async () => {
    const { view } = await renderPage();
    assert.match(view.textContent, /GPU temp\s*32 °C/);
  });

  it('does not report utilisation', async () => {
    // An instantaneous sample, so a five-second page lands between requests
    // more often than not and reads nought per cent on a machine that is
    // working steadily.
    const { view } = await renderPage();
    assert.equal(view.textContent.includes('busy'), false);
  });

  it('says nothing about heat where there is nothing to read', async () => {
    // Apple silicon reports none. A dash would be a number meaning something
    // else, and one pool shown twice would double the machine.
    const { view } = await renderPage({
      card: { temperature_c: null },
      memory: { unified: true, available_mb: 80088, capacity_mb: 114688,
                held_by_models_mb: 21000,
                pools: [{ name: 'machine', kind: 'unified', total_mb: 131072,
                          used_mb: 34600, used_by_models_mb: 21000,
                          reserve_mb: 16384, free_mb: 96472,
                          capacity_mb: 114688, available_mb: 80088 }] },
    });
    assert.match(view.textContent, /System mem\s*34600 \/ 131072 MB/);
    assert.equal(view.textContent.includes('°C'), false);
    assert.equal(view.textContent.includes('Card mem'), false,
                 'unified memory is one pool, reported twice');
  });

  it('says nothing at all when there is no accelerator', async () => {
    const { view } = await renderPage({ card: {} });
    assert.equal(view.textContent.includes('GPU mem'), false);
  });
});

describe('the machine behind the card', () => {
  it('shows how much of the system memory is in use', async () => {
    // Where a model is split between the card and here, this is the half that
    // does not show on the card reading.
    const { view } = await renderPage();
    assert.match(view.textContent, /System mem\s*8200 \/ 64000 MB/);
  });
});
