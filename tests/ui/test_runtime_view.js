// The Models page, from the user's side.
//
// Each test names what is on screen, what a person would expect from it, and
// checks that is what happens. Every one of these corresponds to a fault that
// reached the user because nothing was watching.

import assert from 'node:assert/strict';
import { after, before, beforeEach, describe, it } from 'node:test';

import { installDom, button, settle } from './support/dom.js';

const INSTANCE = {
  id: 'qwen-coder', engine: 'llamacpp',
  model_id: 'gguf/qwen/qwen', port: 8080,
  params: { context_size: 32768, parallel: 1, temperature: 0.8 },
  running: true, enabled: true, pid: 4541, ready: true, web_ui: true,
  last_operation: null,
};
const MODEL = {
  id: 'gguf/qwen/qwen', name: 'qwen', format: 'gguf',
  size_bytes: 21 * 1024 ** 3, file_count: 1, complete: true, missing: [],
};
const ENGINE = {
  id: 'llamacpp', name: 'llama.cpp', available: true, reason: '', binary: '/bin/llama-server',
  formats: ['gguf'],
  params: [
    { key: 'context_size', label: 'Context size', kind: 'int', default: 32768,
      minimum: 512, maximum: 1048576, choices: [], help: '', group: 'memory', advanced: false },
    { key: 'temperature', label: 'Temperature', kind: 'float', default: 0.8,
      minimum: 0, maximum: 2, choices: [], help: '', group: 'generation', advanced: false },
    { key: 'top_k', label: 'Top-k', kind: 'int', default: 40,
      minimum: 0, maximum: 1000, choices: [], help: '', group: 'generation', advanced: true },
  ],
};

function responses(overrides = {}) {
  return {
    '/api/instances': [INSTANCE],
    '/api/models': [MODEL],
    '/api/settings': { title: 'AI-Lab', engines: [ENGINE], repositories: [],
                       accelerator: {}, host: {} },
    ...overrides,
  };
}

async function renderPage(overrides = {}) {
  const context = installDom(responses(overrides));
  const { render } = await import(`../../ai_lab/web/js/views/runtime.js?${Math.random()}`);
  await render(context.view);
  await settle();
  return context;
}

describe('the Models page', () => {
  it('never puts the word null on the page', async () => {
    // replaceChildren turns a null child into a text node reading "null".
    const { view } = await renderPage();
    assert.equal(view.textContent.includes('null'), false, view.textContent);
  });

  it('keeps file deletion out of Models, but lets a stopped entry be removed', async () => {
    const { view } = await renderPage();
    assert.equal(view.textContent.includes('Delete'), false);
    assert.ok(button(view, 'Remove').disabled, 'a running entry should be unloaded first');
  });

  it('asks before removing a stopped entry and explains that files remain', async () => {
    const stopped = { ...INSTANCE, running: false, ready: false };
    const { view, calls } = await renderPage({ '/api/instances': [stopped] });
    button(view, 'Remove').click();
    await settle();

    const dialog = document.querySelector('dialog.confirm');
    assert.match(dialog.textContent, /downloaded model files remain in Library/i);
    assert.equal(document.activeElement.textContent.trim(), 'Cancel');
    assert.equal(calls.filter((call) => call.method === 'DELETE').length, 0);

    button(dialog, 'Remove').click();
    await settle();
    assert.ok(calls.some((call) => call.method === 'DELETE'
      && call.path.includes('/api/instances/qwen-coder')));
  });

  it('shows a full bar for a model that is loaded', async () => {
    // A fresh page has received no events yet, so the bar is drawn from the
    // reported state. It is the only thing saying the model is up: the row no
    // longer spends a word on it.
    const { view } = await renderPage();
    assert.equal(view.querySelector('.bar > span').style.width, '100%');
    assert.match(view.querySelector('.row.instance').title, /ready/);
  });

  it('shows an empty bar for a model that is stopped', async () => {
    const { view } = await renderPage({
      '/api/instances': [{ ...INSTANCE, running: false, ready: false }],
    });
    assert.equal(view.querySelector('.bar > span').style.width, '0%');
    assert.match(view.querySelector('.row.instance').title, /stopped/);
  });

  it('says which engine runs the entry, on the line itself', async () => {
    // llama.cpp and vLLM behave differently enough that a row without it says
    // very little. This one belongs on the line, not in the tooltip.
    const { view } = await renderPage();
    assert.match(view.querySelector('.pill.engine').textContent, /llama/i);
  });

  it('shows the weight format at the right of the line', async () => {
    const { view } = await renderPage();
    assert.equal(view.querySelector('.pill.format').textContent, 'gguf');
  });

  it('does not offer to rename an entry', async () => {
    // The name is what requests carry, so changing it would break whatever is
    // already sending it. Renaming is deleting and adding, which the page
    // already offers.
    const { view } = await renderPage();
    button(view, 'Settings').click();
    await settle();
    assert.equal(view.textContent.includes('Label'), false);
  });

  it('reads sensibly when the label has been cleared', async () => {
    // An empty label is allowed. The row then says what will run - the engine
    // and the model - which is all some entries ever needed.
    const { view } = await renderPage({
      '/api/instances': [{ ...INSTANCE, name: '' }],
    });
    assert.equal(view.textContent.includes('null'), false);
    assert.equal(view.textContent.includes('undefined'), false);
    assert.match(view.querySelector('.pill.engine').textContent, /llama/i);
    assert.match(view.querySelector('.row.instance > .ident').textContent, /qwen/);
  });

  it('keeps port, context and timings in the tooltip rather than on the line', async () => {
    const { view } = await renderPage();
    const line = view.querySelector('.row.instance');
    assert.match(line.title, /port 8080/);
    assert.match(line.title, /ctx/);
    assert.equal(view.textContent.includes('port 8080'), false,
                 'the line is spending screen on something the tooltip carries');
  });

  it('shapes the way in like the pills, not like the buttons', async () => {
    // It belongs with the pills: they describe the model, and this one says
    // the model is answering and here is the door. Sharing their class rather
    // than copying their rules keeps the three the same size and shape
    // however those rules change.
    const { view } = await renderPage();
    const chat = view.querySelector('.chat-link');
    assert.ok(chat.classList.contains('pill'), 'it is not shaped like a pill');
    assert.equal(chat.classList.contains('action'), false,
                 'it is dressed as one of the buttons');
  });

  it('offers a way to talk to a model that is ready', async () => {
    // llama.cpp serves its own chat page on the model's port.
    const { view } = await renderPage();
    const chat = view.querySelector('.chat-link');
    assert.ok(chat, 'no way to reach the model');
    assert.equal(chat.getAttribute('href'), 'http://localhost:8080/');
    assert.equal(chat.getAttribute('target'), '_blank');
  });

  it('offers no chat link before the model answers', async () => {
    // A link to a blank tab is worse than no link.
    const { view } = await renderPage({
      '/api/instances': [{ ...INSTANCE, ready: false }],
    });
    assert.equal(view.querySelector('.chat-link'), null);
  });

  it('offers no chat link when the engine serves no page', async () => {
    // vLLM serves an API and nothing to look at, and llama.cpp built without
    // node is the same. A link that could never work is worse than none.
    const { view } = await renderPage({
      '/api/instances': [{ ...INSTANCE, web_ui: false }],
    });
    assert.equal(view.querySelector('.chat-link'), null);
  });

  it('builds the chat address from the page, not from the server', async () => {
    // The manager may be reached by name, by address or through a tunnel; the
    // engine is on the same host under a different port.
    const { view } = await renderPage({
      '/api/instances': [{ ...INSTANCE, port: 8099 }],
    });
    assert.equal(view.querySelector('.chat-link').getAttribute('href'),
                 'http://localhost:8099/');
  });

  it('offers Unload for something running, and Load for something stopped', async () => {
    const running = await renderPage();
    assert.doesNotThrow(() => button(running.view, 'Unload'));

    const stopped = await renderPage({
      '/api/instances': [{ ...INSTANCE, running: false, ready: false }],
    });
    assert.doesNotThrow(() => button(stopped.view, 'Load'));
  });

  it('explains why Apply & reload is unavailable instead of just disabling it', async () => {
    const { view } = await renderPage();
    const apply = button(view, 'Apply & reload');
    assert.equal(apply.disabled, true);
    assert.match(apply.title, /Settings/, 'a dead button with no explanation');
  });

  it('reveals the settings form when Settings is pressed', async () => {
    const { view } = await renderPage();
    assert.equal(view.querySelector('.settings'), null);
    button(view, 'Settings').click();
    await settle();
    assert.ok(view.querySelector('.settings'), 'nothing appeared');
    assert.match(view.textContent, /Context size/);
    assert.match(view.textContent, /Temperature/);
  });

  it('separates settings that need a reload from ones a client can override', async () => {
    const { view } = await renderPage();
    button(view, 'Settings').click();
    await settle();
    assert.match(view.textContent, /Memory and capacity/);
    assert.match(view.textContent, /Generation defaults/);
  });

  it('shows every setting at once, with nothing hidden behind a toggle', async () => {
    const { view } = await renderPage();
    button(view, 'Settings').click();
    await settle();
    assert.match(view.textContent, /Context size/);
    assert.match(view.textContent, /Temperature/);
    assert.match(view.textContent, /Top-k/, 'a setting was hidden');
    assert.equal(view.textContent.toLowerCase().includes('advanced'), false,
                 'nothing should be tucked away');
  });

  it('offers to add a model, and can be backed out of', async () => {
    const { view } = await renderPage({
      '/api/instances/new': { port: 8081, engines: [ENGINE], models: [MODEL] },
    });
    button(view, '+ Add model').click();
    await settle();
    assert.match(view.textContent, /Add a model/);
    // A field's value is not part of textContent, so it has to be read.
    const port = [...view.querySelectorAll('input[type="number"]')]
      .find((item) => item.value === '8081');
    assert.ok(port, 'no port was suggested');
    button(view, 'Cancel').click();
    await settle();
    assert.equal(view.textContent.includes('Add a model'), false);
  });

  it('says so plainly when nothing is configured yet', async () => {
    const { view } = await renderPage({ '/api/instances': [] });
    assert.match(view.textContent, /Nothing configured yet/);
  });
});

describe('removing an entry', () => {
  it('is offered here because Library refuses to delete a model in use', async () => {
    // The two are a pair. Deleting a model in Library is refused while an
    // entry points at it, and that refusal sends the reader to this button.
    // Without it, the entry cannot be removed and the model cannot be
    // deleted — a dead end.
    const { view } = await renderPage({
      '/api/instances': [{ ...INSTANCE, running: false, ready: false }],
    });
    assert.doesNotThrow(() => button(view, 'Remove'));
  });

  it('is refused while the entry is running', async () => {
    const { view } = await renderPage();
    assert.equal(button(view, 'Remove').disabled, true);
  });
});

describe('stopping a model that is busy', () => {
  // The buttons on this page reach the engines directly, past the queue that
  // keeps agent requests from interrupting each other. Pressing Unload during
  // a streamed answer used to kill it mid sentence, and the agent on the other
  // end saw a connection that simply stopped.

  const BUSY = {
    __status: 409,
    error: 'coder is answering a request right now.',
    busy: { instance_id: 'coder', answering: true },
  };

  // Refuses the first attempt, accepts one that says force.
  function refuseUntilForced(record) {
    return (path, options) => {
      const body = options.body ? JSON.parse(options.body) : {};
      record.push(body);
      return body.force ? { ok: true, total_ms: 900 } : BUSY;
    };
  }

  function dialog() {
    return document.querySelector('dialog');
  }

  it('asks instead of printing a refusal and leaving you stuck', async () => {
    const { view } = await renderPage({
      '/api/instances/qwen-coder/unload': refuseUntilForced([]),
    });
    button(view, 'Unload').click();
    await settle();
    assert.ok(dialog(), 'no dialog appeared');
    assert.match(dialog().textContent, /coder is busy/);
  });

  it('says what going ahead costs, in words rather than a status code', async () => {
    const { view } = await renderPage({
      '/api/instances/qwen-coder/unload': refuseUntilForced([]),
    });
    button(view, 'Unload').click();
    await settle();
    assert.match(dialog().textContent, /cuts that answer off/);
  });

  it('leaves the model alone when the answer is no', async () => {
    const sent = [];
    const { view } = await renderPage({
      '/api/instances/qwen-coder/unload': refuseUntilForced(sent),
    });
    button(view, 'Unload').click();
    await settle();
    button(dialog(), 'Cancel').click();
    await settle();
    assert.equal(sent.length, 1, 'it was stopped despite the answer being no');
    assert.equal(sent[0].force, undefined);
  });

  it('goes ahead when the answer is yes', async () => {
    // A model wedged in a bad state has to be stoppable. The point of the
    // dialog is that it becomes a decision, not that it becomes impossible.
    const sent = [];
    const { view } = await renderPage({
      '/api/instances/qwen-coder/unload': refuseUntilForced(sent),
    });
    button(view, 'Unload').click();
    await settle();
    button(dialog(), 'Stop it anyway').click();
    await settle();
    assert.equal(sent.length, 2, 'the forced attempt was never sent');
    assert.equal(sent[1].force, true);
  });

  it('says a model is loading rather than answering, when that is the case', async () => {
    const { view } = await renderPage({
      '/api/instances/qwen-coder/unload': {
        __status: 409, error: 'busy',
        busy: { instance_id: 'coder', answering: false },
      },
    });
    button(view, 'Unload').click();
    await settle();
    assert.match(dialog().textContent, /still loading/);
  });

  it('does not offer to force past a failure that is not about a busy card', async () => {
    // The failure is still shown — every failure is, now — but offering to
    // force past a model that will not fit would be nonsense.
    const { view } = await renderPage({
      '/api/instances/qwen-coder/unload': {
        __status: 400, error: 'Could not stop it: no such unit',
      },
    });
    button(view, 'Unload').click();
    await settle();
    assert.ok(dialog(), 'the failure was not shown at all');
    assert.match(dialog().textContent, /no such unit/);
    assert.equal(dialog().textContent.includes('Stop it anyway'), false,
                 'it offered to force past something forcing cannot fix');
  });
});

describe('what a setting explains about itself', () => {
  // The explanations are long on purpose — they say what a setting costs, and
  // that is the part worth having. Printed under every label they turned a
  // form of a dozen settings into a wall of prose with the fields lost in it.
  // You read one when deciding about that one setting, and never again.

  const EXPLAINED = {
    ...ENGINE,
    params: [
      { key: 'context_size', label: 'Context size', kind: 'int', default: 32768,
        minimum: 512, maximum: 1048576, choices: [], group: 'memory',
        help: 'Longest prompt plus answer the model will accept.' },
      { key: 'threads', label: 'CPU threads', kind: 'int', default: -1,
        minimum: -1, maximum: 256, choices: [], group: 'memory', help: '' },
    ],
  };

  async function openSettings() {
    const { view } = await renderPage({
      '/api/settings': { title: 'AI-Lab', engines: [EXPLAINED], repositories: [],
                         accelerator: {}, host: {} },
    });
    button(view, 'Settings').click();
    await settle();
    return view;
  }

  it('keeps the explanation off the page', async () => {
    const view = await openSettings();
    assert.equal(view.textContent.includes('Longest prompt plus answer'), false,
                 'the explanation was printed under the label');
  });

  it('puts it where hovering finds it', async () => {
    const view = await openSettings();
    const field = [...view.querySelectorAll('label.field')]
      .find((node) => node.textContent.includes('Context size'));
    assert.equal(field.getAttribute('title'),
                 'Longest prompt plus answer the model will accept.');
  });

  it('marks a label that has something to say', async () => {
    // Otherwise the tooltip is only found by accident.
    const view = await openSettings();
    const explained = [...view.querySelectorAll('label.field')]
      .find((node) => node.textContent.includes('Context size'));
    assert.ok(explained.classList.contains('explained'));
  });

  it('leaves a label with nothing to say unmarked', async () => {
    const view = await openSettings();
    const plain = [...view.querySelectorAll('label.field')]
      .find((node) => node.textContent.includes('CPU threads'));
    assert.equal(plain.classList.contains('explained'), false);
    assert.equal(plain.getAttribute('title'), '');
  });

  it('still shows the label itself', async () => {
    const view = await openSettings();
    assert.match(view.textContent, /Context size/);
  });
});

describe('a model started differently from how it is configured', () => {
  // A request can ask for a bigger context than the entry was set up for. The
  // settings are not saved, so the row would otherwise show a number the
  // running model is not using, with nothing to say so.

  it('says what it was actually started with', async () => {
    const { view } = await renderPage({
      '/api/instances': [{ ...INSTANCE, active_params: { context_size: 65536 } }],
    });
    const row = view.querySelector('.row.instance');
    assert.match(row.getAttribute('title'), /started by a request with context_size 65536/);
  });

  it('says the change will not survive a reload', async () => {
    const { view } = await renderPage({
      '/api/instances': [{ ...INSTANCE, active_params: { context_size: 65536 } }],
    });
    assert.match(view.querySelector('.row.instance').getAttribute('title'),
                 /not saved/);
  });

  it('says nothing when the model runs as configured', async () => {
    const { view } = await renderPage({
      '/api/instances': [{ ...INSTANCE, active_params: {} }],
    });
    assert.equal(
      view.querySelector('.row.instance').getAttribute('title').includes('started by a request'),
      false);
  });

  it('copes with an entry that has no such field at all', async () => {
    const { view } = await renderPage();
    assert.doesNotThrow(() => view.querySelector('.row.instance').getAttribute('title'));
  });
});

describe('a load that fails', () => {
  // A load takes between four seconds and a minute, so you look away. The
  // status line is right for "finished in 12.7 s" and wrong for "it would not
  // start": the page is long, that line sits below it with no border, and the
  // next action wipes it. Twice in one afternoon the answer was on the machine
  // and not on the screen.

  const REFUSED = {
    ok: false, total_ms: 73189, instance_id: 'qwen-coder', kind: 'load',
    steps: [],
    error: 'The engine stopped while loading: ValueError: To serve at least '
         + "one request with the model's max seq len (131072), 12.0 GiB KV "
         + 'cache is needed, which is larger than the available KV cache '
         + 'memory (11.01 GiB). Based on the available memory, the estimated '
         + 'maximum model length is 120256.',
  };

  // The status line is attached by the entry point, not by a view, so a test
  // that renders a view on its own has to do it.
  async function attachFooter() {
    const { attachStatus } = await import('../../ai_lab/web/js/status.js');
    attachStatus(document.getElementById('status'));
  }

  async function failLoad() {
    const { view } = await renderPage({
      '/api/instances': [{ ...INSTANCE, running: false, ready: false }],
      '/api/instances/qwen-coder/load': REFUSED,
    });
    await attachFooter();
    button(view, 'Load').click();
    await settle();
    return document.querySelector('dialog');
  }

  it('takes the page rather than whispering at the bottom of it', async () => {
    assert.ok(await failLoad(), 'nothing was shown');
  });

  it('keeps the number that would have worked', async () => {
    // The whole reason for showing the engine's own words: it worked out the
    // largest context that fits and said so.
    assert.match((await failLoad()).textContent, /120256/);
  });

  it('shows the explanation whole, not trimmed to a line', async () => {
    const text = (await failLoad()).textContent;
    assert.match(text, /12.0 GiB KV cache is needed/);
    assert.match(text, /11.01 GiB/);
  });

  it('says which action failed', async () => {
    assert.match((await failLoad()).textContent, /failed/);
  });

  it('leaves the message in the status line as well', async () => {
    // For anyone who dismisses it and then wonders what it said.
    await failLoad();
    assert.match(document.getElementById('status').textContent, /120256/);
  });

  it('says nothing extra when a load succeeds', async () => {
    const { view } = await renderPage({
      '/api/instances': [{ ...INSTANCE, running: false, ready: false }],
      '/api/instances/qwen-coder/load': { ok: true, total_ms: 12700, steps: [] },
    });
    await attachFooter();
    button(view, 'Load').click();
    await settle();
    assert.equal(document.querySelector('dialog'), null,
                 'a success should not interrupt');
    assert.match(document.getElementById('status').textContent, /12.7 s/);
  });
});

describe('reading what an engine is saying', () => {
  // A model that answers oddly or slowly is talking about itself the whole
  // time, and until now that meant ssh and journalctl.

  const LINES = ['INFO server listening on 8080', 'INFO slot released'];

  // The panel is one panel for the whole page, so it keeps its state in the
  // module. In a browser that is right — there is one page. Here each test
  // builds a fresh document, and a panel left open belongs to the last one.
  // It also runs a timer, which would keep the test process alive.
  beforeEach(async () => {
    const { closeLogs } = await import('../../ai_lab/web/js/logpane.js');
    closeLogs();
  });
  after(async () => {
    const { closeLogs } = await import('../../ai_lab/web/js/logpane.js');
    closeLogs();
  });

  function pane() {
    return document.querySelector('.logpane');
  }

  async function withLogs(overrides = {}) {
    const { view } = await renderPage({
      '/api/instances/qwen-coder/logs': { id: 'qwen-coder', running: true, lines: LINES },
      ...overrides,
    });
    return view;
  }

  it('offers a log button on the row', async () => {
    const view = await withLogs();
    assert.doesNotThrow(() => button(view, 'Log'));
  });

  it('refuses it for a model that is not running', async () => {
    // A stopped model has a journal on Linux and no log at all on macOS, so
    // offering it would work on one machine and not the other.
    const view = await withLogs({
      '/api/instances': [{ ...INSTANCE, running: false, ready: false }],
    });
    assert.equal(button(view, 'Log').disabled, true);
  });

  it('opens a panel with what the engine printed', async () => {
    const view = await withLogs();
    button(view, 'Log').click();
    await settle();
    assert.ok(pane(), 'no panel appeared');
    assert.match(pane().textContent, /slot released/);
  });

  it('names the model the panel is about', async () => {
    const view = await withLogs();
    button(view, 'Log').click();
    await settle();
    assert.match(pane().textContent, /qwen-coder/);
  });

  it('closes again when the same button is pressed', async () => {
    // The button is the toggle, so it has to say which way it will go.
    const view = await withLogs();
    button(view, 'Log').click();
    await settle();
    assert.doesNotThrow(() => button(document.body, 'Hide log'));
    button(document.body, 'Hide log').click();
    await settle();
    assert.equal(pane(), null);
  });

  it('sits outside the list, so a long page does not carry it off', async () => {
    const view = await withLogs();
    button(view, 'Log').click();
    await settle();
    assert.equal(view.contains(pane()), false,
                 'the panel was drawn inside the scrolling list');
  });

  it('says so when the engine has printed nothing yet', async () => {
    const view = await withLogs({
      '/api/instances/qwen-coder/logs': { id: 'qwen-coder', running: true, lines: [] },
    });
    button(view, 'Log').click();
    await settle();
    assert.match(pane().textContent, /printed nothing yet/);
  });

  it('shows the reason rather than emptying itself when the read fails', async () => {
    const view = await withLogs({
      '/api/instances/qwen-coder/logs': {
        __status: 400, error: 'the manager is not in the systemd-journal group',
      },
    });
    button(view, 'Log').click();
    await settle();
    assert.match(pane().textContent, /systemd-journal/);
  });
});

describe('the buttons line up down the page', () => {
  // A row with one button fewer than the rows above it puts every button on
  // that line in a different place, so the column of Loads and Unloads you are
  // aiming at stops being a column. Every row carries the same buttons; the
  // ones that would do nothing are greyed out and say why.

  // Load and Unload are one button wearing two labels, so they count as one
  // slot. What must not differ is how many slots there are and where they sit.
  function slots(view, index = 0) {
    return [...view.querySelectorAll('.card.instance')[index]
      .querySelectorAll('.row.instance .action')]
      .map((node) => node.textContent.trim())
      .map((label) => (label === 'Unload' ? 'Load' : label));
  }

  const VLLM = {
    ...INSTANCE, id: 'coder-fast', name: 'Fast', engine: 'vllm',
    running: false, ready: false, web_ui: false,
  };

  it('gives every row the same buttons, whether or not it offers a way in', async () => {
    const { view } = await renderPage({
      '/api/instances': [
        INSTANCE,
        { ...INSTANCE, id: 'other', name: 'Other', running: false, ready: false },
      ],
    });
    assert.deepEqual(slots(view, 0), slots(view, 1));
  });

  function chat(view) {
    return view.querySelector('.chat-link');
  }

  it('offers a way in to a model that is answering', async () => {
    const { view } = await renderPage();
    assert.ok(chat(view), 'no link');
    assert.match(chat(view).getAttribute('href'), /:8080\//);
  });

  it('offers nothing for a model that is not running', async () => {
    const { view } = await renderPage({
      '/api/instances': [{ ...INSTANCE, running: false, ready: false }],
    });
    assert.equal(chat(view), null);
  });

  it('offers nothing for a running vLLM model, which has no page to offer', async () => {
    // The reason for taking the greyed-out button away: it sat on every vLLM
    // row promising something none of them could ever do, however long you
    // waited.
    const { view } = await renderPage({
      '/api/instances': [{ ...VLLM, running: true, ready: true }],
    });
    assert.equal(chat(view), null);
  });

  it('comes first, so its absence moves nothing after it', async () => {
    // The right-hand group sits against the right edge and is only as wide as
    // its contents, so removing something moves whatever is to its left. First
    // means nothing moves at all.
    const { view } = await renderPage();
    const group = view.querySelector('.row.instance > .inline:not(.ident)');
    assert.ok(group.firstElementChild.classList.contains('chat-link'),
              `first was ${group.firstElementChild.className}`);
  });

  it('is not one of the buttons that act on the model', async () => {
    const { view } = await renderPage();
    assert.equal(chat(view).classList.contains('action'), false);
    assert.equal(chat(view).getAttribute('target'), '_blank');
  });

  it('keeps the same buttons for a vLLM row as for a llama.cpp one', async () => {
    const { view } = await renderPage({ '/api/instances': [INSTANCE, VLLM] });
    assert.deepEqual(slots(view, 0), slots(view, 1));
  });
});

describe('one page does not lay out another', () => {
  it('leaves the Models page as it was', async () => {
    // The Gateway page used to put its layout class on the page container,
    // which every tab shares, so Models and Library came out in columns after
    // a visit to it.
    const context = installDom(responses({
      '/api/gateway': {
        current: null, current_settings: {}, busy: false, holder: null,
        in_flight: 0, places: 0, switching: false, waiting: 0, waiting_for: [],
        longest_wait_s: 0, max_waiting: 150, first_byte_s: 120,
        between_bytes_s: 30, requests: 0, switches: 0, average_wait_s: 0,
        average_switch_s: 0, total_switch_s: 0, last_error: '',
        shapes: [{ path: '/v1/chat/completions', models: ['qwen-coder'],
                   engines: ['llama.cpp'] }],
        recent: [],
      },
    }));
    const gateway = await import(`../../ai_lab/web/js/views/gateway.js?${Math.random()}`);
    await gateway.render(context.view);
    await settle();
    const { render } = await import(`../../ai_lab/web/js/views/runtime.js?${Math.random()}`);
    await render(context.view);
    await settle();
    const { closeLogs } = await import('../../ai_lab/web/js/logpane.js');
    closeLogs();
    gateway.stopRefreshing();
    assert.equal(context.view.className, '',
                 `the container kept "${context.view.className}"`);
    assert.equal(context.view.querySelector('.gateway-grid'), null);
    // Not closing the window: the timers this test started are stopped above,
    // and closing it out from under whatever jsdom is still finishing raises
    // from inside jsdom rather than from anything under test.
  });
});

describe('settings opened under a row', () => {
  it('is separated from the row it belongs to', async () => {
    // Without a line the first field sits directly under the name and reads
    // as part of it.
    const { view } = await renderPage();
    button(view, 'Settings').click();
    await settle();
    assert.ok(view.querySelector('.settings-open'),
              'the settings block is not marked off from the row');
  });

  it('is not there when the settings are closed', async () => {
    const { view } = await renderPage();
    assert.equal(view.querySelector('.settings-open'), null);
  });
});

describe('the name a request has to carry', () => {
  it('is the name on the row', async () => {
    // One name: a request carries it and a person reads it. There used to be a
    // label as well, and it became a second way of saying the same thing.
    const { view } = await renderPage();
    const row = view.querySelector('.row.instance');
    assert.match(row.querySelector('strong').textContent, /qwen-coder/);
  });

  it('says what it is for', async () => {
    const { view } = await renderPage();
    assert.match(view.querySelector('.row.instance strong').getAttribute('title'),
                 /send as "model"/);
  });

  it('shows the model beside it, smaller', async () => {
    const { view } = await renderPage();
    assert.match(view.querySelector('.row.instance .model').textContent, /qwen/);
  });
});

describe('adding a model', () => {
  async function openAdd() {
    const { view } = await renderPage({
      '/api/instances/new': { port: 8081, engines: [ENGINE], models: [MODEL] },
    });
    button(view, '+ Add model').click();
    await settle();
    return view;
  }

  it('shows the rules where the name is typed', async () => {
    // Not in a message after it is refused. This is the only name the entry
    // has, and it cannot be changed later.
    const view = await openAdd();
    const field = [...view.querySelectorAll('label.field')]
      .find((node) => node.textContent.includes('Name'));
    assert.match(field.textContent, /lower-case letters, digits and hyphens/);
    assert.match(field.textContent, /sent as "model"/);
    assert.match(field.textContent, /cannot be changed later/);
  });

  it('sends what was typed, unchanged', async () => {
    // It used to lowercase a label and turn everything else into hyphens, so
    // the name a request carried was decided by a sentence written for
    // reading.
    const context = installDom(responses({
      '/api/instances/new': { port: 8081, engines: [ENGINE], models: [MODEL] },
      'POST /api/instances': { id: 'gemma-31b-nvfp4' },
    }));
    const { render } = await import(`../../ai_lab/web/js/views/runtime.js?${Math.random()}`);
    await render(context.view);
    await settle();
    button(context.view, '+ Add model').click();
    await settle();
    const field = [...context.view.querySelectorAll('label.field')]
      .find((node) => node.textContent.includes('Name'));
    field.querySelector('input').value = 'gemma-31b-nvfp4';
    button(context.view, 'Add').click();
    await settle();
    const posted = context.calls.find((call) => call.method === 'POST');
    assert.equal(JSON.parse(posted.body).id, 'gemma-31b-nvfp4');
  });
});
