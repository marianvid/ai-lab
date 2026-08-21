// The Models page, from the user's side.
//
// Each test names what is on screen, what a person would expect from it, and
// checks that is what happens. Every one of these corresponds to a fault that
// reached the user because nothing was watching.

import assert from 'node:assert/strict';
import { after, before, describe, it } from 'node:test';

import { installDom, button, settle } from './support/dom.js';

const INSTANCE = {
  id: 'qwen-coder', name: 'Coding', engine: 'llamacpp',
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

  it('lets the label be edited, and sends it when the model is loaded', async () => {
    const { view, calls } = await renderPage({
      '/api/instances': [{ ...INSTANCE, running: false, ready: false }],
    });
    button(view, 'Settings').click();
    await settle();
    const field = [...view.querySelectorAll('input')]
      .find((input) => input.value === 'Coding');
    assert.ok(field, 'the label cannot be edited');
    field.value = 'Coding, renamed';
    button(view, 'Load').click();
    await settle();
    const saved = calls.find((call) => call.method === 'PATCH');
    assert.ok(saved, 'nothing was saved before loading');
    assert.equal(JSON.parse(saved.body).name, 'Coding, renamed');
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

  it('makes the chat link look like the buttons beside it', async () => {
    // Left alone a link takes the browser's colour, which reads as a different
    // kind of thing sitting in a row of buttons.
    const { view } = await renderPage();
    const chat = [...view.querySelectorAll('a')]
      .find((item) => item.textContent.trim() === 'Chat');
    assert.ok(chat.classList.contains('action'));
  });

  it('offers a way to talk to a model that is ready', async () => {
    // llama.cpp serves its own chat page on the model's port.
    const { view } = await renderPage();
    const chat = [...view.querySelectorAll('a')]
      .find((item) => item.textContent.trim() === 'Chat');
    assert.ok(chat, 'no way to reach the model');
    assert.equal(chat.getAttribute('href'), 'http://localhost:8080/');
    assert.equal(chat.getAttribute('target'), '_blank');
  });

  it('offers no chat link before the model answers', async () => {
    // A link to a blank tab is worse than no link.
    const { view } = await renderPage({
      '/api/instances': [{ ...INSTANCE, ready: false }],
    });
    assert.equal([...view.querySelectorAll('a')]
      .some((item) => item.textContent.trim() === 'Chat'), false);
  });

  it('offers no chat link when the engine serves no page', async () => {
    // Building that page needs npm; a machine without node gets an API only.
    const { view } = await renderPage({
      '/api/instances': [{ ...INSTANCE, web_ui: false }],
    });
    assert.equal([...view.querySelectorAll('a')]
      .some((item) => item.textContent.trim() === 'Chat'), false);
  });

  it('builds the chat address from the page, not from the server', async () => {
    // The manager may be reached by name, by address or through a tunnel; the
    // engine is on the same host under a different port.
    const { view } = await renderPage({
      '/api/instances': [{ ...INSTANCE, port: 8099 }],
    });
    const chat = [...view.querySelectorAll('a')]
      .find((item) => item.textContent.trim() === 'Chat');
    assert.equal(chat.getAttribute('href'), 'http://localhost:8099/');
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

  it('does not ask about failures that have nothing to do with a busy card', async () => {
    // Every other refusal is still a plain message. Offering to force past a
    // model that will not fit would be nonsense.
    const { view } = await renderPage({
      '/api/instances/qwen-coder/unload': {
        __status: 400, error: 'Could not stop it: no such unit',
      },
    });
    button(view, 'Unload').click();
    await settle();
    assert.equal(dialog(), null, 'a dialog appeared for an unrelated failure');
  });
});
