// The small pictures saying what a model can do.
//
// The rule they exist to show is a one-way one: the weights decide what a
// model *can* do, and an entry's settings can only take something away. An
// icon that stays on after the setting that switches it off is a promise the
// running model will not keep, which is worse than no icon at all.

import assert from 'node:assert/strict';
import { after, before, beforeEach, describe, it } from 'node:test';

import { installDom, settle } from './support/dom.js';

const INSTANCE = {
  id: 'gemma-vision', engine: 'vllm',
  model_id: 'nvfp4/gemma', port: 8080,
  params: { context_size: 32768, language_model_only: false },
  running: true, pid: 1, ready: true, web_ui: false, last_operation: null,
};
const MODEL = {
  id: 'nvfp4/gemma', name: 'gemma-4-26b-a4b', format: 'nvfp4',
  size_bytes: 21 * 1024 ** 3, file_count: 5, complete: true, missing: [],
  capabilities: ['images', 'tools'],
};
const ENGINE = {
  id: 'vllm', name: 'vLLM', available: true, reason: '', binary: '/bin/vllm',
  formats: ['nvfp4'],
  params: [
    { key: 'context_size', label: 'Context size', kind: 'int', default: 32768,
      minimum: 512, maximum: 1048576, choices: [], help: '', group: 'memory',
      advanced: false },
    { key: 'language_model_only', label: 'Text only', kind: 'bool',
      default: false, minimum: null, maximum: null, choices: [], help: '',
      group: 'memory', advanced: false },
  ],
};

const REPOSITORY = {
  id: 'nvfp4', name: 'NVFP4 models', path: '/models/nvfp4', format: 'nvfp4',
  writable: true, exists: true, free_bytes: 3.4 * 1024 ** 4,
  total_bytes: 4 * 1024 ** 4,
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

async function renderModels(overrides = {}) {
  const context = installDom(responses(overrides));
  const { render } = await import(`../../ai_lab/web/js/views/runtime.js?${Math.random()}`);
  await render(context.view);
  await settle();
  return context;
}

function icons(view) {
  return [...view.querySelectorAll('svg.capability')]
    .map((svg) => svg.querySelector('title').textContent);
}

describe('what a model can do, on the Models page', () => {
  let dom;
  before(() => { dom = installDom(responses()); });
  after(() => dom.restore && dom.restore());
  beforeEach(() => { document.body.innerHTML = ''; });

  it('shows one picture per capability, and labels each of them', async () => {
    const { view } = await renderModels();
    assert.deepEqual(icons(view), ['Can call tools', 'Can read pictures']);
  });

  it('takes the picture icon away when the entry is set to text only', async () => {
    // vLLM's "Text only" loads a model that can see without the part that
    // sees. The weights still can; this running model cannot.
    const textOnly = { ...INSTANCE,
                       params: { ...INSTANCE.params, language_model_only: true } };
    const { view } = await renderModels({ '/api/instances': [textOnly] });
    assert.deepEqual(icons(view), ['Can call tools']);
  });

  it('leaves the tools icon alone when pictures are switched off', async () => {
    // The setting is about the vision tower. It has nothing to do with the
    // chat template, which is what decides whether tools can be asked for.
    const textOnly = { ...INSTANCE,
                       params: { ...INSTANCE.params, language_model_only: true } };
    const { view } = await renderModels({ '/api/instances': [textOnly] });
    assert.equal(icons(view).includes('Can call tools'), true);
  });

  it('shows nothing for a model whose files claimed nothing', async () => {
    const plain = { ...MODEL, capabilities: [] };
    const { view } = await renderModels({ '/api/models': [plain] });
    assert.deepEqual(icons(view), []);
  });

  it('does not fall over on a model saved before capabilities existed', async () => {
    // An older answer has no such field at all. It should mean "nothing
    // known", not an exception that takes the whole page down.
    const { capabilities: _gone, ...older } = MODEL;
    const { view } = await renderModels({ '/api/models': [older] });
    assert.deepEqual(icons(view), []);
    assert.match(view.textContent, /gemma-vision/);
  });

  it('draws each icon in the colour of the text around it', async () => {
    // So that dark and light need no second set of pictures.
    const { view } = await renderModels();
    const path = view.querySelector('svg.capability path');
    assert.equal(path.getAttribute('stroke'), 'currentColor');
    assert.equal(path.getAttribute('fill'), 'none');
  });
});

describe('what a model can do, in Library', () => {
  let dom;
  before(() => { dom = installDom(responses()); });
  after(() => dom.restore && dom.restore());
  beforeEach(() => { document.body.innerHTML = ''; });

  it('shows the weights own capabilities, since a file on disk has no settings', async () => {
    const context = installDom({
      '/api/models': [{ ...MODEL, id: 'nvfp4/gemma', format: 'nvfp4' }],
      '/api/settings': { title: 'AI-Lab', engines: [ENGINE],
                         repositories: [REPOSITORY], accelerator: {}, host: {} },
      '/api/downloads': [],
      '/api/hf/search': { results: [], hidden: 0 },
      '/api/hf/sets': [],
    });
    const { render } = await import(`../../ai_lab/web/js/views/library.js?${Math.random()}`);
    await render(context.view);
    await settle();
    assert.deepEqual(icons(context.view), ['Can call tools', 'Can read pictures']);
  });
});

describe('saving a setting without touching the card', () => {
  let dom;
  before(() => { dom = installDom(responses()); });
  after(() => dom.restore && dom.restore());
  beforeEach(() => { document.body.innerHTML = ''; });

  function find(view, label) {
    return [...view.querySelectorAll('button')]
      .find((node) => node.textContent.trim() === label);
  }

  function openSettings(view) {
    find(view, 'Settings').click();
  }

  it('offers Save on a stopped entry, where Apply & reload cannot help', async () => {
    // The fault this exists for: unticking a box on a stopped entry did
    // nothing at all. Apply & reload was disabled because nothing was running,
    // and it was the only thing that wrote settings down.
    const stopped = { ...INSTANCE, running: false, ready: false };
    const { view } = await renderModels({ '/api/instances': [stopped] });
    assert.ok(find(view, 'Save'), 'a stopped entry must still be able to save');
    assert.ok(find(view, 'Apply & reload').disabled);
  });

  it('keeps Save asleep until something is actually touched', async () => {
    const stopped = { ...INSTANCE, running: false, ready: false };
    const { view } = await renderModels({ '/api/instances': [stopped] });
    openSettings(view);
    await settle();
    assert.equal(find(view, 'Save').disabled, true, 'nothing changed yet');

    const box = view.querySelector('input[data-key="language_model_only"]');
    box.checked = !box.checked;
    box.dispatchEvent(new window.Event('change', { bubbles: true }));
    assert.equal(find(view, 'Save').disabled, false, 'a touched box should wake Save');
  });

  it('writes the setting down and does not start or stop anything', async () => {
    const stopped = { ...INSTANCE, running: false, ready: false,
                      params: { ...INSTANCE.params, language_model_only: true } };
    const { view, calls } = await renderModels({ '/api/instances': [stopped] });
    openSettings(view);
    await settle();

    const box = view.querySelector('input[data-key="language_model_only"]');
    box.checked = false;
    box.dispatchEvent(new window.Event('change', { bubbles: true }));
    find(view, 'Save').click();
    await settle();

    const wrote = calls.find((call) => call.method === 'PATCH');
    assert.ok(wrote, 'nothing was written down');
    assert.equal(JSON.parse(wrote.body).params.language_model_only, false);
    assert.equal(calls.some((call) => /\/(load|unload)/.test(call.path)), false,
                 'saving a setting must not touch the card');
  });
});

describe('where the icons sit', () => {
  let dom;
  before(() => { dom = installDom(responses()); });
  after(() => dom.restore && dom.restore());
  beforeEach(() => { document.body.innerHTML = ''; });

  it('puts them on the right, immediately before the format', async () => {
    const { view } = await renderModels();
    const right = view.querySelector('.row.instance > div:last-child');
    const marks = [...right.children]
      .filter((node) => node.matches('svg.capability, .pill.format'))
      .map((node) => (node.matches('svg') ? 'icon' : 'format'));
    assert.deepEqual(marks, ['icon', 'icon', 'format']);
  });

  it('leaves the name on the left with nothing but the model beside it', async () => {
    const { view } = await renderModels();
    const left = view.querySelector('.row.instance .ident');
    assert.equal(left.querySelectorAll('svg.capability').length, 0);
  });
});
