// Choosing between installed versions of a package engine.
//
// The rule these defend: what is working is never written over, and never
// tidied away on its own. The previous version is the way back, and it goes
// when somebody decides it can.

import assert from 'node:assert/strict';
import { after, before, beforeEach, describe, it } from 'node:test';

import { installDom, settle } from './support/dom.js';

const ENGINE = { id: 'vllm', name: 'vLLM' };
const STATE = {
  engine: 'vllm', root: '/opt/ai/vllm', linked: true, state: 'idle', error: '',
  free_bytes: 421 * 1024 ** 3, spare_bytes: 7.7 * 1024 ** 3, log: [],
  environments: [
    { name: '.venv-0.26.1rc1.dev949', version: '0.26.1rc1.dev949',
      active: false, size_bytes: 7.7 * 1024 ** 3, path: '/opt/ai/vllm/.venv-0.26.1rc1.dev949' },
    { name: '.venv-0.27.1', version: '0.27.1', active: true,
      size_bytes: 7.9 * 1024 ** 3, path: '/opt/ai/vllm/.venv-0.27.1' },
  ],
};

async function draw(state = STATE, responses = {}) {
  const context = installDom(responses);
  const { versions } = await import(`../../ai_lab/web/js/views/versions.js?${Math.random()}`);
  const node = versions(state, ENGINE, () => {});
  if (node) context.view.append(node);
  await settle();
  return context;
}

function rowFor(view, version) {
  return [...view.querySelectorAll('.row.version')]
    .find((row) => row.textContent.includes(version));
}

function buttonsIn(row) {
  return [...row.querySelectorAll('button')].map((b) => b.textContent.trim());
}

describe('choosing between installed versions', () => {
  let dom;
  before(() => { dom = installDom({}); });
  after(() => dom.restore && dom.restore());
  beforeEach(() => { document.body.innerHTML = ''; });

  it('marks the one in use and leaves it alone', async () => {
    const { view } = await draw();
    const active = rowFor(view, '0.27.1');
    assert.match(active.textContent, /in use/);
    // No way to delete or switch to what is already in use: one would leave
    // the engine with nothing to run and the other does nothing.
    assert.deepEqual(buttonsIn(active), []);
  });

  it('offers the older one as a way back, and as space', async () => {
    const { view } = await draw();
    const old = rowFor(view, '0.26.1rc1.dev949');
    assert.deepEqual(buttonsIn(old), ['Use this one', 'Delete']);
    assert.match(view.textContent, /recoverable/);
  });

  it('asks before deleting, and says it may not come back', async () => {
    // A nightly that has left its index cannot be fetched again. Deleting it
    // is not the same as deleting something you can re-download.
    const { view } = await draw();
    [...rowFor(view, '0.26.1rc1.dev949').querySelectorAll('button')]
      .find((b) => b.textContent.trim() === 'Delete').click();
    await settle();
    const dialog = document.querySelector('dialog.confirm');
    assert.ok(dialog, 'it deleted without asking');
    assert.match(dialog.textContent, /go back to/);
    assert.match(dialog.textContent, /may not be reinstallable/);
    assert.equal(document.activeElement.textContent.trim(), 'Cancel');
  });

  it('deletes nothing when the question is declined', async () => {
    const { view, calls } = await draw();
    [...rowFor(view, '0.26.1rc1.dev949').querySelectorAll('button')]
      .find((b) => b.textContent.trim() === 'Delete').click();
    await settle();
    [...document.querySelector('dialog.confirm').querySelectorAll('button')]
      .find((b) => b.textContent.trim() === 'Cancel').click();
    await settle();
    assert.equal(calls.some((call) => call.method === 'DELETE'), false);
  });

  it('switches without downloading or deleting anything', async () => {
    const { view, calls } = await draw(STATE, {
      'POST /api/installs/vllm/.venv-0.26.1rc1.dev949/activate': STATE,
    });
    [...rowFor(view, '0.26.1rc1.dev949').querySelectorAll('button')]
      .find((b) => b.textContent.trim() === 'Use this one').click();
    await settle();
    assert.ok(calls.some((call) => call.path.includes('/activate')));
    assert.equal(calls.some((call) => call.method === 'DELETE'), false);
    assert.equal(calls.some((call) => call.method === 'POST'
      && /\/api\/installs\/vllm$/.test(call.path)), false, 'it downloaded something');
  });

  it('says nothing at all for an engine that is not installed this way', async () => {
    const { view } = await draw({ environments: [] });
    assert.equal(view.querySelector('.row.version'), null);
  });

  it('holds still while an install is running', async () => {
    // Switching folders underneath an install in progress is a way to end up
    // with neither.
    const { view } = await draw({ ...STATE, state: 'running' });
    const old = rowFor(view, '0.26.1rc1.dev949');
    assert.ok([...old.querySelectorAll('button')].every((b) => b.disabled));
  });

  it('uses source-build controls for a compiled llama.cpp rollback', async () => {
    const source = { ...STATE, kind: 'source', engine: 'llamacpp',
      environments: STATE.environments.map((item) => ({ ...item,
        name: item.active ? 'build-v0.3.0' : 'build-b10448' })) };
    const context = installDom({
      'POST /api/builds/llamacpp/build-b10448/activate': source,
    });
    const { versions } = await import(`../../ai_lab/web/js/views/versions.js?${Math.random()}`);
    context.view.append(versions(source, { id: 'llamacpp', name: 'llama.cpp' }, () => {}));
    [...context.view.querySelectorAll('button')]
      .find((button) => button.textContent === 'Use this one').click();
    await settle();
    assert.ok(context.calls.some((call) =>
      call.path === '/api/builds/llamacpp/build-b10448/activate'));
    assert.equal(context.calls.some((call) => call.path.includes('/api/installs/')), false);
  });
});
