// The Library page, from the user's side.

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { installDom, button, settle } from './support/dom.js';

const MODEL = {
  id: 'gguf/qwen/qwen', name: 'qwen', format: 'gguf',
  size_bytes: 21 * 1024 ** 3, file_count: 1, complete: true, missing: [],
};
const REPOSITORY = {
  id: 'gguf', name: 'GGUF models', path: '/models/gguf', format: 'gguf',
  writable: true, exists: true, free_bytes: 3.4 * 1024 ** 4, total_bytes: 4 * 1024 ** 4,
};
const SEARCH_RESULT = { repo: 'unsloth/Qwen3-GGUF', downloads: 91123, likes: 4, updated: '' };
const VARIANT = {
  repo: 'unsloth/Qwen3-GGUF', name: 'Qwen3-Q4_K_M', format: 'gguf',
  size_bytes: 400 * 1024 ** 2, complete: true, missing: [],
  files: [{ path: 'Qwen3-Q4_K_M.gguf', size_bytes: 400 * 1024 ** 2 }],
};
const TRANSFER = {
  id: 'unsloth_Qwen3-GGUF_Qwen3-Q4_K_M', repo: VARIANT.repo, name: VARIANT.name,
  state: 'running', percent: 25, received_bytes: 100 * 1024 ** 2,
  total_bytes: 400 * 1024 ** 2, files_done: 0, files_total: 1, error: '',
};

function responses(overrides = {}) {
  return {
    '/api/models': [MODEL],
    '/api/settings': { title: 'AI-Lab', engines: [], repositories: [REPOSITORY],
                       accelerator: {}, host: {} },
    '/api/downloads': [],
    'POST /api/downloads': TRANSFER,
    '/api/hf/search': [SEARCH_RESULT],
    '/api/hf/sets': [VARIANT],
    ...overrides,
  };
}

async function renderPage(overrides = {}) {
  const context = installDom(responses(overrides));
  const { render } = await import(`../../ai_lab/web/js/views/library.js?${Math.random()}`);
  await render(context.view);
  await settle();
  return { ...context, render };
}

async function searchFor(context, query) {
  const input = context.view.querySelector('input[type="text"], input:not([type])');
  input.value = query;
  button(context.view, 'Search').click();
  await settle();
}

describe('the Library page', () => {
  it('never puts the word null on the page', async () => {
    const { view } = await renderPage();
    assert.equal(view.textContent.includes('null'), false, view.textContent);
  });

  it('keeps what is being typed when the page refreshes itself', async () => {
    // The page redraws every few seconds. Rebuilding the field wiped the query
    // mid-word, which is how this was noticed.
    const { view, render } = await renderPage();
    const input = view.querySelector('input:not([type="checkbox"])');
    input.value = 'qwen3 gguf';
    await render(view);
    await settle();
    const after = view.querySelector('input:not([type="checkbox"])');
    assert.equal(after.value, 'qwen3 gguf', 'the refresh ate the query');
  });

  it('filters to runnable formats by default, and says what ticking does', async () => {
    const { view } = await renderPage();
    const box = view.querySelector('input[type="checkbox"]');
    assert.equal(box.checked, true, 'the filter should start on');
    assert.match(view.textContent, /Show only supported formats/);
  });

  it('shows the variants of a repository right beneath it', async () => {
    // They used to render below the whole result list, so clicking looked like
    // it had done nothing at all.
    const context = await renderPage();
    await searchFor(context, 'qwen');
    assert.match(context.view.textContent, /unsloth\/Qwen3-GGUF/);

    button(context.view, 'Show models').click();
    await settle();

    const variants = context.view.querySelector('.variants');
    assert.ok(variants, 'nothing expanded');
    assert.match(variants.textContent, /Qwen3-Q4_K_M/);

    // And beneath its own repository, not at the bottom of the page.
    const repositoryRow = variants.previousElementSibling;
    assert.match(repositoryRow.textContent, /unsloth\/Qwen3-GGUF/);
  });

  it('says which format each variant is', async () => {
    const context = await renderPage();
    await searchFor(context, 'qwen');
    button(context.view, 'Show models').click();
    await settle();
    assert.match(context.view.querySelector('.variants').textContent, /gguf/);
  });

  it('can be collapsed again', async () => {
    const context = await renderPage();
    await searchFor(context, 'qwen');
    button(context.view, 'Show models').click();
    await settle();
    button(context.view, 'Hide').click();
    await settle();
    assert.equal(context.view.querySelector('.variants'), null);
  });

  it('downloads without asking where to put it', async () => {
    const context = await renderPage();
    await searchFor(context, 'qwen');
    button(context.view, 'Show models').click();
    await settle();
    button(context.view, 'Download').click();
    await settle();

    const posted = context.calls.find((call) => call.method === 'POST'
      && call.path.includes('/api/downloads'));
    assert.ok(posted, 'nothing was requested');
    const body = JSON.parse(posted.body);
    assert.equal(body.repo, 'unsloth/Qwen3-GGUF');
    assert.equal(body.name, 'Qwen3-Q4_K_M');
    assert.equal('repository_id' in body, false, 'it should work the destination out');
  });

  it('shows a running download beside the variant it belongs to', async () => {
    const context = await renderPage({ '/api/downloads': [TRANSFER] });
    await searchFor(context, 'qwen');
    button(context.view, 'Show models').click();
    await settle();

    const variant = context.view.querySelector('.variant');
    assert.match(variant.textContent, /25\.0%/);
    assert.match(variant.textContent, /100 MB of 400 MB/);
    assert.ok(variant.querySelector('.bar'), 'the variant has no progress bar');
  });

  it('shows a cancelled transfer and offers to resume it', async () => {
    const cancelled = { ...TRANSFER, state: 'cancelled' };
    const context = await renderPage({ '/api/downloads': [cancelled] });
    await searchFor(context, 'qwen');
    button(context.view, 'Show models').click();
    await settle();

    const variant = context.view.querySelector('.variant');
    assert.match(variant.textContent, /cancelled/);
    assert.ok(button(variant, 'Resume'));
  });

  it('asks before deleting from disk, with Cancel holding the focus', async () => {
    const { view, calls } = await renderPage();
    button(view, 'Delete').click();
    await settle();
    assert.equal(calls.filter((call) => call.method === 'DELETE').length, 0);
    assert.equal(document.activeElement.textContent.trim(), 'Cancel');
    assert.match(document.querySelector('dialog.confirm').textContent, /cannot be undone/);
  });

  it('deletes only once the destructive button is pressed', async () => {
    const { view, calls } = await renderPage();
    button(view, 'Delete').click();
    await settle();
    button(document.querySelector('dialog.confirm'), 'Delete').click();
    await settle();
    const deleted = calls.find((call) => call.method === 'DELETE');
    assert.ok(deleted, 'the model was never deleted');
    assert.match(decodeURIComponent(deleted.path), /gguf\/qwen\/qwen/);
  });

  it('lists what is on disk with its size and state', async () => {
    const { view } = await renderPage();
    assert.match(view.textContent, /qwen/);
    assert.match(view.textContent, /21 GB/);
    assert.match(view.textContent, /complete/);
  });

  it('does not show configured folders that contain no models', async () => {
    const { view } = await renderPage({ '/api/models': [] });
    assert.equal(view.textContent.includes('GGUF models'), false, view.textContent);
  });
});
