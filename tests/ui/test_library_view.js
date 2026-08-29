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
const ROOTS = [
  { id: 'core', name: 'Core', path: '/models', enabled: true, writable: true,
    exists: true, free_bytes: 600 * 1024 ** 3 },
  { id: 'benchmark', name: 'Benchmark', path: '/test_models', enabled: true,
    writable: true, exists: true, free_bytes: 3.4 * 1024 ** 4 },
];
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
                       model_roots: ROOTS,
                       accelerator: {}, host: {} },
    '/api/downloads': [],
    'POST /api/downloads': TRANSFER,
    '/api/hf/search': { results: [SEARCH_RESULT], hidden: 0 },
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

  it('does not ask whether to filter, because there is no other answer', async () => {
    // A machine with no engine that reads safetensors cannot be helped by a
    // list of them, and the download would be thirty gigabytes of nothing. The
    // switch was one more decision in front of a search box.
    const { view } = await renderPage();
    assert.equal(view.querySelector('input[type="checkbox"]'), null);
    assert.equal(view.textContent.includes('supported formats'), false);
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

  it('asks where to put each download and sends that exact tier', async () => {
    const context = await renderPage();
    await searchFor(context, 'qwen');
    button(context.view, 'Show models').click();
    await settle();
    button(context.view, 'Download…').click();
    await settle();

    const dialog = document.querySelector('dialog.download-destination');
    assert.ok(dialog, 'destination dialog did not open');
    assert.match(dialog.textContent, /Temporary \/ benchmark/);
    assert.match(dialog.textContent, /Production/);
    dialog.querySelector('input[value="core"]').click();
    button(dialog, 'Start download').click();
    await settle();

    const posted = context.calls.find((call) => call.method === 'POST'
      && call.path.includes('/api/downloads'));
    assert.ok(posted, 'nothing was requested');
    const body = JSON.parse(posted.body);
    assert.equal(body.repo, 'unsloth/Qwen3-GGUF');
    assert.equal(body.name, 'Qwen3-Q4_K_M');
    assert.equal(body.storage_tier, 'core');
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
    assert.ok(button(variant, 'Resume…'));
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

describe('getting out of a search', () => {
  // Without a way out, the results sat there for the rest of the session,
  // pushing the models actually on disk — which is what this page is for —
  // off the bottom of the screen.

  async function searchThen(responses = {}) {
    const context = await renderPage(responses);
    const input = context.view.querySelector('input.grow');
    input.value = 'qwen';
    button(context.view, 'Search').click();
    await settle();
    return context;
  }

  it('offers nothing to clear before anything has been searched', async () => {
    const { view } = await renderPage();
    assert.equal([...view.querySelectorAll('button')]
      .some((item) => item.textContent.trim() === 'Clear'), false,
      'a button that does nothing is one people stop trusting');
  });

  it('offers a way out once there are results', async () => {
    const { view } = await searchThen();
    assert.doesNotThrow(() => button(view, 'Clear'));
    assert.match(view.textContent, /unsloth\/Qwen3-GGUF/);
  });

  it('puts the search away and shows what is on disk', async () => {
    const { view } = await searchThen();
    button(view, 'Clear').click();
    await settle();
    assert.equal(view.textContent.includes('unsloth/Qwen3-GGUF'), false,
                 'the results stayed');
    assert.match(view.textContent, /qwen/, 'the model on disk went with them');
    assert.equal(view.querySelector('input.grow').value, '',
                 'the box still held the old query');
  });

  it('offers a way out of a search that found nothing, too', async () => {
    // That leaves a sentence rather than a list, and it needs clearing just
    // the same.
    const { view } = await searchThen({ '/api/hf/search': { results: [], hidden: 4 } });
    assert.match(view.textContent, /none in a format this machine can run/);
    button(view, 'Clear').click();
    await settle();
    assert.equal(view.textContent.includes('none in a format'), false);
  });
});

describe('what a search that found nothing means', () => {
  // Two different answers used to look identical. Either Hugging Face has
  // nothing by that name, or it has plenty and none of it is in a format this
  // machine can run — which on the Mac is everything but GGUF. The switch that
  // showed the rest was the only way to tell, and it is gone.

  // Said where the results would be, right under the box that was typed into.
  // There used to be a line at the foot of the page for this; it was a long
  // way from where anybody was looking, and the next action wiped it.
  async function searchFor(answer) {
    const context = await renderPage({ '/api/hf/search': answer });
    const input = context.view.querySelector('input.grow');
    input.value = 'towerinstruct';
    button(context.view, 'Search').click();
    await settle();
    const said = context.view.querySelector('.outcome');
    return said ? said.textContent : '';
  }

  it('says so plainly when there really is nothing', async () => {
    assert.match(await searchFor({ results: [], hidden: 0 }), /Nothing found/);
  });

  it('says there were results, in formats this machine cannot run', async () => {
    const text = await searchFor({ results: [], hidden: 4 });
    assert.match(text, /4 found/);
    assert.match(text, /none in a format this machine can run/);
  });

  it('says nothing at all when there are results to look at', async () => {
    // The list is the answer. Counting it in a sentence beside it was the
    // page telling you what you could already see — and it counted what it
    // was showing, never what it hid, which nobody could tell from the
    // sentence anyway.
    const context = await renderPage({
      '/api/hf/search': { results: [SEARCH_RESULT], hidden: 9 } });
    const input = context.view.querySelector('input.grow');
    input.value = 'towerinstruct';
    button(context.view, 'Search').click();
    await settle();
    assert.equal(context.view.querySelector('.outcome'), null);
    assert.match(context.view.textContent, /unsloth\/Qwen3-GGUF/);
  });
});
