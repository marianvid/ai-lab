// The Settings page, from the user's side.

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { installDom, button, settle } from './support/dom.js';

const REPOSITORY = {
  id: 'gguf', name: 'GGUF models', path: '/models/gguf', format: 'gguf',
  writable: true, exists: true, free_bytes: 3.4 * 1024 ** 4, total_bytes: 4 * 1024 ** 4,
};

function responses(overrides = {}) {
  return {
    '/api/settings': {
      title: 'AI-Lab',
      host: { supervisor: 'systemd', accelerator_kind: 'cuda',
              can_configure_accelerator: false },
      accelerator: { available: true, name: 'RTX PRO 4500', kind: 'cuda',
                     memory_kind: 'dedicated', memory_used_mb: 24817,
                     memory_total_mb: 32623, temperature_c: 32,
                     utilization_percent: 0 },
      repositories: [REPOSITORY],
      engines: [
        { id: 'llamacpp', name: 'llama.cpp', available: true, reason: '',
          binary: '/opt/ai/llama.cpp/build/bin/llama-server',
          formats: ['gguf'], params: [],
          source: { engine: 'llamacpp', path: '/opt/ai/llama.cpp', exists: true,
                    installed: 'b10398', commit: '8e7f22b67', latest: '',
                    update_available: false, note: '', state: 'idle',
                    error: '', log: [] } },
        { id: 'vllm', name: 'vLLM', available: false,
          reason: 'Requires an NVIDIA GPU', binary: '',
          formats: ['fp8', 'safetensors'], params: [], source: null },
      ],
    },
    '/api/builds': [],
    '/api/browse': { path: '/models', parent: '/', writable: true,
                     entries: [{ name: 'gguf', path: '/models/gguf', writable: true },
                               { name: 'fp8', path: '/models/fp8', writable: true }] },
    ...overrides,
  };
}

async function renderPage(overrides = {}) {
  const context = installDom(responses(overrides));
  const { render } = await import(`../../ai_lab/web/js/views/settings.js?${Math.random()}`);
  await render(context.view);
  await settle();
  return context;
}

describe('the Settings page', () => {
  it('never puts the word null on the page', async () => {
    const { view } = await renderPage();
    assert.equal(view.textContent.includes('null'), false, view.textContent);
  });

  it('shows the path in a field that can be edited', async () => {
    // A path set up on another machine, or a folder that moved, makes every
    // other screen useless. Fixing it should not mean editing a file over ssh.
    const { view } = await renderPage();
    const field = [...view.querySelectorAll('input')]
      .find((item) => item.value === '/models/gguf');
    assert.ok(field, 'the path is not editable');
    assert.equal(field.disabled, false);
  });

  it('offers a way to pick a folder rather than only typing one', async () => {
    const { view } = await renderPage();
    assert.doesNotThrow(() => button(view, 'Browse…'));
  });

  it('lists the folders on the server when browsing', async () => {
    // The browser cannot open a dialog on the machine the server runs on, so
    // the server has to offer the listing.
    const { view } = await renderPage();
    button(view, 'Browse…').click();
    await settle();
    const dialog = document.querySelector('dialog.confirm');
    assert.ok(dialog, 'no chooser appeared');
    assert.match(dialog.textContent, /gguf/);
    assert.match(dialog.textContent, /fp8/);
    assert.match(dialog.textContent, /up/, 'no way to go up a level');
  });

  it('saves nothing if the chooser is cancelled', async () => {
    const { view, calls } = await renderPage();
    button(view, 'Browse…').click();
    await settle();
    button(document.querySelector('dialog.confirm'), 'Cancel').click();
    await settle();
    assert.equal(calls.filter((call) => call.method === 'PATCH').length, 0);
  });

  it('saves the chosen folder', async () => {
    const { view, calls } = await renderPage();
    button(view, 'Browse…').click();
    await settle();
    button(document.querySelector('dialog.confirm'), 'Use this folder').click();
    await settle();
    const saved = calls.find((call) => call.method === 'PATCH');
    assert.ok(saved, 'nothing was saved');
    assert.equal(JSON.parse(saved.body).path, '/models');
  });

  it('saves a path typed by hand', async () => {
    const { view, calls } = await renderPage();
    const field = [...view.querySelectorAll('input')]
      .find((item) => item.value === '/models/gguf');
    field.value = '/somewhere/else';
    button(view, 'Save').click();
    await settle();
    const saved = calls.find((call) => call.method === 'PATCH');
    assert.equal(JSON.parse(saved.body).path, '/somewhere/else');
  });

  it('leaves out what is already known: no format pill, no free space', async () => {
    // The name says which format it is, and free space belongs where a
    // download picks its destination, not on four repeated lines.
    const { view } = await renderPage();
    assert.equal(view.textContent.includes('3.4 TB'), false, 'free space is noise here');
  });

  it('puts each repository on one line', async () => {
    const { view } = await renderPage();
    const rows = [...view.querySelectorAll('.row.tight')];
    assert.equal(rows.length, 1, 'one row per repository');
    const row = rows[0];
    assert.ok(row.querySelector('input'), 'the path field is on that line');
    assert.equal(row.querySelectorAll('button').length, 2, 'Browse and Save, same line');
  });

  it('leads with the engines, since they decide what the rest can do', async () => {
    const { view } = await renderPage();
    const headings = [...view.querySelectorAll('h3')].map((item) => item.textContent);
    assert.equal(headings[0], 'Engines');
  });

  it('puts every heading above its panel, so they line up', async () => {
    const { view } = await renderPage();
    const headings = [...view.querySelectorAll('h3')];
    assert.equal(headings.length, 3);
    headings.forEach((heading) => {
      assert.ok(heading.parentElement.classList.contains('section'),
                `"${heading.textContent}" is inside a panel instead of above one`);
      assert.equal(heading.parentElement.classList.contains('card'), false);
    });
  });

  it('separates what can be changed from what can only be read', async () => {
    // Engines and repositories have buttons; the accelerator is a report.
    const { view } = await renderPage();
    const columns = view.querySelector('.columns');
    assert.ok(columns, 'no two-column layout');
    const [left, right] = columns.children;
    assert.match(left.textContent, /Engines/);
    assert.match(left.textContent, /Model repositories/);
    assert.match(right.textContent, /Accelerator/);
    assert.equal(right.querySelectorAll('button').length, 0,
                 'the read-only column should have nothing to press');
  });

  it('does not repeat the page name, which the tab already shows', async () => {
    const { view } = await renderPage();
    assert.equal(view.querySelector('h2'), null);
  });

  it('says plainly when a folder is missing rather than showing a size', async () => {
    const { view } = await renderPage({
      '/api/settings': { ...responses()['/api/settings'],
        repositories: [{ ...REPOSITORY, exists: false, writable: false, free_bytes: 0 }] },
    });
    assert.match(view.textContent, /missing/);
  });

  it('says a folder is read-only rather than pretending it can be used', async () => {
    const { view } = await renderPage({
      '/api/settings': { ...responses()['/api/settings'],
        repositories: [{ ...REPOSITORY, writable: false }] },
    });
    assert.match(view.textContent, /read-only/);
  });

  it('reports the accelerator as something it will not change', async () => {
    const { view } = await renderPage();
    assert.match(view.textContent, /RTX PRO 4500/);
    assert.match(view.textContent, /Read-only/);
  });

  it('describes an engine once, not in two places', async () => {
    // It used to appear under "Engines" and again under "Engine sources",
    // spelled differently, with two paths and two badges to join up.
    const { view } = await renderPage();
    assert.equal(view.textContent.includes('Engine sources'), false);
    assert.equal(view.textContent.includes('llamacpp'), false,
                 'the internal id leaked into the page');
    assert.equal((view.textContent.match(/llama\.cpp/g) || []).length >= 1, true);
  });

  it('fits an engine into two lines', async () => {
    // What it is and whether it works, then where it is and what to press.
    const { view } = await renderPage();
    const engine = [...view.querySelectorAll('.card')]
      .find((card) => card.textContent.includes('llama.cpp'));
    assert.ok(engine, 'no card for the engine');
    assert.equal(engine.querySelectorAll(':scope > .row').length, 2);
    assert.match(engine.textContent, /b10398/, 'the version belongs on the first line');
    assert.match(engine.textContent, /\/opt\/ai\/llama\.cpp\/build\/bin\/llama-server/);
    assert.ok([...engine.querySelectorAll('button')]
      .some((item) => item.textContent.includes('Check for updates')));
  });

  it('says what an engine reads, which is what decides if it is any use', async () => {
    const { view } = await renderPage();
    assert.match(view.textContent, /gguf/);
    assert.match(view.textContent, /fp8, safetensors/);
  });

  it('offers no update controls for an engine that is not installed', async () => {
    const { view } = await renderPage();
    const cards = [...view.querySelectorAll('.card')];
    const vllm = cards.find((card) => card.textContent.includes('vLLM'));
    assert.match(vllm.textContent, /Requires an NVIDIA GPU/);
    assert.equal(vllm.querySelectorAll('button').length, 0,
                 'buttons for something that cannot run here');
  });

  it('names the version to be read about when one is waiting', async () => {
    const base = responses()['/api/settings'];
    const withUpdate = {
      ...base,
      engines: [{ ...base.engines[0],
        source: { ...base.engines[0].source, latest: 'v0.2.0', update_available: true } }],
    };
    const { view } = await renderPage({ '/api/settings': withUpdate });
    assert.ok([...view.querySelectorAll('button')]
      .some((item) => item.textContent.includes('Review v0.2.0')));
  });

  it('never updates straight from this page', async () => {
    // The whole point: an update is a decision. Nothing on the engine row may
    // start one — it opens what would change, and the real button is at the
    // foot of that.
    const base = responses()['/api/settings'];
    const withUpdate = {
      ...base,
      engines: [{ ...base.engines[0],
        source: { ...base.engines[0].source, latest: 'v0.2.0', update_available: true } }],
    };
    const { view, calls } = await renderPage({ '/api/settings': withUpdate });
    const review = [...view.querySelectorAll('button')]
      .find((item) => item.textContent.includes('Review'));
    review.click();
    await settle();
    assert.equal(calls.some((call) => call.path.includes('/update')), false,
                 'reading what would change must not start it');
  });
});
