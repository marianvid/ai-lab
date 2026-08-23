// The Settings page, from the user's side.

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { installDom, button, settle } from './support/dom.js';

const REPOSITORY = {
  id: 'gguf', name: 'GGUF models', path: '/models/gguf', format: 'gguf',
  writable: true, exists: true, free_bytes: 3.4 * 1024 ** 4, total_bytes: 4 * 1024 ** 4,
};
const NVFP4 = {
  id: 'nvfp4', name: 'NVFP4', path: '/models/nvfp4', format: 'nvfp4',
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
      models_root: '/models',
      repositories: [REPOSITORY, NVFP4],
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

  it('offers one editable path per real choice', async () => {
    // A path set up on another machine, or a folder that moved, makes every
    // other screen useless. Fixing it should not mean editing a file over ssh
    // — but the model store is one root, not a path per format.
    const { view } = await renderPage();
    const fields = [...view.querySelectorAll('input.path')];
    const values = fields.map((item) => item.value);
    assert.deepEqual(values, ['/models', '/opt/ai/llama.cpp/build/bin/llama-server'],
                     'expected the models root and each engine, and nothing else');
    fields.forEach((item) => assert.equal(item.disabled, false));
  });

  it('keeps every Save asleep until its own field is changed', async () => {
    const { view } = await renderPage();
    const saves = [...view.querySelectorAll('button')]
      .filter((item) => item.textContent.trim() === 'Save');
    assert.ok(saves.length >= 2);
    saves.forEach((item) => assert.equal(item.disabled, true));

    const field = view.querySelector('input.path');
    field.value = '/somewhere/else';
    field.dispatchEvent(new window.Event('input', { bubbles: true }));
    assert.equal(saves[0].disabled, false);
    assert.equal(saves[1].disabled, true, 'a Save woke for another field');
  });

  it('puts the engine programs with the other paths', async () => {
    // They are the same kind of thing — somewhere on disk that has to be right
    // — and looking for them in two places was the only reason it took two.
    const { view } = await renderPage();
    const engine = [...view.querySelectorAll('.card')]
      .find((card) => card.textContent.includes('llama.cpp · '));
    assert.equal(engine.textContent.includes('/opt/ai/llama.cpp/build/bin'), false,
                 'the program is still on the engine card');
    // It lives in a field now, so its value is not in the page text.
    const values = [...view.querySelectorAll('input.path')].map((item) => item.value);
    assert.ok(values.includes('/opt/ai/llama.cpp/build/bin/llama-server'),
              values.join(' | '));
  });

  it('shows each format as a folder in it, and does not let it be set', async () => {
    // Setting them separately let GGUF sit on one disk and NVFP4 on another,
    // which nothing else in this application expects.
    const { view } = await renderPage();
    const derived = [...view.querySelectorAll('.row.derived')];
    assert.equal(derived.length, 2);
    assert.match(derived[0].textContent, /\/models\/gguf/);
    assert.match(derived[1].textContent, /\/models\/nvfp4/);
    derived.forEach((row) => {
      assert.equal(row.querySelector('input'), null, 'a folder is editable');
      assert.equal(row.querySelector('button'), null, 'a folder has a button');
    });
  });

  it('says so when there is no root to work from', async () => {
    const { view } = await renderPage({ '/api/settings': {
      title: 'AI-Lab', engines: [], accelerator: {}, host: {}, memory: { pools: [] },
      models_root: '',
      repositories: [{ ...REPOSITORY, path: '', exists: false, writable: false }] } });
    assert.match(view.textContent, /no models root set/);
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
    assert.match(saved.path, /models-root/, 'saved as a repository path');
  });

  it('saves a root typed by hand', async () => {
    const { view, calls } = await renderPage();
    const field = [...view.querySelectorAll('input.path')][0];
    field.value = '/somewhere/else';
    field.dispatchEvent(new window.Event('input', { bubbles: true }));
    button(view, 'Save').click();
    await settle();
    const saved = calls.find((call) => call.method === 'PATCH');
    assert.equal(JSON.parse(saved.body).path, '/somewhere/else');
    assert.match(saved.path, /models-root/);
  });

  it('saves an engine program to that engine', async () => {
    const { view, calls } = await renderPage();
    const field = [...view.querySelectorAll('input.path')][1];
    field.value = '/usr/local/bin/llama-server';
    field.dispatchEvent(new window.Event('input', { bubbles: true }));
    [...view.querySelectorAll('button')]
      .filter((item) => item.textContent.trim() === 'Save')[1].click();
    await settle();
    const saved = calls.find((call) => call.method === 'PATCH');
    assert.match(saved.path, /\/api\/engines\/llamacpp\/binary/);
    assert.equal(JSON.parse(saved.body).path, '/usr/local/bin/llama-server');
  });

  it('leaves out what is already known: no format pill, no free space', async () => {
    // The name says which format it is, and free space belongs where a
    // download picks its destination, not on four repeated lines.
    const { view } = await renderPage();
    assert.equal(view.textContent.includes('3.4 TB'), false, 'free space is noise here');
  });

  it('puts the root and its buttons on one line', async () => {
    const { view } = await renderPage();
    const row = [...view.querySelectorAll('.row')]
      .find((item) => item.querySelector('input.path'));
    assert.ok(row, 'no root field');
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

  it('follows a build while it runs, with nothing in the way', async () => {
    const base = responses()['/api/settings'];
    const running = { ...base, engines: [{ ...base.engines[0],
      source: { ...base.engines[0].source, state: 'running',
                log: ['[ 42%] Building CXX object'] } }] };
    const { view } = await renderPage({ '/api/settings': running });
    assert.ok(view.querySelector('pre.log'), 'no output while building');
    assert.equal(view.querySelector('details.build-log'), null,
                 'a running build must not be folded away');
    assert.match(view.textContent, /building…/);
  });

  it('folds the log once the build is over, and keeps it', async () => {
    // A page that opens into eight hundred lines of cmake every time is a page
    // with a wall in it. But that output is where "Finished at v0.2.0" is, and
    // where a failure explains itself, so it is kept — closed.
    const base = responses()['/api/settings'];
    const finished = { ...base, engines: [{ ...base.engines[0],
      source: { ...base.engines[0].source, state: 'done',
                log: ['[100%] Built target llama-app', 'Finished at v0.2.0'] } }] };
    const { view } = await renderPage({ '/api/settings': finished });
    const fold = view.querySelector('details.build-log');
    assert.ok(fold, 'the log stayed open with no way to close it');
    assert.equal(fold.open, false);
    assert.match(fold.querySelector('summary').textContent,
                 /Last update log · finished at v0\.2\.0/);
    assert.match(fold.textContent, /Built target llama-app/);
  });

  it('separates what is worked on from what this machine is', async () => {
    // On the left the things that get updated and pointed somewhere else. On
    // the right what the machine is, which is read far more often than it is
    // changed and has exactly one setting in it.
    const { view } = await renderPage();
    const columns = view.querySelector('.columns');
    assert.ok(columns, 'no two-column layout');
    const [left, right] = columns.children;
    assert.match(left.textContent, /Engines/);
    assert.match(left.textContent, /Paths/);
    assert.match(right.textContent, /Machine/);
    // Every path is on the left, with the things that get worked on.
    assert.equal(right.querySelectorAll('input.path').length, 0);
    assert.ok(left.querySelectorAll('input.path').length >= 1);
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

  it('names the machine, and no longer calls it read-only', async () => {
    // The accelerator used to be a report and said so. It is now part of the
    // Machine card, which has a setting in it — how much memory to hold back
    // — so the sentence was both clutter and untrue.
    const { view } = await renderPage();
    assert.match(view.textContent, /RTX PRO 4500/);
    assert.equal(/read-only/i.test(view.textContent), false);
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
    // What it is and whether it works, then what it could become. Where its
    // program lives is with the other paths.
    const { view } = await renderPage();
    const engine = [...view.querySelectorAll('.card')]
      .find((card) => card.textContent.includes('llama.cpp · '));
    assert.ok(engine, 'no card for the engine');
    assert.equal(engine.querySelectorAll(':scope > .row').length, 2);
    assert.match(engine.textContent, /b10398/, 'the version belongs on the first line');
    assert.match(engine.textContent, /Update version/);
  });

  it('says only whether an engine works, on the line with its name', async () => {
    // The pill used to turn orange and announce a new version, which put the
    // news in competition with the state: an engine working perfectly well
    // looked like something was wrong with it.
    const base = responses()['/api/settings'];
    const withUpdate = { ...base, engines: [{ ...base.engines[0],
      source: { ...base.engines[0].source, latest: 'v0.2.0', update_available: true } }] };
    const { view } = await renderPage({ '/api/settings': withUpdate });
    const engine = [...view.querySelectorAll('.card')]
      .find((card) => card.textContent.includes('llama.cpp · '));
    const heading = engine.querySelector(':scope > .row');
    assert.match(heading.textContent, /available/);
    assert.equal(heading.textContent.includes('v0.2.0'), false,
                 'the news is on the heading line, competing with the state');
  });

  it('puts what is waiting on the update line, beside the button', async () => {
    const base = responses()['/api/settings'];
    const withUpdate = { ...base, engines: [{ ...base.engines[0],
      source: { ...base.engines[0].source, latest: 'v0.2.0', update_available: true } }] };
    const { view } = await renderPage({ '/api/settings': withUpdate });
    const line = view.querySelector('.row.update');
    assert.match(line.textContent, /v0\.2\.0 available/);
    assert.match(line.querySelector('button').textContent, /Update…/);
  });

  it('has nothing to press that only asks upstream a question', async () => {
    // "Check for updates" did what the timer already does, and its only real
    // effect was to make the page look like it needed pressing.
    const { view } = await renderPage();
    assert.equal(view.textContent.includes('Check for updates'), false);
  });

  it('leaves the weight formats off the engine name', async () => {
    // "vLLM · awq, fp8, gptq, nvfp4, safetensors" said something a reader of
    // this page cannot act on. The model list says it better, by only offering
    // an entry the formats its engine can read.
    const { view } = await renderPage();
    const engine = [...view.querySelectorAll('.card')]
      .find((card) => card.textContent.includes('llama.cpp'));
    const heading = engine.querySelector(':scope > .row');
    assert.equal(heading.textContent.includes('gguf'), false, heading.textContent);
  });

  it('offers no update controls for an engine that is not installed', async () => {
    const { view } = await renderPage();
    const cards = [...view.querySelectorAll('.card')];
    const vllm = cards.find((card) => card.textContent.includes('vLLM'));
    assert.match(vllm.textContent, /Requires an NVIDIA GPU/);
    assert.equal(vllm.querySelectorAll('button').length, 0,
                 'buttons for something that cannot run here');
  });

  it('reads the same on both engines, whether or not one is waiting', async () => {
    // The button says what it does — it opens something — rather than
    // changing shape depending on whether upstream has been asked yet.
    const { view } = await renderPage();
    const buttons = [...view.querySelectorAll('.row.update button')]
      .map((item) => item.textContent.trim());
    assert.deepEqual(buttons, ['Update…']);
  });

  it('never updates straight from this page', async () => {
    // The whole point: an update is a decision. Nothing on the engine row may
    // start one — it opens what would change, and the real button is at the
    // foot of that.
    const { view, calls } = await renderPage();
    view.querySelector('.row.update button').click();
    await settle();
    assert.equal(calls.some((call) => call.path.includes('/update')), false,
                 'reading what would change must not start it');
  });
});
