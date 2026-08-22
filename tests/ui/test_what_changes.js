// Reading an update before taking it.
//
// Every test here corresponds to a way this screen could quietly mislead:
// offering a button it cannot honour, hiding changes rather than folding them,
// or showing a version bump without saying that eight packages go backwards.

import assert from 'node:assert/strict';
import { after, before, beforeEach, describe, it } from 'node:test';

import { installDom, settle } from './support/dom.js';

const ENGINE = { id: 'vllm', name: 'vLLM', available: true, binary: '/bin/vllm' };

const FOUND = {
  engine: 'vllm', engine_name: 'vLLM',
  installed: '0.26.1rc1.dev949+geac636a7f', latest: '0.27.1',
  yours: [], others: [], by_area: {}, other_areas: {},
  notes: '# v0.27.1\n\nThis is a patch release on top of v0.27.0.',
  packages: [
    { name: 'flashinfer-python', was: '0.6.17', becomes: '0.6.16.post3',
      backwards: true, heavy: true },
    { name: 'nvidia-cutlass-dsl', was: '4.6.2', becomes: '4.6.0',
      backwards: true, heavy: false },
    { name: 'transformers', was: '5.15.0', becomes: '5.15.1',
      backwards: false, heavy: true },
  ],
  unreadable: '',
};

const LLAMA = {
  engine: 'llamacpp', engine_name: 'llama.cpp',
  installed: 'b10448', latest: 'v0.2.0',
  yours: [{ area: 'CUDA', title: 'cuda : fix mma', reference: 'aaa' },
          { area: 'server', title: 'server: add a thing', reference: 'bbb' }],
  others: [{ area: 'SYCL', title: 'sycl : something', reference: 'ccc' },
           { area: 'build', title: 'ci : a job', reference: 'ddd' }],
  by_area: { CUDA: 1, server: 1 },
  other_areas: { SYCL: 1, build: 1 },
  notes: '',
  packages: [],
  unreadable: 'b10448 and v0.2.0 are on different lines, so what follows '
            + 'describes v0.2.0 itself.',
};

async function review(found, { canUpdate = true } = {}) {
  const context = installDom({ [`/api/builds/${found.engine}/changes`]: found });
  const { reviewUpdate } = await import(
    `../../ai_lab/web/js/views/whatchanges.js?${Math.random()}`);
  const started = [];
  reviewUpdate({ id: found.engine, name: found.engine_name },
               canUpdate ? async () => started.push(1) : null);
  await settle();
  return { dialog: document.querySelector('dialog.review'), started, context };
}

function buttons(dialog) {
  return [...dialog.querySelectorAll('button')].map((b) => b.textContent.trim());
}

describe('reading an update before taking it', () => {
  let dom;
  before(() => { dom = installDom({}); });
  after(() => dom.restore && dom.restore());
  beforeEach(() => { document.body.innerHTML = ''; });

  it('says what is installed and what it would become', async () => {
    const { dialog } = await review(FOUND);
    assert.match(dialog.textContent, /0\.26\.1rc1\.dev949/);
    assert.match(dialog.textContent, /0\.27\.1/);
  });

  it('calls out the packages that would go backwards', async () => {
    // The single most important thing on this screen, and the one no release
    // note anywhere will tell you.
    const { dialog } = await review(FOUND);
    assert.match(dialog.textContent, /2 going backwards/);
    const back = [...dialog.querySelectorAll('tr.backwards')]
      .map((row) => row.textContent);
    assert.equal(back.length, 2);
    assert.ok(back.some((row) => row.includes('flashinfer-python')));
    assert.ok(back.every((row) => !row.includes('transformers')),
              'a package moving forward marked as going backwards');
  });

  it('puts the real Update button at the foot, after all of it', async () => {
    const { dialog } = await review(FOUND);
    const names = buttons(dialog);
    assert.ok(names.some((name) => name.includes('Update to 0.27.1')));
    // Last, so everything above has been scrolled past to reach it.
    assert.match(names[names.length - 1], /Update to/);
  });

  it('starts nothing until that button is pressed', async () => {
    const { dialog, started } = await review(FOUND);
    assert.deepEqual(started, [], 'opening the screen started an update');
    [...dialog.querySelectorAll('button')]
      .find((b) => b.textContent.includes('Update to')).click();
    await settle();
    assert.deepEqual(started, [1]);
  });

  it('offers no button for an engine it cannot update itself', async () => {
    // vLLM arrives as packages. Everything above is still worth reading; a
    // button that cannot honour itself is worse than none.
    const { dialog } = await review(FOUND, { canUpdate: false });
    assert.equal(buttons(dialog).some((name) => name.includes('Update to')), false);
    assert.match(dialog.textContent, /not done from here yet/);
    assert.match(dialog.textContent, /going backwards/, 'the reading is still there');
  });

  it('offers no button when there is nothing to move to', async () => {
    const { dialog } = await review({ ...FOUND, latest: FOUND.installed });
    assert.equal(buttons(dialog).some((name) => name.includes('Update to')), false);
    assert.match(dialog.textContent, /Up to date/);
  });

  it('shows what matters here and folds the rest rather than dropping it', async () => {
    const { dialog } = await review(LLAMA);
    assert.match(dialog.textContent, /2 changes here/);
    assert.match(dialog.textContent, /cuda : fix mma/);

    const fold = [...dialog.querySelectorAll('details')]
      .find((node) => /does not use/.test(node.querySelector('summary').textContent));
    assert.ok(fold, 'the rest was dropped instead of folded');
    assert.match(fold.querySelector('summary').textContent, /2 for parts/);
    assert.match(fold.textContent, /sycl : something/,
                 'a change this machine does not use must still be openable');
  });

  it('repeats a warning about what could not be read', async () => {
    const { dialog } = await review(LLAMA);
    assert.match(dialog.textContent, /different lines/);
  });

  it('folds very long release notes instead of opening into them', async () => {
    const long = { ...FOUND, notes: '# v0.27.0\n\n' + 'x'.repeat(30000) };
    const { dialog } = await review(long);
    const fold = [...dialog.querySelectorAll('details')]
      .find((node) => /Release notes/.test(node.querySelector('summary').textContent));
    assert.ok(fold, '30,000 characters were shown unfolded');
    assert.equal(fold.open, false);
  });

  it('shows short release notes without making them a click away', async () => {
    const { dialog } = await review(FOUND);
    assert.match(dialog.textContent, /patch release on top of v0\.27\.0/);
    const folded = [...dialog.querySelectorAll('details summary')]
      .some((node) => /Release notes/.test(node.textContent));
    assert.equal(folded, false);
  });
});
