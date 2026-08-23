// How much memory this machine has for models, on the Settings page.
//
// A capacity, not a reading: what is free right now belongs on the Gateway
// page with the rest of what is happening. Mixing the two makes a number that
// answers neither question, and these tests hold the line.

import assert from 'node:assert/strict';
import { after, before, beforeEach, describe, it } from 'node:test';

import { installDom, settle } from './support/dom.js';

// A card holding a 29 GB model, so that "capacity" and "free right now" are
// far apart and a test cannot pass by confusing them.
const LINUX = {
  unified: false, reserve_mb: 8192,
  capacity_mb: 32623 + 40960, available_mb: 3677 + 35803,
  held_by_models_mb: 28946,
  pools: [
    { name: 'card', kind: 'dedicated', total_mb: 32623, used_mb: 28946,
      used_by_models_mb: 28946, reserve_mb: 0, free_mb: 3677,
      capacity_mb: 32623, available_mb: 3677 },
    { name: 'machine', kind: 'dedicated', total_mb: 49152, used_mb: 5157,
      used_by_models_mb: 0, reserve_mb: 8192, free_mb: 43995,
      capacity_mb: 40960, available_mb: 35803 },
  ],
};

const MAC = {
  unified: true, reserve_mb: 16384,
  capacity_mb: 114688, available_mb: 80088, held_by_models_mb: 21000,
  pools: [
    { name: 'machine', kind: 'unified', total_mb: 131072, used_mb: 34600,
      used_by_models_mb: 21000, reserve_mb: 16384, free_mb: 96472,
      capacity_mb: 114688, available_mb: 80088 },
  ],
};

async function draw(state, responses = {}) {
  const context = installDom(responses);
  const { memory } = await import(`../../ai_lab/web/js/views/memory.js?${Math.random()}`);
  const node = memory(state, () => {});
  if (node) context.view.append(node);
  await settle();
  return context;
}

const field = (view) => view.querySelector('input[type="number"]');
const saveButton = (view) => [...view.querySelectorAll('button')]
  .find((b) => b.textContent.trim() === 'Save');
// Label and value are separate elements, so they are read separately: joining
// them gives "VRAM32623 MB", which no assertion should have to know about.
const rows = (view) => [...view.querySelectorAll('.row')].map((r) => ({
  label: r.children[0].textContent.trim(),
  value: r.children[1].textContent.trim(),
}));

describe('available memory', () => {
  let dom;
  before(() => { dom = installDom({}); });
  after(() => dom.restore && dom.restore());
  beforeEach(() => { document.body.innerHTML = ''; });

  it('is named for what it answers', async () => {
    const { view } = await draw(LINUX);
    assert.match(view.querySelector('h3').textContent, /Available memory/);
  });

  it('lists VRAM and RAM on a machine with a card', async () => {
    const { view } = await draw(LINUX);
    const listed = rows(view);
    assert.deepEqual(listed[0], { label: 'VRAM', value: '32623 MB' });
    assert.deepEqual(listed[1], { label: 'RAM', value: '40960 MB' });
  });

  it('lists the card whole, not what is left of it', async () => {
    // 28.9 GB of this card is held by a running model. That is a fact about
    // now and belongs on the Gateway page; this line says how big the card is.
    const { view } = await draw(LINUX);
    assert.match(rows(view)[0].value, /32623/);
    assert.equal(view.textContent.includes('3677'), false,
                 'this card is showing what is free rather than what there is');
  });

  it('takes the reserve out of RAM but not out of VRAM', async () => {
    // Nothing but models uses the card. The machine is shared.
    const { view } = await draw(LINUX);
    assert.equal(rows(view)[0].value, '32623 MB');    // the whole card
    assert.equal(rows(view)[1].value, '40960 MB');    // 49152 less 8192
  });

  it('says unified memory once, and calls it that', async () => {
    const { view } = await draw(MAC);
    const listed = rows(view);
    assert.deepEqual(listed[0], { label: 'Unified memory', value: '114688 MB' });
    assert.equal(view.textContent.includes('VRAM'), false,
                 'one pool shown twice doubles the machine');
    assert.equal(listed.length, 2, 'expected one memory line and the reserve');
  });

  it('names the reserve for what it protects, not for what it withholds', async () => {
    // "Reserved memory" alone reads as reserved *for models*, which is the
    // opposite of what it does.
    const { view } = await draw(MAC);
    assert.match(view.textContent, /Reserved for the system/);
  });

  it('shows the reserve that is actually in force', async () => {
    const { view } = await draw(LINUX);
    assert.equal(field(view).value, '8192');
  });

  it('keeps Save asleep until the reserve is actually changed', async () => {
    const { view } = await draw(MAC);
    assert.equal(saveButton(view).disabled, true);
    const input = field(view);
    input.value = '24576';
    input.dispatchEvent(new window.Event('input', { bubbles: true }));
    assert.equal(saveButton(view).disabled, false);
  });

  it('sends the reserve, not the amount left over', async () => {
    // The field is the reserve. Sending what is available instead would go
    // stale the moment this machine is given more memory.
    const { view, calls } = await draw(MAC, { 'PATCH /api/memory': MAC });
    const input = field(view);
    input.value = '24576';
    input.dispatchEvent(new window.Event('input', { bubbles: true }));
    saveButton(view).click();
    await settle();
    const sent = calls.find((call) => call.method === 'PATCH');
    assert.ok(sent, 'nothing was saved');
    assert.deepEqual(JSON.parse(sent.body), { reserve_mb: 24576 });
  });

  it('draws nothing at all when the machine could not be read', async () => {
    const { view } = await draw({ unified: false, pools: [], capacity_mb: 0 });
    assert.equal(view.querySelector('.card.memory'), null);
  });
});
