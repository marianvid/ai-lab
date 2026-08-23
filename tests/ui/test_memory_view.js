// How much of this machine models may use, on the Settings page.

import assert from 'node:assert/strict';
import { after, before, beforeEach, describe, it } from 'node:test';

import { installDom, settle } from './support/dom.js';

const CARD_MACHINE = {
  unified: false, for_models_mb: 3485 + 28960,
  held_by_models_mb: 29138,
  pools: [
    { name: 'card', kind: 'dedicated', total_mb: 32623, used_mb: 29138,
      used_by_models_mb: 29138, reserve_mb: 0, free_mb: 3485, available_mb: 3485 },
    { name: 'machine', kind: 'dedicated', total_mb: 49152, used_mb: 12000,
      used_by_models_mb: 0, reserve_mb: 8192, free_mb: 37152, available_mb: 28960 },
  ],
};

const MAC = {
  unified: true, for_models_mb: 80088, held_by_models_mb: 21000,
  pools: [
    { name: 'machine', kind: 'unified', total_mb: 131072, used_mb: 34600,
      used_by_models_mb: 21000, reserve_mb: 16384, free_mb: 96472,
      available_mb: 80088 },
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

function field(view) {
  return view.querySelector('input[type="number"]');
}

function saveButton(view) {
  return [...view.querySelectorAll('button')]
    .find((b) => b.textContent.trim() === 'Save');
}

describe('memory for models', () => {
  let dom;
  before(() => { dom = installDom({}); });
  after(() => dom.restore && dom.restore());
  beforeEach(() => { document.body.innerHTML = ''; });

  it('shows both pools on a machine with a dedicated card', async () => {
    const { view } = await draw(CARD_MACHINE);
    const rows = [...view.querySelectorAll('.row.pool')].map((r) => r.textContent);
    assert.equal(rows.length, 2);
    assert.ok(rows[0].includes('Card'));
    assert.ok(rows[1].includes('This machine'));
  });

  it('shows one pool on unified memory, never two views of the same thing', async () => {
    const { view } = await draw(MAC);
    assert.equal(view.querySelectorAll('.row.pool').length, 1);
    assert.equal(view.textContent.includes('Card'), false,
                 'a card row on unified memory counts the same memory twice');
  });

  it('says the card is used whole rather than offering a reserve for it', async () => {
    const { view } = await draw(CARD_MACHINE);
    const card = [...view.querySelectorAll('.row.pool')]
      .find((r) => r.textContent.includes('Card'));
    assert.match(card.querySelector('span:last-child').title,
                 /All of it is available to models/);
  });

  it('shows the reserve that is actually in force', async () => {
    const { view } = await draw(CARD_MACHINE);
    assert.equal(field(view).value, '8192');
  });

  it('ends with the number that is the answer', async () => {
    const { view } = await draw(MAC);
    const total = view.querySelector('.row.total');
    assert.match(total.textContent, /Available for models/);
    assert.match(total.textContent, /80088 MB/);
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
    // stale the moment the machine is given more memory.
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
    const { view } = await draw({ unified: false, pools: [], for_models_mb: 0 });
    assert.equal(view.querySelector('.card.memory'), null);
  });
});
