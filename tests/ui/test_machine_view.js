// What this machine is, on the Settings page.
//
// One card: the chip, what starts the engines, and how much memory models may
// use. Nothing in it moves on its own — how much is used, how warm the card
// is, how many requests are running are facts about right now, and right now
// is the Gateway page. Mixing the two makes a figure that answers neither
// question, and these tests hold that line.

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

const CHIP = { name: 'NVIDIA RTX PRO 4500 Blackwell', kind: 'cuda',
               memory_kind: 'dedicated', temperature_c: 32,
               utilization_percent: 0, memory_used_mb: 28946,
               memory_total_mb: 32623 };
const APPLE = { name: 'Apple M3 Max', kind: 'metal', memory_kind: 'unified',
                temperature_c: null, utilization_percent: null,
                memory_used_mb: 21000, memory_total_mb: 131072 };

async function draw(memory, { chip = CHIP, host = { supervisor: 'systemd' },
                              responses = {} } = {}) {
  const context = installDom(responses);
  const { machine } = await import(`../../ai_lab/web/js/views/machine.js?${Math.random()}`);
  const node = machine({ accelerator: chip, memory, host }, () => {});
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

describe('the machine', () => {
  let dom;
  before(() => { dom = installDom({}); });
  after(() => dom.restore && dom.restore());
  beforeEach(() => { document.body.innerHTML = ''; });

  it('is one card about one subject', async () => {
    const { view } = await draw(LINUX);
    assert.equal(view.querySelectorAll('.section').length, 1);
    assert.equal(view.querySelector('h3').textContent, 'Machine');
    // The heading goes above the panel, like every other section here.
    assert.equal(view.querySelector('h3').parentElement.className, 'section machine');
  });

  it('names the chip', async () => {
    const { view } = await draw(LINUX);
    assert.deepEqual(rows(view)[0],
                     { label: 'Accelerator', value: 'NVIDIA RTX PRO 4500 Blackwell' });
  });

  it('holds nothing that moves on its own', async () => {
    // Used, temperature and utilisation all belong to right now, and right now
    // is the Gateway page. A figure read here should still be true an hour
    // later.
    const { view } = await draw(LINUX);
    assert.equal(view.textContent.includes('28946'), false, 'shows what is used');
    assert.equal(view.textContent.includes('°C'), false, 'shows the temperature');
    assert.match(view.textContent, /32623/, 'but does show how big the card is');
  });

  it('says nothing about being read-only, now that it is not', async () => {
    const { view } = await draw(LINUX);
    assert.equal(/read-only/i.test(view.textContent), false);
  });

  it('lists VRAM and RAM on a machine with a card', async () => {
    const { view } = await draw(LINUX);
    const listed = rows(view);
    assert.deepEqual(listed[1], { label: 'VRAM', value: '32623 MB' });
    assert.deepEqual(listed[2], { label: 'RAM', value: '49152 MB' });
  });

  it('shows what this machine has, not what is left after the reserve', async () => {
    // The reserve is its own line directly below. Subtracting it here as well
    // would print a number matching neither the hardware nor the line beside
    // it, and leave nowhere to read the real size of the machine.
    const { view } = await draw(LINUX);
    assert.equal(rows(view)[2].value, '49152 MB');
    assert.equal(view.textContent.includes('40960'), false,
                 'RAM is showing what is left rather than what there is');
  });

  it('lists the card whole, not what is left of it', async () => {
    // 28.9 GB of this card is held by a running model. That is a fact about
    // now and belongs on the Gateway page; this line says how big the card is.
    const { view } = await draw(LINUX);
    assert.match(rows(view)[1].value, /32623/);
    assert.equal(view.textContent.includes('3677'), false,
                 'this card is showing what is free rather than what there is');
  });

  it('shows the reserve as its own line, to be taken off in the head', async () => {
    // The card carries none — nothing but models uses it — so only one of
    // these two figures has anything taken off it, and saying which in prose
    // under them would be an explanation nobody rereads.
    const { view } = await draw(LINUX);
    const listed = rows(view);
    assert.equal(listed[listed.length - 1].label, 'Reserved for the system');
  });

  it('says unified memory once, and calls it that', async () => {
    const { view } = await draw(MAC, { chip: APPLE });
    const listed = rows(view);
    assert.deepEqual(listed[1], { label: 'Unified memory', value: '131072 MB' });
    assert.equal(view.textContent.includes('VRAM'), false,
                 'one pool shown twice doubles the machine');
    assert.equal(listed.length, 3,
                 'expected the chip, one memory line and the reserve');
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
    const { view, calls } = await draw(MAC, {
      responses: { 'PATCH /api/memory': MAC } });
    const input = field(view);
    input.value = '24576';
    input.dispatchEvent(new window.Event('input', { bubbles: true }));
    saveButton(view).click();
    await settle();
    const sent = calls.find((call) => call.method === 'PATCH');
    assert.ok(sent, 'nothing was saved');
    assert.deepEqual(JSON.parse(sent.body), { reserve_mb: 24576 });
  });

  it('leaves the memory lines out when they could not be read', async () => {
    // No pools means the machine could not answer — not that it has no room.
    // A row of zeroes would be a claim, and the wrong one. The chip is still
    // worth saying.
    const { view } = await draw({ unified: false, pools: [], capacity_mb: 0 });
    assert.match(view.textContent, /NVIDIA RTX PRO 4500/);
    assert.equal(view.querySelector('input[type="number"]'), null);
  });

  it('draws nothing at all when there is nothing to say', async () => {
    const { view } = await draw({ pools: [] }, { chip: {}, host: {} });
    assert.equal(view.querySelector('.section.machine'), null);
  });
});
