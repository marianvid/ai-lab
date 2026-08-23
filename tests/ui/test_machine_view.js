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
  available_mb: 3677 + 35803,
  held_by_models_mb: 28946,
  pools: [
    { name: 'card', kind: 'dedicated', total_mb: 32623, used_mb: 28946,
      used_by_models_mb: 28946, reserve_mb: 0, free_mb: 3677,
      for_models_mb: 32623, available_mb: 3677 },
    { name: 'machine', kind: 'dedicated', total_mb: 49152, used_mb: 5157,
      used_by_models_mb: 0, reserve_mb: 8192, free_mb: 43995,
      for_models_mb: 40960, available_mb: 35803 },
  ],
};

const MAC = {
  unified: true, reserve_mb: 16384,
  available_mb: 80088, held_by_models_mb: 21000,
  pools: [
    { name: 'machine', kind: 'unified', total_mb: 131072, used_mb: 34600,
      used_by_models_mb: 21000, reserve_mb: 16384, free_mb: 96472,
      for_models_mb: 114688, available_mb: 80088 },
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
    assert.deepEqual(listed[2], { label: 'RAM', value: '40960 / 49152 MB' });
  });

  it('does the subtraction for you where something is held back', async () => {
    // 49152 less the 8192 on the line below. Every figure is on screen and
    // they check against each other, so nothing has to be worked out and
    // nothing has to be believed.
    const { view } = await draw(LINUX);
    assert.equal(rows(view)[2].value, '40960 / 49152 MB');
  });

  it('shows one figure where nothing is held back', async () => {
    // 32623 / 32623 is two ways of saying the same thing and reads like a
    // fault. Decided by whether there is a reserve, not by which pool it is,
    // so a card that one day holds something back gets the pair by itself.
    const { view } = await draw(LINUX);
    assert.equal(rows(view)[1].value, '32623 MB');
  });

  it('neither figure moves when a model loads', async () => {
    // This card is what the machine is. 28.9 GB of that card is held right
    // now, and that belongs on the Gateway page.
    const { view } = await draw(LINUX);
    assert.equal(view.textContent.includes('3677'), false,
                 'showing what is free right now');
    assert.equal(view.textContent.includes('35803'), false,
                 'showing what is free right now');
  });

  it('lists the card whole, not what is left of it', async () => {
    // 28.9 GB of this card is held by a running model. That is a fact about
    // now and belongs on the Gateway page; this line says how big the card is.
    const { view } = await draw(LINUX);
    assert.match(rows(view)[1].value, /32623/);
    assert.equal(view.textContent.includes('3677'), false,
                 'this card is showing what is free rather than what there is');
  });

  it('puts the reserve last, where it explains the gap above it', async () => {
    // It is the difference between the two figures on the RAM line, so it
    // reads as the working rather than as a fact on its own.
    const { view } = await draw(LINUX);
    const listed = rows(view);
    assert.equal(listed[listed.length - 1].label, 'Reserved for the system');
  });

  it('says unified memory once, and calls it that', async () => {
    const { view } = await draw(MAC, { chip: APPLE });
    const listed = rows(view);
    assert.deepEqual(listed[1],
                     { label: 'Unified memory', value: '114688 / 131072 MB' });
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

  it('works it out when the server is older than this page', async () => {
    // Found on the Mac: the browser had this file and the manager, started
    // half an hour earlier, was still sending the figure under its previous
    // name. `Math.round(undefined)` is NaN, and NaN reaches the screen as the
    // word "NaN". The page and the server are never replaced in the same
    // instant, so this is the ordinary case during a deployment.
    const older = {
      unified: true, reserve_mb: 8192, available_mb: 88280,
      pools: [{ name: 'machine', kind: 'unified', total_mb: 131072,
                used_mb: 34600, used_by_models_mb: 21000, reserve_mb: 8192,
                free_mb: 96472, available_mb: 88280 }],   // no for_models_mb
    };
    const { view } = await draw(older, { chip: APPLE });
    assert.equal(view.textContent.includes('NaN'), false, view.textContent);
    assert.deepEqual(rows(view)[1],
                     { label: 'Unified memory', value: '122880 / 131072 MB' });
  });

  it('never prints NaN, whatever the server sends', async () => {
    // Every field missing at once, which is what a much older manager, or a
    // failed read, looks like.
    const nonsense = {
      unified: false, reserve_mb: undefined,
      pools: [
        { name: 'card', kind: 'dedicated' },
        { name: 'machine', kind: 'dedicated', total_mb: 'not a number',
          reserve_mb: null },
      ],
    };
    const { view } = await draw(nonsense);
    assert.equal(view.textContent.includes('NaN'), false, view.textContent);
    assert.equal(view.textContent.includes('undefined'), false, view.textContent);
    assert.equal(view.textContent.includes('null'), false, view.textContent);
  });

  it('leaves out a pool it cannot say anything true about', async () => {
    const half = {
      unified: false, reserve_mb: 8192,
      pools: [
        { name: 'card', kind: 'dedicated' },                    // no total
        { name: 'machine', kind: 'dedicated', total_mb: 49152,
          reserve_mb: 8192, for_models_mb: 40960 },
      ],
    };
    const { view } = await draw(half);
    const listed = rows(view);
    assert.equal(listed.filter((r) => r.label === 'VRAM').length, 0);
    assert.ok(listed.some((r) => r.value === '40960 / 49152 MB'));
  });

  it('leaves the memory lines out when they could not be read', async () => {
    // No pools means the machine could not answer — not that it has no room.
    // A row of zeroes would be a claim, and the wrong one. The chip is still
    // worth saying.
    const { view } = await draw({ unified: false, pools: [] });
    assert.match(view.textContent, /NVIDIA RTX PRO 4500/);
    assert.equal(view.querySelector('input[type="number"]'), null);
  });

  it('draws nothing at all when there is nothing to say', async () => {
    const { view } = await draw({ pools: [] }, { chip: {}, host: {} });
    assert.equal(view.querySelector('.section.machine'), null);
  });
});
