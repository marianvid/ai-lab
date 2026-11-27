// A page that redraws itself on a timer never throws away what is typed.
//
// The Gateway page redraws every 5 s and the GPU health page every 15 s. A
// value typed and not yet saved used to be replaced by the saved one at the
// next redraw (found in the audit, 2026-11-27).

import assert from 'node:assert/strict';
import { describe, it, mock } from 'node:test';

import { button, installDom, settle } from './support/dom.js';

const GATEWAY = {
  loaded: [], busy: false, holder: null, in_flight: 0, switching: false, waiting: 0,
  waiting_for: [], longest_wait_s: 0, max_waiting: 150, first_byte_s: 120,
  between_bytes_s: 30, stall_s: 60, safety_s: 1800, default_max_tokens: 4096,
  requests_per_minute: 0, switches: 0, average_wait_s: 0, switching_share: 0,
  last_error: '', shapes: [], card: {}, memory: { pools: [] }, queue_runs: [], recent: [],
};

describe('a field being typed in', () => {
  it('counts as being edited while it differs from what was drawn, or has the focus', async () => {
    const context = installDom({});
    const { beingEdited, element } = await import(`../../ai_lab/web/js/format.js?${Math.random()}`);
    const box = element('input', { type: 'number', value: '120' });
    const tick = element('input', { type: 'checkbox' });
    const form = element('div', {}, [box, tick, element('button', { text: 'Save' })]);
    context.document.body.append(form);
    assert.equal(beingEdited(form), false);
    box.value = '300';
    assert.equal(beingEdited(form), true, 'typed and not saved');
    box.value = '120';
    tick.checked = true;
    assert.equal(beingEdited(form), true, 'a box ticked and not saved');
    tick.checked = false;
    box.focus();
    assert.equal(beingEdited(form), true, 'the focus is in a field');
    form.querySelector('button').focus();
    assert.equal(beingEdited(form), false, 'a button with the focus is not typing');
    context.window.close();
  });
});

describe('the Gateway page', () => {
  it('skips its timed redraw while a limit is typed and not saved', async () => {
    mock.timers.enable({ apis: ['setInterval'] });
    const context = installDom({ '/api/gateway': GATEWAY });
    // The page uses the window's timer; hand it the mocked one.
    context.window.setInterval = setInterval;
    context.window.clearInterval = clearInterval;
    const view = await import(`../../ai_lab/web/js/views/gateway.js?${Math.random()}`);
    try {
      await view.render(context.view);
      await settle();
      const box = context.view.querySelector('input[type=number]');
      box.value = '300';
      box.dispatchEvent(new context.window.Event('input'));
      const asked = context.calls.filter((call) => call.path === '/api/gateway').length;
      mock.timers.tick(5000);
      await settle();
      assert.equal(context.calls.filter((call) => call.path === '/api/gateway').length, asked,
                   'redrew under the typing');
      assert.equal(context.view.querySelector('input[type=number]').value, '300');
      box.value = box.defaultValue;
      mock.timers.tick(5000);
      await settle();
      assert.equal(context.calls.filter((call) => call.path === '/api/gateway').length, asked + 1,
                   'the redraw comes back once nothing is typed');
    } finally {
      view.stopRefreshing();
      mock.timers.reset();
      context.window.close();
    }
  });

  it('shows the own waits of each task and sends a changed one', async () => {
    const context = installDom({
      '/api/gateway': { ...GATEWAY, task_timeouts: {
        ocr: { first_byte_s: 60, between_bytes_s: 30 },
        'music-generation': { first_byte_s: 1800, between_bytes_s: 60 } } },
      'PATCH /api/gateway': GATEWAY,
    });
    const view = await import(`../../ai_lab/web/js/views/gateway.js?${Math.random()}`);
    await view.render(context.view);
    await settle();
    view.stopRefreshing();
    const row = [...context.view.querySelectorAll('label.field')]
      .find((item) => item.textContent.includes('ocr (s)'));
    assert.ok(row, 'a row for ocr');
    assert.match(row.title, /first byte, then between two bytes/);
    const [first] = row.querySelectorAll('input');
    first.value = '90';
    first.dispatchEvent(new context.window.Event('input'));
    const save = button(context.view, 'Save');
    assert.equal(save.disabled, false);
    save.click();
    await settle();
    const sent = JSON.parse(context.calls.find((call) => call.method === 'PATCH').body);
    assert.deepEqual(sent.task_timeouts.ocr, { first_byte_s: 90, between_bytes_s: 30 });
    context.window.close();
  });

  it('shows a task with one wait of its own with the other taken from above', async () => {
    const context = installDom({ '/api/gateway': { ...GATEWAY, task_timeouts: {
      ocr: { first_byte_s: 60 } } } });
    const view = await import(`../../ai_lab/web/js/views/gateway.js?${Math.random()}`);
    await view.render(context.view);
    await settle();
    view.stopRefreshing();
    const row = [...context.view.querySelectorAll('label.field')]
      .find((item) => item.textContent.includes('ocr (s)'));
    const [first, between] = row.querySelectorAll('input');
    assert.deepEqual([first.value, between.value], ['60', '30']);
    assert.equal(button(context.view, 'Save').disabled, true);
    context.window.close();
  });

  it('keeps what was typed when the save is refused', async () => {
    const context = installDom({
      '/api/gateway': GATEWAY,
      'PATCH /api/gateway': { __status: 400, error: 'first_byte_s must be between 1 and 3600' },
    });
    const view = await import(`../../ai_lab/web/js/views/gateway.js?${Math.random()}`);
    await view.render(context.view);
    await settle();
    view.stopRefreshing();
    const box = context.view.querySelector('input[type=number]');
    box.value = '0';
    box.dispatchEvent(new context.window.Event('input'));
    button(context.view, 'Save').click();
    await settle();
    assert.equal(context.view.querySelector('input[type=number]').value, '0');
    context.window.close();
  });
});
