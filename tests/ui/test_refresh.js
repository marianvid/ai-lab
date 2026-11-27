// When the page redraws itself, and when it deliberately does not.
//
// It used to redraw on a timer, which threw away whatever was half-typed or
// selected. Now it redraws when the server says something changed, and waits
// if somebody is in the middle of something.

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { installDom, settle } from './support/dom.js';

const INSTANCE = {
  id: 'qwen-coder', name: 'Coding', engine: 'llamacpp',
  model_id: 'gguf/qwen/qwen', port: 8080, params: { context_size: 32768 },
  running: false, enabled: true, pid: null, ready: false, web_ui: false,
  last_operation: null,
};

async function boot(beforeImport = () => {}) {
  const context = installDom({
    '/api/instances': [INSTANCE],
    '/api/models': [],
    '/api/settings': { title: 'AI-Lab', engines: [], repositories: [],
                       accelerator: {}, host: {} },
  });

  // A stand-in for the connection, so a test can deliver an event itself.
  let handler = null;
  global.EventSource = class {
    constructor() { handler = (data) => this.onmessage({ data: JSON.stringify(data) }); }
    close() {}
  };

  beforeImport(context);
  const { onChange, startEventStream } = await import(
    `../../ai_lab/web/js/events.js?${Math.random()}`);
  startEventStream();
  return { ...context, onChange, send: (data) => handler(data) };
}

describe('refreshing', () => {
  it('does nothing at all while the machine is quiet', async () => {
    // The whole point: a page nobody is changing stays exactly as it is.
    // The old timer redrew every few seconds, so waiting a moment could not
    // see it come back (the reviewer's item 6): no repeating timer may be set.
    const intervals = [];
    const spied = [];
    const context = await boot(({ window }) => {
      for (const host of [globalThis, window]) {
        const real = host.setInterval;
        spied.push([host, real]);
        host.setInterval = (callback, ms, ...rest) => {
          intervals.push(ms);
          return real.call(host, callback, ms, ...rest);
        };
      }
    });
    let called = 0;
    context.onChange(() => { called += 1; });
    await settle();
    for (const [host, real] of spied) host.setInterval = real;
    assert.deepEqual(intervals, [], 'a repeating timer was set');
    assert.equal(called, 0, 'something redrew without being told to');
    context.window.close();
  });

  it('reacts when the server says something moved', async () => {
    const context = await boot();
    const topics = [];
    context.onChange((event) => topics.push(event.topic));
    context.send({ kind: 'change', topic: 'instances' });
    await settle();
    assert.deepEqual(topics, ['instances']);
  });

  it('keeps the three kinds of event apart', async () => {
    const context = await boot();
    const { onProgress, onLog } = await import(
      `../../ai_lab/web/js/events.js?${Math.random()}`);
    const seen = { change: 0, progress: 0, log: 0 };
    context.onChange(() => { seen.change += 1; });
    onProgress(() => { seen.progress += 1; });
    onLog(() => { seen.log += 1; });

    context.send({ kind: 'change', topic: 'models' });
    await settle();
    assert.deepEqual(seen, { change: 1, progress: 0, log: 0 },
                     'a change notice reached the wrong listeners');
  });

  it('ignores anything it cannot read rather than falling over', async () => {
    const context = await boot();
    let called = 0;
    context.onChange(() => { called += 1; });
    assert.doesNotThrow(() => {
      global.EventSource.prototype.onmessage;
      context.send({ kind: 'nonsense' });
    });
    await settle();
    assert.equal(called, 0);
  });
});
