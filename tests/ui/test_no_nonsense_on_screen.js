// Words that are never an answer.
//
// "null", "undefined" and "NaN" are what JavaScript prints when a value was
// not there and nobody checked. All three have reached this interface:
// `replaceChildren` turns a null child into the text "null", and
// `Math.round(undefined)` is NaN, which renders as the word.
//
// The last one was found on the Mac, on screen, reading "NaN / 131072 MB" —
// the browser had a new copy of a view and the manager, started half an hour
// earlier, was still sending a figure under its previous name. The page and
// the server are never replaced in the same instant, so a view must survive
// being handed the shape it was given last week.
//
// These render each view against an answer with everything missing, which is
// the worst version of that and also of a server that failed to read the
// machine.

import assert from 'node:assert/strict';
import { after, before, beforeEach, describe, it } from 'node:test';

import { installDom, settle } from './support/dom.js';

const NONSENSE = ['null', 'undefined', 'NaN'];

// Every endpoint any view asks for, answered with something shaped roughly
// right and empty of everything else.
const EMPTY = {
  '/api/instances': [{ id: 'entry', engine: 'llamacpp', model_id: 'x/y',
                       port: 8080, params: {}, running: false }],
  '/api/models': [{ id: 'x/y', name: 'y', format: 'gguf' }],
  '/api/settings': { title: 'AI-Lab', engines: [{ id: 'llamacpp', name: 'llama.cpp',
                                                  available: true, formats: ['gguf'],
                                                  params: [] }],
                     repositories: [{ id: 'r', name: 'R', path: '/p', format: 'gguf' }],
                     accelerator: {}, host: {}, memory: { pools: [{ name: 'machine',
                                                                    kind: 'unified' }] } },
  '/api/gateway': { loaded: [{ instance_id: 'entry' }], memory: { pools: [{}] },
                    card: {}, queue_runs: [], recent: [], shapes: [],
                    waiting_for: [] },
  '/api/downloads': [],
  '/api/installs': [],
  '/api/hf/search': { results: [], hidden: 0 },
  '/api/hf/sets': [],
};

const VIEWS = ['runtime', 'library', 'settings', 'gateway'];

describe('words that are never an answer', () => {
  let dom;
  before(() => { dom = installDom({}); });
  after(() => dom.restore && dom.restore());
  beforeEach(() => { document.body.innerHTML = ''; });

  for (const name of VIEWS) {
    it(`${name} says none of them when everything is missing`, async () => {
      const context = installDom(EMPTY);
      const view = await import(`../../ai_lab/web/js/views/${name}.js?${Math.random()}`);
      await view.render(context.view);
      await settle();
      if (view.stopRefreshing) view.stopRefreshing();
      for (const word of NONSENSE) {
        assert.equal(context.view.textContent.includes(word), false,
                     `${name} printed "${word}": ${context.view.textContent}`);
      }
    });
  }
});
