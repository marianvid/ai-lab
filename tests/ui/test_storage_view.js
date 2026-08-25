import assert from 'node:assert/strict';
import { after, before, beforeEach, describe, it } from 'node:test';
import { installDom, settle } from './support/dom.js';

const ANSWERS = {
  '/api/storage': { recoverable_bytes: 4096, items: [{
    id: 'uv', name: 'Package downloads', path: '/var/lib/ai-lab/.cache/uv',
    kind: 'cache', exists: true, size_bytes: 4096,
    description: 'Downloaded packages kept for reuse.',
  }] },
  '/api/installs': [],
  '/api/settings': { engines: [] },
};

describe('the Storage page', () => {
  let dom;
  before(() => { dom = installDom({}); });
  after(() => dom.restore && dom.restore());
  beforeEach(() => { document.body.innerHTML = ''; });

  it('shows reclaimable non-model files', async () => {
    const context = installDom(ANSWERS);
    const view = await import(`../../ai_lab/web/js/views/storage.js?${Math.random()}`);
    await view.render(context.view);
    assert.match(context.view.textContent, /Package downloads/);
    assert.doesNotMatch(context.view.textContent, /Delete model/);
  });

  it('asks before clearing and sends only the configured id', async () => {
    const context = installDom({ ...ANSWERS,
      'DELETE /api/storage/uv': { items: [], recoverable_bytes: 0 } });
    const view = await import(`../../ai_lab/web/js/views/storage.js?${Math.random()}`);
    await view.render(context.view);
    [...context.view.querySelectorAll('button')].find((b) => b.textContent === 'Clear').click();
    await settle();
    assert.ok(document.querySelector('dialog.confirm'));
    [...document.querySelectorAll('dialog.confirm button')]
      .find((b) => b.textContent === 'Clear').click();
    await settle();
    assert.ok(context.calls.some((call) => call.method === 'DELETE'
      && call.path === '/api/storage/uv'));
  });
});
