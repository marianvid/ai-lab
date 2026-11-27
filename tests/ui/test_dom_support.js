// The test page itself: an unanswered path is caught, not answered 200 [].

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { installDom } from './support/dom.js';

describe('the test page', () => {
  it('records a path no test answers, so the test fails', async () => {
    const context = installDom({ '/api/known': { fine: true } });
    await fetch('/api/known');
    await fetch('/api/misspelt?x=1');
    await fetch('/api/settings', { method: 'PATCH' });
    assert.deepEqual(context.unmocked, ['GET /api/misspelt', 'PATCH /api/settings']);
    context.unmocked.length = 0;            // caught here on purpose: let this test pass
    context.window.close();
  });

  it('catches a stray path on the first of two pages a test builds', async () => {
    // The reviewer's probe on 98a3818: only the last page used to be checked.
    const first = installDom({});
    await fetch('/api/stray');
    const second = installDom({ '/api/known': {} });
    await fetch('/api/known');
    assert.deepEqual([first.unmocked, second.unmocked], [['GET /api/stray'], []]);
    first.unmocked.length = 0;              // caught here on purpose: let this test pass
  });

  it('answers the reads every page makes in passing with their real empty shapes', async () => {
    installDom({});
    const read = async (path) => (await fetch(path)).json();
    assert.deepEqual(await read('/api/setting-fields'), { groups: {}, fields: [] });
    assert.deepEqual(await read('/api/file-settings'), { sections: [] });
    assert.equal((await read('/api/session')).token_holds, true);
    assert.deepEqual(await read('/api/builds'), []);
  });

  it('lets the reads every page makes in passing, and the ones a test names, through', async () => {
    const context = installDom({}, { quiet: ['/api/not-cared-about'] });
    await fetch('/api/setting-fields');
    await fetch('/api/not-cared-about');
    assert.deepEqual(context.unmocked, []);
    context.window.close();
  });
});
