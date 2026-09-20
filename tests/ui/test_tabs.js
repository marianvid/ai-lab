// A tab click must feel like a click even when its API takes seconds.

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { installDom, settle } from './support/dom.js';


describe('switching pages', () => {
  it('selects the tab and shows progress before its data arrives', async () => {
    let finishStorage;
    const storage = new Promise((resolve) => { finishStorage = resolve; });
    const context = installDom({
      '/api/instances': [], '/api/models': [], '/api/installs': [],
      '/api/downloads': [],
      '/api/settings': { title: 'AI-Lab', engines: [], repositories: [],
                         accelerator: {}, host: {} },
      '/api/storage': storage,
    });
    global.localStorage = {
      getItem: () => null,
      setItem: () => {},
    };
    const theme = document.createElement('button');
    theme.id = 'theme';
    document.body.append(theme);
    await import(`../../ai_lab/web/js/app.js?${Math.random()}`);
    await settle();

    const tab = [...document.querySelectorAll('#tabs button')]
      .find((button) => button.textContent === 'Storage');
    tab.click();

    // These are synchronous effects of the click, before the promise above is
    // allowed to resolve.
    assert.ok(tab.classList.contains('active'));
    assert.match(context.view.textContent, /Loading Storage…/);
    assert.equal(context.view.firstChild.getAttribute('aria-busy'), 'true');
    assert.ok(document.body.classList.contains('view-loading'));

    finishStorage({ items: [], recoverable_bytes: 0 });
    await settle();
    assert.doesNotMatch(context.view.textContent, /Loading Storage…/);
    assert.equal(context.view.firstChild.hasAttribute('aria-busy'), false);
    assert.equal(document.body.classList.contains('view-loading'), false);
  });
});
