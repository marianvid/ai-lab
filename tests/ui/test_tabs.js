// The frame around every page: the warning under the header, and a tab click that
// feels like a click even when its API takes seconds.

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { installDom, settle } from './support/dom.js';


describe('the warning on every page', () => {
  it('is shown under the header while the token requirement does not hold', async () => {
    installDom({ '/api/session': { login_required: false, logged_in: true, token_holds: false },
                 '/api/instances': [], '/api/models': [], '/api/installs': [],
                 '/api/settings': { title: 'AI-Lab', engines: [], repositories: [],
                                    accelerator: {}, host: {} } });
    global.localStorage = { getItem: () => null, setItem: () => {} };
    const theme = document.createElement('button');
    theme.id = 'theme';
    document.body.append(theme);
    await import(`../../ai_lab/web/js/app.js?${Math.random()}`);
    await settle();
    const warning = document.getElementById('token-warning');
    assert.ok(warning, 'no warning drawn');
    assert.match(warning.textContent, /set a password on Access/);
  });
});

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
