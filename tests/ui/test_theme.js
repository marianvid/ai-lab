// The light/dark control.

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { installDom, settle } from './support/dom.js';

// A fresh page, optionally keeping what the last one stored. Building a new
// DOM gives a new localStorage, which would make "remembers the choice"
// trivially pass by forgetting.
async function load({ keepStorage = false } = {}) {
  const previous = keepStorage && global.localStorage
    ? { ...global.localStorage }
    : null;
  const context = installDom({});
  global.localStorage = context.window.localStorage;
  global.localStorage.clear();
  if (previous) {
    Object.entries(previous).forEach(([key, value]) =>
      global.localStorage.setItem(key, value));
  }
  const module = await import(`../../ai_lab/web/js/theme.js?${Math.random()}`);
  const button = context.document.createElement('button');
  context.document.body.append(button);
  module.installTheme(button);
  return { ...context, module, button, root: context.document.documentElement };
}

describe('the theme control', () => {
  it('starts from whatever the system is already doing', async () => {
    // So the first click changes the page rather than confirming it.
    const { root, button } = await load();
    assert.ok(['light', 'dark'].includes(root.getAttribute('data-theme')));
    assert.ok(['Light', 'Dark'].includes(button.textContent));
  });

  it('switches between the two', async () => {
    const { root, button } = await load();
    const first = root.getAttribute('data-theme');
    button.click();
    assert.notEqual(root.getAttribute('data-theme'), first);
    button.click();
    assert.equal(root.getAttribute('data-theme'), first);
  });

  it('remembers the choice', async () => {
    const first = await load();
    first.button.click();
    const chosen = first.root.getAttribute('data-theme');

    const second = await load({ keepStorage: true });
    assert.equal(second.root.getAttribute('data-theme'), chosen);
    assert.equal(second.module.currentTheme(), chosen);
  });

  it('ignores a stored value it does not understand', async () => {
    const { window } = installDom({});
    global.localStorage = window.localStorage;
    global.localStorage.setItem('ai-lab-theme', 'neon');
    const module = await import(`../../ai_lab/web/js/theme.js?${Math.random()}`);
    assert.ok(['light', 'dark'].includes(module.currentTheme()));
  });
});
