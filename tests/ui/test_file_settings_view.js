// The settings kept only in config.json, read-only on their pages.

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { installDom, settle } from './support/dom.js';

const PROFILES = {
  key: 'images', page: 'library', title: 'Image profiles',
  help: 'The named image profiles. Changed in config.json under images.',
  restart: 'Read when AI-Lab starts.',
  value: { workflow_root: '/w', profiles: { flux: { model: 'images/flux', steps: [1, 2] } } },
};

describe('a section kept only in the file', () => {
  it('is shown read-only, with what it is and where it is changed, one line per value', async () => {
    const context = installDom({ '/api/file-settings': { sections: [PROFILES] } });
    const { fileSettingsSections } = await import(
      `../../ai_lab/web/js/views/file_settings.js?${Math.random()}`);
    const [node] = await fileSettingsSections('library');
    context.view.append(node);
    await settle();
    assert.match(node.textContent, /Image profiles \(read-only\)/);
    assert.match(node.textContent, /Changed in config.json under images/);
    assert.match(node.textContent, /Read when AI-Lab starts/);
    assert.match(node.textContent, /profiles\.flux\.model\s*images\/flux/);
    assert.match(node.textContent, /profiles\.flux\.steps\s*1, 2/);
    assert.equal(node.querySelector('input'), null);
    assert.ok(context.calls.some((call) => call.path === '/api/file-settings?page=library'));
    context.window.close();
  });

  it('says so when they could not be read', async () => {
    const context = installDom({ '/api/file-settings': { __status: 500, error: 'down' } });
    const { fileSettingsSections } = await import(
      `../../ai_lab/web/js/views/file_settings.js?${Math.random()}`);
    const [node] = await fileSettingsSections('lanes');
    assert.match(node.textContent, /could not be read/);
    context.window.close();
  });
});
