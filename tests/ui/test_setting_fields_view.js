// The settings that once lived only in the configuration file, on the Settings page.

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { installDom, button, settle } from './support/dom.js';

const LOAD = {
  key: 'limits.runtime.load_s', group: 'loading', label: 'Model load', unit: 's',
  default: 900, low: 1, high: 604800, kind: 'number', restart: true,
  help: 'How long the engine may take to answer once started.',
  restart_note: 'Takes effect when AI-Lab is next started.', value: 900, saved: false,
};
const PATH = {
  key: 'limits.builds.system_path', group: 'updates', label: 'Search path for installs',
  unit: '', default: '/usr/bin:/bin', low: null, high: null, kind: 'text', restart: true,
  help: 'Where the programs are looked for.', restart_note: 'Takes effect when AI-Lab is next started.',
  value: '/usr/bin:/bin', saved: false,
};
const PORT = {
  key: 'limits.first_port', group: 'loading', label: 'First port offered', unit: '',
  default: 8080, low: 1, high: 65535, kind: 'integer', restart: false,
  help: 'The first port suggested.', restart_note: '', value: 8080, saved: false,
};
const FIELDS = {
  groups: { loading: 'Loading models', updates: 'Engine updates' },
  fields: [LOAD, PORT, PATH],
};

async function renderPage() {
  const context = installDom({
    '/api/settings': { title: 'AI-Lab', host: {}, accelerator: {}, repositories: [],
                       engines: [], models_root: '/models' },
    '/api/setting-fields': FIELDS,
    'PATCH /api/setting-fields': FIELDS,
  });
  const { render } = await import(`../../ai_lab/web/js/views/settings.js?${Math.random()}`);
  await render(context.view);
  await settle();
  return context;
}

function namedSection(view, name) {
  const heading = [...view.querySelectorAll('.section > h3')]
    .find((item) => item.textContent.trim() === name);
  return heading && heading.parentElement;
}

describe('the settings from the configuration file', () => {
  it('shows each group as a section of its own, with every field in it', async () => {
    const { view } = await renderPage();
    const loading = namedSection(view, 'Loading models');
    assert.ok(loading, 'the Loading models section');
    assert.match(loading.textContent, /Model load \(s\)/);
    assert.match(loading.textContent, /First port offered/);
    assert.ok(namedSection(view, 'Engine updates').querySelector('input[type=text]'));
  });

  it('explains a field when pointed at: meaning, default, and when it takes effect', async () => {
    const { view } = await renderPage();
    const field = namedSection(view, 'Loading models').querySelector('label.field');
    assert.match(field.title, /How long the engine may take/);
    assert.match(field.title, /Default: 900 s\./);
    assert.match(field.title, /next started/);
    assert.match(field.textContent, /after a restart/);
  });

  it('says nothing about a restart for a field that takes effect at once', async () => {
    const { view } = await renderPage();
    const fields = [...namedSection(view, 'Loading models').querySelectorAll('label.field')];
    const port = fields.find((item) => item.textContent.includes('First port'));
    assert.doesNotMatch(port.textContent, /after a restart/);
  });

  it('saves only what was changed, and only once something was', async () => {
    const { view, calls, window } = await renderPage();
    const loading = namedSection(view, 'Loading models');
    const save = button(loading, 'Save');
    assert.equal(save.disabled, true);
    const box = loading.querySelector('input[type=number]');
    box.value = '1800';
    box.dispatchEvent(new window.Event('input'));
    assert.equal(save.disabled, false);
    save.click();
    await settle();
    const sent = calls.find((call) => call.method === 'PATCH');
    assert.deepEqual(JSON.parse(sent.body), { 'limits.runtime.load_s': 1800 });
  });
});
